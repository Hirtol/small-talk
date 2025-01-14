use crate::{
    config::SharedConfig,
    system::{
        tts_backends::{BackendTtsRequest, BackendTtsResponse, TtsBackend, TtsResult},
        voice_manager::{VoiceDestination, VoiceManager, VoiceReference},
        CharacterName, CharacterVoice, Gender, TtsResponse, TtsSystemHandle, Voice, VoiceLine,
    },
};
use eyre::{Context, ContextCompat};
use itertools::Itertools;
use path_abs::PathOps;
use rand::{prelude::IteratorRandom, thread_rng};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};
use serde::de::Error;
use tokio::{
    sync::{broadcast::error::RecvError, Mutex},
};
use tokio::sync::broadcast;
use crate::system::{PostProcessing, TtsModel, TtsVoice};
use crate::system::config::TtsSystemConfig;
use crate::system::error::GameSessionError;
use crate::system::playback::PlaybackEngineHandle;
use crate::system::voice_manager::FsVoiceData;

const CONFIG_NAME: &str = "config.json";
const LINES_NAME: &str = "lines.json";

pub type GameSessionActorHandle = tokio::sync::mpsc::UnboundedSender<GameSessionMessage>;
type GameResult<T> = std::result::Result<T, GameSessionError>;

#[derive(Clone)]
pub struct GameSessionHandle {
    send: tokio::sync::mpsc::Sender<GameSessionMessage>,
    pub playback: PlaybackEngineHandle,
    data: Arc<GameSharedData>,
    voice_man: Arc<VoiceManager>,
}

impl GameSessionHandle {
    #[tracing::instrument(skip(config, tts, voice_man))]
    pub async fn new(
        game_name: &str,
        voice_man: Arc<VoiceManager>,
        tts: TtsBackend,
        config: SharedConfig,
    ) -> eyre::Result<Self> {
        tracing::info!("Starting: {}", game_name);
        // Small amount before we exert back-pressure
        let (send, recv) = tokio::sync::mpsc::channel(10);
        let (send_b, recv_b) = broadcast::channel(100);
        let (notify_send, notify_recv) = tokio::sync::mpsc::channel(1);
        let (game_data, line_cache) = GameSessionActor::create_or_load_from_file(game_name, &config.dirs).await?;
        let line_cache = Arc::new(Mutex::new(line_cache));
        let shared_queue = Arc::new(Mutex::new(VecDeque::new()));

        let shared_data = Arc::new(GameSharedData {
            config,
            voice_manager: voice_man.clone(),
            game_data,
            queue: shared_queue,
            line_cache,
        });

        let actor = GameSessionActor {
            recv,
            b_recv: recv_b,
            data: shared_data.clone(),
            notify: notify_send,
        };
        let queue_actor = GameQueueActor {
            broadcast: send_b,
            tts,
            data: shared_data.clone(),
            notify: notify_recv,
            generations_count: 0,
        };

        tokio::task::spawn(async move {
            if let Err(e) = actor.run().await {
                tracing::error!("GameSession stopped with error: {e}");
            }
        });
        tokio::task::spawn(async move {
            if let Err(e) = queue_actor.run().await {
                tracing::error!("GameSessionQueue stopped with error: {e}");
            }
        });
        
        let playback = PlaybackEngineHandle::new(send.clone()).await?;

        Ok(Self {
            send,
            playback,
            data: shared_data,
            voice_man,
        })
    }
    
    /// Check whether this session is still alive, or was somehow taken offline.
    pub fn is_alive(&self) -> bool {
        !self.send.is_closed()
    }
    
    /// Request a handle to the stream of [TtsResponse]s from the generation queue for this session.
    pub async fn broadcast_handle(&self) -> eyre::Result<broadcast::Receiver<Arc<TtsResponse>>> {
        let (send, recv) = tokio::sync::oneshot::channel();
        self.send.send(GameSessionMessage::BroadcastHandle(send)).await?;

        Ok(recv.await?)
    }
    
    /// Force the character mapping to use the given voice.
    pub async fn force_character_voice(&self, character: CharacterName, voice: VoiceReference) -> eyre::Result<()> {
        tracing::debug!(?character, ?voice, "Forced voice mapping");
        self.data.game_data.character_map.pin().insert(character, voice);
        Ok(())
    }
    
    /// Return all current character voice mappings
    pub async fn character_voices(&self) -> eyre::Result<HashMap<CharacterName, VoiceReference>> {
        Ok(self.data.game_data.character_map.pin().iter().map(|(c, v)| (c.clone(), v.clone())).collect())
    }

    /// Return all available voices for this particular game, including global voices.
    pub async fn available_voices(&self) -> eyre::Result<Vec<FsVoiceData>> {
        Ok(self.voice_man.get_voices(&self.data.game_data.game_name))
    }
    
    /// Will add the given items onto the queue for TTS generation.
    /// 
    /// These items will be prioritised over previous queue items
    pub async fn add_all_to_queue(&self, items: Vec<VoiceLine>) -> eyre::Result<()> {
        Ok(self.send.send(GameSessionMessage::AddToQueue(items)).await?)
    }
    
    /// Request a single voice line with the highest priority.
    /// 
    /// If this future is dropped prematurely the request will still be handled, and the response will be sent on
    /// the [Self::broadcast_handle]. This will be done even if this future is _not_ dropped.
    #[tracing::instrument(skip(self))]
    pub async fn request_tts(&self, request: VoiceLine) -> eyre::Result<Arc<TtsResponse>> {
        let (send, recv) = tokio::sync::oneshot::channel();
        self.send.send(GameSessionMessage::Single(request, send)).await?;

        Ok(recv.await?)
    }
}

#[derive(Debug)]
pub enum GameSessionMessage {
    /// Add all given lines, in order, to the existing queue (first item in Vec will be front of the queue)
    AddToQueue(Vec<VoiceLine>),
    Single(VoiceLine, tokio::sync::oneshot::Sender<Arc<TtsResponse>>),
    /// Request a direct handle to the TtsResponse stream
    BroadcastHandle(tokio::sync::oneshot::Sender<broadcast::Receiver<Arc<TtsResponse>>>)
}

type SingleRequest = (VoiceLine, Option<tokio::sync::oneshot::Sender<Arc<TtsResponse>>>);

pub struct GameSessionActor {
    recv: tokio::sync::mpsc::Receiver<GameSessionMessage>,
    b_recv: broadcast::Receiver<Arc<TtsResponse>>,
    data: Arc<GameSharedData>,

    notify: tokio::sync::mpsc::Sender<()>,
}

struct GameQueueActor {
    broadcast: tokio::sync::broadcast::Sender<Arc<TtsResponse>>,

    tts: TtsBackend,
    data: Arc<GameSharedData>,
    notify: tokio::sync::mpsc::Receiver<()>,

    generations_count: usize,
}

struct GameSharedData {
    config: SharedConfig,

    voice_manager: Arc<VoiceManager>,
    game_data: GameData,
    /// Current queue of requests
    queue: Arc<Mutex<VecDeque<SingleRequest>>>,
    line_cache: Arc<Mutex<LineCache>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct GameData {
    game_name: String,
    character_map: papaya::HashMap<CharacterName, VoiceReference>,
    /// The voices which should be in the random pool of assignment
    male_voices: Vec<VoiceReference>,
    female_voices: Vec<VoiceReference>,
}

impl GameSessionActor {
    pub async fn run(mut self) -> eyre::Result<()> {
        loop {
            let Some(msg) = self.recv.recv().await else {
                tracing::trace!("Stopping GameSessionActor as channel was closed");
                break;
            };
            self.handle_message(msg).await?;
        }

        self.data.save_cache().await?;
        self.data.save_state().await?;

        Ok(())
    }

    #[tracing::instrument(skip(self, message))]
    async fn handle_message(&mut self, message: GameSessionMessage) -> eyre::Result<()> {
        match message {
            GameSessionMessage::AddToQueue(new_lines) => {
                // Reverse iterator to ensure the push_front will leave us with the correct order in the queue
                let mut queue = self.data.queue.lock().await;
                for line in new_lines.into_iter().rev() {
                    queue.retain(|v| v.0 != line && v.1.is_none());
                    queue.push_front((line, None));
                }
                // Notify the queue worker that we have added new items
                let _ = self.notify.try_send(());
            }
            GameSessionMessage::Single(req, response) => {
                // First check if the cache already contains the required data
                let to_send = if let Some(tts_response) = self.data.try_cache_retrieve(&req).await? {
                    Arc::new(tts_response)
                } else {
                    // Otherwise send a priority request to our queue
                    let (snd, rcv) = tokio::sync::oneshot::channel();
                    
                    self.data.queue.lock().await.push_front((req, Some(snd)));
                    let _ = self.notify.try_send(());
                    
                    rcv.await?
                };
                // We don't care if the one who requested it has stopped listening
                let _ = response.send(to_send);
            }
            GameSessionMessage::BroadcastHandle(respond) => {
                let _ = respond.send(self.b_recv.resubscribe());
            }
        }
        Ok(())
    }

    async fn create_or_load_from_file(
        game_name: &str,
        config: &TtsSystemConfig,
    ) -> eyre::Result<(GameData, LineCache)> {
        if tokio::fs::try_exists(config.game_dir(game_name)).await? {
            Self::load_from_dir(config, game_name).await
        } else {
            Self::create(game_name, config).await
        }
    }

    async fn create(game_name: &str, config: &TtsSystemConfig) -> eyre::Result<(GameData, LineCache)> {
        let data = GameData {
            game_name: game_name.into(),
            character_map: Default::default(),
            male_voices: vec![],
            female_voices: vec![],
        };
        let out = serde_json::to_vec_pretty(&data)?;

        let dir = config.game_dir(game_name);
        tokio::fs::create_dir_all(&dir).await?;
        tokio::fs::write(dir.join(CONFIG_NAME), &out).await?;

        Ok((data, Default::default()))
    }

    async fn load_from_dir(conf: &TtsSystemConfig, game_name: &str) -> eyre::Result<(GameData, LineCache)> {
        let dir = conf.game_dir(game_name);
        let game_data = tokio::fs::read(dir.join(CONFIG_NAME)).await?;
        let data = serde_json::from_slice(&game_data)?;
        // If the below doesn't exist we can just re-create it.
        let line_file = conf.game_dir_lines_cache(&dir).join(LINES_NAME);
        let line_cache = tokio::fs::read(line_file).await.unwrap_or_default();
        let lines = serde_json::from_slice(&line_cache).unwrap_or_default();

        Ok((data, lines))
    }
}

impl GameQueueActor {
    #[tracing::instrument(skip(self))]
    pub async fn run(mut self) -> eyre::Result<()> {
        loop {
            let Some(_) = self.notify.recv().await else {
                tracing::trace!(game=?self.data.game_data.game_name, "Stopping GameQueueActor actor as notify channel was closed");
                break;
            };
            
            loop {
                match self.pop_queue().await {
                    Ok(keep_looping) if !keep_looping => break,
                    Err(e) => match e {
                        GameSessionError::Other(e) => {
                            eyre::bail!(e)
                        }
                        GameSessionError::VoiceDoesNotExist { voice } => {
                            tracing::warn!("Ignoring request which requested non-existant voice: {voice}")
                        }
                        GameSessionError::IncorrectGeneration => {
                            tracing::warn!("Skipping line request after too many generation failure")
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn pop_queue(&mut self) -> GameResult<bool> {
        let Some((next_item, respond)) = self.data.queue.lock().await.pop_front() else {
            // Can just return as the queue won't be modified
            return Ok(false);
        };

        let game_response = Arc::new(self.cache_or_request(next_item).await?);
        if let Some(response_channel) = respond {
            // If the consumer drops the other end we don't care
            let _ = response_channel.send(game_response.clone());
        }
        // Don't care whether there are receivers
        let _ = self.broadcast.send(game_response);

        Ok(true)
    }

    /// Either use a cached TTS line, or generate a new one based on the given `voice_line`.
    #[tracing::instrument(skip(self))]
    async fn cache_or_request(&mut self, voice_line: VoiceLine) -> GameResult<TtsResponse> {
        // First check if we have a cache reference
        if let Some(response) = self.data.try_cache_retrieve(&voice_line).await? {
            return Ok(response);
        }
        let voice_to_use = match &voice_line.person {
            TtsVoice::ForceVoice(forced) => {
                forced.clone()
            }
            TtsVoice::CharacterVoice(character) => {
                self.data.map_character(character).await?
            }
        };

        // TODO: Line emotion detection
        let voice = self.data.voice_manager.get_voice(voice_to_use.clone())?;
        let sample = match voice_line.model {
            TtsModel::E2 => voice.try_random_sample(|fs| {
                // The way alltalk/f5 cut off samples when > 15 seconds in length is, frankly, quite shit.
                // So the best way of dealing with that is to just ensure we don't have any samples > 15 seconds
                let Ok(reader) = wavers::Wav::<f32>::from_path(&fs.sample) else {
                    return false;
                };
                reader.duration() < 15
            }).or_else(|_| {
                tracing::debug!("Failed to find a sample which matches with < 15 seconds duration, falling back to normal samples");
                voice.random_sample()
            })?,
            _ => voice.random_sample()?,
        };

        // TODO: Configurable language
        let request = BackendTtsRequest {
            gen_text: voice_line.line.clone(),
            language: "en".to_string(),
            voice_reference: vec![sample],
            speed: None,
        };

        let mut response = None;
        for i in 0..3 {
            let response_gen = self.tts.tts_request(voice_line.model, request.clone()).await?;
            response = if let Some(post) = voice_line.post.as_ref() {
                match self.postprocess(&voice_line, post, response_gen).await {
                    Ok(response) => Some(response),
                    Err(GameSessionError::IncorrectGeneration) => {
                        tracing::trace!(attempt=i, "Failed to generate voice line, retrying");
                        // Retry with a new generation
                        continue;
                    }
                    Err(e) => return Err(e)
                }
            } else {
                Some(response_gen)
            };

            break;
        }
        let Some(response) = response else {
            return Err(GameSessionError::IncorrectGeneration)
        };

        let out = self.transform_response(voice_to_use, voice_line, response).await?;
        // Once in a while save our line cache in case it crashes.
        self.generations_count += 1;
        if self.generations_count > 20 {
            self.data.save_cache().await?
        }

        Ok(out)
    }
    
    /// Perform post-processing on the newly generated raw TTS files.
    /// 
    /// This includes but is not limited to, silence trimming, low/high-pass filters.
    #[tracing::instrument(skip_all)]
    async fn postprocess(&mut self, voice_line: &VoiceLine, post_processing: &PostProcessing, response: BackendTtsResponse) -> Result<BackendTtsResponse, GameSessionError> {
        let should_trim = post_processing.trim_silence;
        let should_normalise = post_processing.normalise;
        
        let timer = std::time::Instant::now();

        let new_audio_path = match response.result.clone() {
            TtsResult::File(temp_path) => {
                // First we check with Whisper (if desired) matches our prompt.
                if let Some(percent) = post_processing.verify_percentage {
                    let score = self.tts.verify_prompt(&temp_path, &voice_line.line).await?;
                    tracing::trace!(?score, "Whisper TTS match");
                    // There will obviously be transcription errors, so we choose a relatively
                    if score < (percent as f32 / 100.0) {
                        return Err(GameSessionError::IncorrectGeneration)
                    }
                }

                // Then we run our audio post-processing to clean it up for human ears.
                tokio::task::spawn_blocking(move || {
                    let mut raw_audio_data = wavers::Wav::<f32>::from_path(&temp_path)?;
                    let mut sample_data: &mut [f32] = &mut raw_audio_data.read()?;

                    let mut made_change = false;

                    if should_trim {
                        // Basically any signal should count.
                        sample_data = super::postprocessing::trim_lead(sample_data, raw_audio_data.n_channels(), 0.01);
                        made_change = true;
                    }
                    if should_normalise {
                        super::postprocessing::loudness_normalise(sample_data, raw_audio_data.sample_rate(), raw_audio_data.n_channels());
                        made_change = true;
                    }

                    if made_change {
                        wavers::write(&temp_path, sample_data, raw_audio_data.sample_rate(), raw_audio_data.n_channels())?;
                    }
                    
                    Ok::<_, eyre::Error>(temp_path)
                }).await.context("Failed to join")??
            }
            TtsResult::Stream => unimplemented!("Streams are not yet supported")
        };
       
        if let Some(rvc) = &post_processing.rvc {
            // TODO
        }
        let took = timer.elapsed();
        tracing::debug!(?took, "Finished post-processing");
        
        Ok(BackendTtsResponse {
            gen_time: response.gen_time + timer.elapsed(),
            result: TtsResult::File(new_audio_path),
        })
    }

    /// Transfer a TTS file from its temporary directory to a permanent one and track its contents
    async fn transform_response(
        &mut self,
        voice: VoiceReference,
        line: VoiceLine,
        response: BackendTtsResponse,
    ) -> eyre::Result<TtsResponse> {
        let target_dir = self.data.line_cache_path().join(&voice.name);
        tokio::fs::create_dir_all(&target_dir).await?;

        match response.result {
            TtsResult::File(temp_path) => {
                // TODO: Perhaps think of a better method to naming the generated lines
                let current_time = SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis();
                let file_name = format!("{}.wav", current_time);
                let target_voice_file = target_dir.join(&file_name);

                // Move the file to its permanent spot, and add it to the tracking
                tokio::fs::rename(&temp_path, &target_voice_file).await?;

                let voice_used = voice.name.clone();

                let old_value = self.data
                    .line_cache
                    .lock()
                    .await
                    .voice_to_line
                    .entry(voice)
                    .or_default()
                    .insert(line.line.clone(), file_name);

                // Delete old lines if they existed (e.g., this was forcefully regenerated)
                if let Some(old_line) = old_value {
                    tracing::debug!(?old_line, "Deleting old line for new generation");
                    let target_voice_file = target_dir.join(&old_line);
                    // If it fails we don't care.
                    let _ = tokio::fs::remove_file(target_voice_file).await;
                }

                Ok(TtsResponse {
                    file_path: target_voice_file,
                    line,
                    voice_used,
                })
            }
            TtsResult::Stream => unimplemented!("Implement stream handling (still want to cache the output as well!)"),
        }
    }
}

impl GameSharedData {
    #[tracing::instrument(skip_all)]
    async fn try_cache_retrieve(&self, voice_line: &VoiceLine) -> eyre::Result<Option<TtsResponse>> {
        let voice_to_use = match &voice_line.person {
            TtsVoice::ForceVoice(forced) => {
                forced.clone()
            }
            TtsVoice::CharacterVoice(character) => {
                self.map_character(character).await?
            }
        };
        tracing::trace!(?voice_to_use, "Will try to use voice for cache");
        
        // First check if we have a cache reference
        if !voice_line.force_generate {
            if let Some(file_name) = self
                .line_cache
                .lock()
                .await
                .voice_to_line
                .entry(voice_to_use.clone())
                .or_default()
                .get(&voice_line.line)
            {
                let target_voice_file = self.line_cache_path().join(&voice_to_use.name).join(file_name);

                return Ok(Some(TtsResponse {
                    file_path: target_voice_file,
                    line: voice_line.clone(),
                    voice_used: voice_to_use.name.clone(),
                }));
            }
        }

        Ok(None)
    }

    /// Try map the given character to a voice in our backend.
    async fn map_character(&self, character: &CharacterVoice) -> eyre::Result<VoiceReference> {
        let pin = self.game_data.character_map.pin_owned();

        if let Some(voice) = pin.get(&character.name) {
            Ok(voice.clone())
        } else {
            // First check if a game specific voice exists with the same name as the given character
            let voice_ref = VoiceReference::game(&character.name, self.game_data.game_name.clone());

            let voice_to_use = if let Some(matched) = self.voice_manager.get_voice(voice_ref).ok() {
                matched.reference
            } else {
                let voice_counts = pin.values().fold(HashMap::new(), |mut map, v| {
                    *map.entry(v).or_insert(0usize) += 1;
                    map
                });
                let mut least_used_count = usize::MAX;

                // Otherwise assign a least-used gendered voice
                match character.gender {
                    // Assume male by default
                    Some(Gender::Male) | None => {
                        let male_voice = self
                            .game_data
                            .male_voices
                            .iter()
                            .map(|v| {
                                let count = voice_counts.get(v).copied().unwrap_or(0);

                                if count < least_used_count {
                                    least_used_count = count;
                                }

                                (v, count)
                            })
                            .sorted_by_key(|(_, count)| *count)
                            .take_while(|(_, count)| *count == least_used_count)
                            .map(|(v, _)| v)
                            .choose(&mut thread_rng())
                            .context("No available male voice to assign, please make sure there is at least one!")?;

                        male_voice.clone()
                    }
                    Some(Gender::Female) => {
                        let female_voice = self
                            .game_data
                            .female_voices
                            .iter()
                            .map(|v| {
                                let count = voice_counts.get(v).copied().unwrap_or(0);

                                if count < least_used_count {
                                    least_used_count = count;
                                }

                                (v, count)
                            })
                            .sorted_by_key(|(_, count)| *count)
                            .take_while(|(_, count)| *count == least_used_count)
                            .map(|(v, _)| v)
                            .choose(&mut thread_rng())
                            .context("No available female voice to assign, please make sure there is at least one!")?;

                        female_voice.clone()
                    }
                }
            };

            let out = pin.get_or_insert(character.name.clone(), voice_to_use).clone();
            drop(pin);

            // Save the character mapping as we _really_ don't want to lose that.
            self.save_state().await?;

            Ok(out)
        }
    }

    /// Sync function as we can't do async writes to serde writers
    async fn save_cache(&self) -> eyre::Result<()> {
        let target_dir = self.line_cache_path();
        let json_file = target_dir.join(LINES_NAME);
        tokio::fs::create_dir_all(&target_dir).await?;

        let writer = std::io::BufWriter::new(std::fs::File::create(json_file)?);
        Ok(serde_json::to_writer_pretty(writer, &*self.line_cache.lock().await)?)
    }

    /// Serialize all variable state (such as character assignments) to disk.
    async fn save_state(&self) -> eyre::Result<()> {

        let config_save = self.config.dirs.game_dir(&self.game_data.game_name)
            .join(CONFIG_NAME);
        let writer = std::io::BufWriter::new(std::fs::File::create(config_save)?);

        Ok(serde_json::to_writer_pretty(writer, &self.game_data)?)
    }

    fn line_cache_path(&self) -> PathBuf {
        self.config.dirs.game_lines_cache(&self.game_data.game_name)
    }
}

#[derive(Debug, Clone, Default)]
pub struct LineCache {
    /// Voice -> Line voiced -> file name
    pub voice_to_line: HashMap<VoiceReference, HashMap<String, String>>,
}

// Needed in order to properly handle VoiceReference
impl Serialize for LineCache {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Create a temporary HashMap<String, HashMap<String, String>>
        let transformed: HashMap<String, HashMap<String, String>> = self
            .voice_to_line
            .iter()
            .map(|(key, value)| {
                let key_str = match &key.location {
                    VoiceDestination::Global => format!("global_{}", key.name),
                    VoiceDestination::Game(game_name) => format!("game_{game_name}_{}", key.name),
                };
                (key_str, value.clone())
            })
            .collect();

        transformed.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LineCache {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize into a temporary HashMap<String, HashMap<String, String>>
        let raw_map: HashMap<String, HashMap<String, String>> =
            HashMap::deserialize(deserializer)?;

        // Convert back to HashMap<VoiceReference, HashMap<String, String>>
        let voice_to_line = raw_map
            .into_iter()
            .map(|(key, value)| {
                let (location, name) = if let Some(rest) = key.strip_prefix("global_") {
                    (VoiceDestination::Global, rest.to_string())
                } else if let Some(rest) = key.strip_prefix("game_") {
                    let (game_name, character) = rest.split_once("_").ok_or_else(|| D::Error::custom("No game identifier found"))?;
                    (VoiceDestination::Game(game_name.into()), character.to_string())
                } else {
                    return Err(serde::de::Error::custom(format!("Invalid key format: {}", key)));
                };

                Ok((VoiceReference { name, location }, value))
            })
            .collect::<Result<HashMap<_, _>, D::Error>>()?;

        Ok(LineCache { voice_to_line })
    }
}