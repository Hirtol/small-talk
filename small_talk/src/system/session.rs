use crate::{
    config::SharedConfig,
    system::{
        config::TtsSystemConfig,
        error::GameSessionError,
        playback::PlaybackEngineHandle,
        postprocessing::AudioData,
        rvc_backends::{BackendRvcRequest, RvcBackend, RvcResult},
        tts_backends::{BackendTtsRequest, BackendTtsResponse, TtsBackend, TtsResult},
        voice_manager::{FsVoiceData, FsVoiceSample, VoiceDestination, VoiceManager, VoiceReference},
        CharacterName, CharacterVoice, Gender, PostProcessing, TtsModel, TtsResponse, TtsSystemHandle, TtsVoice, Voice,
        VoiceLine,
    },
};
use eyre::{Context, ContextCompat};
use itertools::Itertools;
use path_abs::PathOps;
use rand::{prelude::IteratorRandom, thread_rng};
use serde::{de::Error, Deserialize, Deserializer, Serialize, Serializer};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};
use std::sync::atomic::AtomicBool;
use tokio::sync::{broadcast, broadcast::error::RecvError, Mutex, Notify};
use tokio::sync::mpsc::error::TrySendError;

const CONFIG_NAME: &str = "config.json";
const LINES_NAME: &str = "lines.json";

type GameResult<T> = std::result::Result<T, GameSessionError>;

#[derive(Clone)]
pub struct GameSessionHandle {
    pub playback: PlaybackEngineHandle,
    game_tts: Arc<GameTts>,
    voice_man: Arc<VoiceManager>,
}

impl GameSessionHandle {
    #[tracing::instrument(skip(config, tts, rvc, voice_man))]
    pub async fn new(
        game_name: &str,
        voice_man: Arc<VoiceManager>,
        tts: TtsBackend,
        rvc: RvcBackend,
        config: SharedConfig,
    ) -> eyre::Result<Self> {
        tracing::info!("Starting: {}", game_name);
        // Small amount before we exert back-pressure
        let (send_b, recv_b) = broadcast::channel(10);
        let (game_data, line_cache) = GameData::create_or_load_from_file(game_name, &config.dirs).await?;
        let line_cache = Arc::new(Mutex::new(line_cache));
        let (q_send, q_recv) = ordered_channel();
        let (p_send, p_recv) = ordered_channel();

        let shared_data = Arc::new(GameSharedData {
            config,
            voice_manager: voice_man.clone(),
            game_data,
            b_recv: recv_b,
            line_cache,
        });

        let queue_actor = GameQueueActor {
            broadcast: send_b,
            tts,
            rvc,
            data: shared_data.clone(),
            queue: q_recv,
            priority: p_recv,
            generations_count: 0,
        };

        tokio::task::spawn(async move {
            if let Err(e) = queue_actor.run().await {
                tracing::error!("GameSessionQueue stopped with error: {e}");
            }
        });

        let game_tts = Arc::new(GameTts {
            data: shared_data,
            queue: q_send,
            priority: p_send,
        });

        let playback = PlaybackEngineHandle::new(Arc::downgrade(&game_tts)).await?;

        Ok(Self {
            playback,
            game_tts,
            voice_man,
        })
    }

    /// Check whether this session is still alive, or was somehow taken offline.
    pub fn is_alive(&self) -> bool {
        !self.game_tts.priority.is_closed()
    }

    /// Request a handle to the stream of [TtsResponse]s from the generation queue for this session.
    pub async fn broadcast_handle(&self) -> eyre::Result<broadcast::Receiver<Arc<TtsResponse>>> {
        Ok(self.game_tts.data.b_recv.resubscribe())
    }

    /// Force the character mapping to use the given voice.
    pub async fn force_character_voice(&self, character: CharacterName, voice: VoiceReference) -> eyre::Result<()> {
        tracing::debug!(?character, ?voice, "Forced voice mapping");
        self.game_tts.data.game_data.character_map.pin().insert(character, voice);
        Ok(())
    }

    /// Return all current character voice mappings
    pub async fn character_voices(&self) -> eyre::Result<HashMap<CharacterName, VoiceReference>> {
        Ok(self
            .game_tts
            .data
            .game_data
            .character_map
            .pin()
            .iter()
            .map(|(c, v)| (c.clone(), v.clone()))
            .collect())
    }

    /// Return all available voices for this particular game, including global voices.
    pub async fn available_voices(&self) -> eyre::Result<Vec<FsVoiceData>> {
        Ok(self.voice_man.get_voices(&self.game_tts.data.game_data.game_name))
    }

    /// Will add the given items onto the queue for TTS generation.
    ///
    /// These items will be prioritised over previous queue items
    pub async fn add_all_to_queue(&self, items: Vec<VoiceLine>) -> eyre::Result<()> {
        self.game_tts.add_all_to_queue(items).await
    }

    /// Request a single voice line with the highest priority.
    ///
    /// If this future is dropped prematurely the request will still be handled, and the response will be sent on
    /// the [Self::broadcast_handle]. This will be done even if this future is _not_ dropped.
    #[tracing::instrument(skip(self))]
    pub async fn request_tts(&self, request: VoiceLine) -> eyre::Result<Arc<TtsResponse>> {
        self.game_tts.request_tts(request).await
    }
}

pub struct GameTts {
    data: Arc<GameSharedData>,
    queue: OrderedSender<SingleRequest>,
    priority: OrderedSender<SingleRequest>,
}

impl GameTts {
    /// Will add the given items onto the queue for TTS generation.
    ///
    /// These items will be prioritised over previous queue items
    pub async fn add_all_to_queue(&self, items: Vec<VoiceLine>) -> eyre::Result<()> {
        // First invalidate all lines which have a `force_generate` flag.
        // self.data.invalidate_cache_lines(items.iter().filter(|v| v.force_generate).cloned()).await?;
        // Reverse iterator to ensure the push_front will leave us with the correct order in the queue
        self.queue
            .change_queue(|queue| {
                for line in items.into_iter().rev() {
                    queue.retain(|v| v.0 != line && v.1.is_none());
                    queue.push_front((line, None));
                }
            })
            .await
    }

    /// Request a single voice line with the highest priority.
    ///
    /// If this future is dropped prematurely the request will still be handled, and the response will be sent on
    /// the [Self::broadcast_handle]. This will be done even if this future is _not_ dropped.
    #[tracing::instrument(skip(self))]
    pub async fn request_tts(&self, request: VoiceLine) -> eyre::Result<Arc<TtsResponse>> {
        // First check if the cache already contains the required data
        let out = if let Some(tts_response) = self.data.try_cache_retrieve(&request).await? {
            Arc::new(tts_response)
        } else {
            // Otherwise send a priority request to our queue
            let (snd, rcv) = tokio::sync::oneshot::channel();

            self.priority.change_queue(|queue| queue.push_front((request, Some(snd)))).await?;

            rcv.await?
        };

        Ok(out)
    }

    // pub async fn channel_tts(&self, request: VoiceLine, send: tokio::sync::oneshot::Sender<Arc<TtsResponse>>) -> eyre::Result<()> {
    //     let out = if let Some(tts_response) = self.data.try_cache_retrieve(&request).await? {
    //         Arc::new(tts_response)
    //     } else {
    //         // Otherwise send a priority request to our queue
    //         let (snd, rcv) = tokio::sync::oneshot::channel();
    //
    //         self.priority.change_queue(|queue| queue.push_front((request, Some(snd)))).await?;
    //
    //         rcv.await?
    //     };
    //
    //     Ok(out)
    // }
}

type SingleRequest = (VoiceLine, Option<tokio::sync::oneshot::Sender<Arc<TtsResponse>>>);

struct GameQueueActor {
    broadcast: tokio::sync::broadcast::Sender<Arc<TtsResponse>>,

    tts: TtsBackend,
    rvc: RvcBackend,
    data: Arc<GameSharedData>,
    queue: OrderedReceiver<SingleRequest>,
    priority: OrderedReceiver<SingleRequest>,

    generations_count: usize,
}

struct GameSharedData {
    config: SharedConfig,

    voice_manager: Arc<VoiceManager>,
    game_data: GameData,
    line_cache: Arc<Mutex<LineCache>>,
    b_recv: broadcast::Receiver<Arc<TtsResponse>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct GameData {
    game_name: String,
    character_map: papaya::HashMap<CharacterName, VoiceReference>,
    /// The voices which should be in the random pool of assignment
    male_voices: Vec<VoiceReference>,
    female_voices: Vec<VoiceReference>,
}

impl GameData {
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
            tokio::select! {
                biased;

                Some(next_item) = self.priority.recv() => {
                    self.handle_request_err(next_item).await?
                },
                Some(next_item) = self.queue.recv() => {
                    self.handle_request_err(next_item).await?
                },
                else => break
            }
        }

        self.data.save_cache().await?;
        self.data.save_state().await?;

        Ok(())
    }

    async fn handle_request_err(&mut self, req: SingleRequest) -> eyre::Result<()> {
        match self.handle_request(req).await {
            Err(e) => match e {
                GameSessionError::Other(e) => {
                    // First persist our data
                    tracing::trace!(game=?self.data.game_data.game_name, "Stopping GameQueueActor actor as notify channel was closed");
                    self.data.save_cache().await?;
                    self.data.save_state().await?;
                    eyre::bail!(e)
                }
                GameSessionError::VoiceDoesNotExist { voice } => {
                    tracing::warn!("Ignoring request which requested non-existant voice: {voice}");
                    Ok(())
                }
                GameSessionError::IncorrectGeneration => {
                    tracing::warn!("Skipping line request after too many generation failure");
                    Ok(())
                }
            },
            _ => Ok(())
        }
    }

    #[tracing::instrument(skip(self))]
    async fn handle_request(&mut self, (next_item, respond): SingleRequest) -> GameResult<()> {
        let game_response = Arc::new(self.cache_or_request(next_item).await?);
        if let Some(response_channel) = respond {
            // If the consumer drops the other end we don't care
            let _ = response_channel.send(game_response.clone());
        }
        // Don't care whether there are receivers
        let _ = self.broadcast.send(game_response);

        Ok(())
    }

    /// Either use a cached TTS line, or generate a new one based on the given `voice_line`.
    #[tracing::instrument(skip(self))]
    async fn cache_or_request(&mut self, voice_line: VoiceLine) -> GameResult<TtsResponse> {
        // First check if we have a cache reference
        if let Some(response) = self.data.try_cache_retrieve(&voice_line).await? {
            return Ok(response);
        }
        // If we want to use RVC we'll try and warm it up before the TTS request to save time
        if let Some(post) = &voice_line.post {
            if let Some(rvc) = &post.rvc {
                self.rvc.prepare_instance(rvc.high_quality).await?;
            }
        }

        let voice_to_use = match &voice_line.person {
            TtsVoice::ForceVoice(forced) => forced.clone(),
            TtsVoice::CharacterVoice(character) => self.data.map_character(character).await?,
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

        let sample_path = sample.sample.clone();
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
                match self
                    .postprocess(&voice_line, sample_path.clone(), post, response_gen)
                    .await
                {
                    Ok(response) => Some(response),
                    Err(GameSessionError::IncorrectGeneration) => {
                        tracing::trace!(attempt = i, "Failed to generate voice line, retrying");
                        // Retry with a new generation
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            } else {
                Some(response_gen)
            };

            break;
        }
        let Some(response) = response else {
            return Err(GameSessionError::IncorrectGeneration);
        };

        let out = self.transform_response(voice_to_use, voice_line, response).await?;
        // Once in a while save our line cache in case it crashes.
        self.generations_count += 1;
        if self.generations_count > 20 {
            self.generations_count = 0;
            self.data.save_cache().await?
        }

        Ok(out)
    }

    /// Perform post-processing on the newly generated raw TTS files.
    ///
    /// This includes but is not limited to, silence trimming, low/high-pass filters.
    #[tracing::instrument(skip_all)]
    async fn postprocess(
        &mut self,
        voice_line: &VoiceLine,
        voice_sample: PathBuf,
        post_processing: &PostProcessing,
        response: BackendTtsResponse,
    ) -> Result<BackendTtsResponse, GameSessionError> {
        let should_trim = post_processing.trim_silence;
        let should_normalise = post_processing.normalise;

        let timer = std::time::Instant::now();

        let (new_audio, destination_path) = match response.result.clone() {
            TtsResult::File(temp_path) => {
                // First we check with Whisper (if desired) matches our prompt.
                if let Some(percent) = post_processing.verify_percentage {
                    let score = self.tts.verify_prompt(&temp_path, &voice_line.line).await?;
                    tracing::trace!(?score, "Whisper TTS match");
                    // There will obviously be transcription errors, so we choose a relatively
                    if score < (percent as f32 / 100.0) {
                        return Err(GameSessionError::IncorrectGeneration);
                    }
                }

                // Then we run our audio post-processing to clean it up for human ears.
                tokio::task::spawn_blocking(move || {
                    let mut raw_audio_data = wavers::Wav::<f32>::from_path(&temp_path)?;
                    let mut audio = AudioData::new(&mut raw_audio_data)?;
                    let mut sample_data: &mut [f32] = &mut audio.samples;

                    if should_trim {
                        // Basically any signal should count.
                        sample_data = super::postprocessing::trim_lead(sample_data, raw_audio_data.n_channels(), 0.01);
                    }
                    if should_normalise {
                        super::postprocessing::loudness_normalise(
                            sample_data,
                            raw_audio_data.sample_rate() as u32,
                            raw_audio_data.n_channels(),
                        );
                    }
                    audio.samples = sample_data.to_vec();

                    Ok::<_, eyre::Error>((audio, temp_path))
                })
                .await
                .context("Failed to join")??
            }
            TtsResult::Stream => unimplemented!("Streams are not yet supported"),
        };

        if let Some(rvc) = &post_processing.rvc {
            let req = BackendRvcRequest {
                audio: new_audio,
                target_voice: voice_sample,
            };
            let out = self.rvc.rvc_request(req, rvc.high_quality).await?;

            match out.result {
                RvcResult::Wav(mut data) => {
                    // Silence is still cut out, but we might need to re-normalise.
                    if should_normalise {
                        super::postprocessing::loudness_normalise(&mut data.samples, data.sample_rate, data.n_channels);
                    }
                    data.write_to_file(&destination_path)?;
                }
                RvcResult::Stream => unimplemented!("Streams are not yet supported"),
            }
            tracing::debug!(?out.gen_time, "Finished Rvc")
        }

        let took = timer.elapsed();
        tracing::debug!(?took, "Finished post-processing");

        Ok(BackendTtsResponse {
            gen_time: response.gen_time + took,
            result: TtsResult::File(destination_path),
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

                let old_value = self
                    .data
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
            TtsVoice::ForceVoice(forced) => forced.clone(),
            TtsVoice::CharacterVoice(character) => self.map_character(character).await?,
        };
        tracing::trace!(?voice_to_use, "Will try to use voice for cache");
        // TODO: Debug race condition causing rare hanging after this line
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

    /// Remove all cached lines matching the given `items`.
    async fn invalidate_cache_lines(&self, items: impl IntoIterator<Item=VoiceLine>) -> eyre::Result<()> {
        let mut cache = self.line_cache.lock().await;

        for item in items {
            let voice_to_use = match &item.person {
                TtsVoice::ForceVoice(forced) => forced.clone(),
                TtsVoice::CharacterVoice(character) => self.map_character(character).await?,
            };
            cache.voice_to_line
                .entry(voice_to_use)
                .or_default()
                .remove(&item.line);
        }

        Ok(())
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
        let config_save = self.config.dirs.game_dir(&self.game_data.game_name).join(CONFIG_NAME);
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
        let raw_map: HashMap<String, HashMap<String, String>> = HashMap::deserialize(deserializer)?;

        // Convert back to HashMap<VoiceReference, HashMap<String, String>>
        let voice_to_line = raw_map
            .into_iter()
            .map(|(key, value)| {
                let (location, name) = if let Some(rest) = key.strip_prefix("global_") {
                    (VoiceDestination::Global, rest.to_string())
                } else if let Some(rest) = key.strip_prefix("game_") {
                    let (game_name, character) = rest
                        .split_once("_")
                        .ok_or_else(|| D::Error::custom("No game identifier found"))?;
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

struct OrderedReceiver<T> {
    queue: Arc<Mutex<VecDeque<T>>>,
    notify: tokio::sync::mpsc::Receiver<()>,
}

#[derive(Clone)]
struct OrderedSender<T> {
    queue: Arc<Mutex<VecDeque<T>>>,
    notify: tokio::sync::mpsc::Sender<()>,
}

fn ordered_channel<T>() -> (OrderedSender<T>, OrderedReceiver<T>) {
    let queue = Arc::new(Mutex::new(VecDeque::new()));
    // We use a channel to piggyback off their Drop handling.
    let (send, recv) = tokio::sync::mpsc::channel(1);

    (OrderedSender {
        queue: queue.clone(),
        notify: send,
    }, OrderedReceiver {
        queue,
        notify: recv,
    })
}

impl<T> OrderedSender<T> {
    pub async fn change_queue(&self, closure: impl for<'a> FnOnce(&'a mut VecDeque<T>)) -> eyre::Result<()> {
        let mut q = self.queue.lock().await;
        closure(&mut *q);
        // Notify the queue worker that we have added new items
        match self.notify.try_send(()) {
            Err(TrySendError::Closed(_)) => Err(eyre::eyre!("Channel was closed")),
            _ => Ok(())
        }
    }

    pub fn is_closed(&self) -> bool {
        self.notify.is_closed()
    }
}

impl<T> OrderedReceiver<T> {
    /// Receive or await from the underlying queue.
    pub async fn recv(&mut self) -> Option<T> {
        let mut q = self.queue.lock().await;
        if let Some(value) = q.pop_front() {
            return Some(value)
        }
        drop(q);
        let _ = self.notify.recv().await;

        let mut q = self.queue.lock().await;
        q.pop_front()
    }
}