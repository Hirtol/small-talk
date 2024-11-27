use crate::{
    config::SharedConfig,
    system::{
        tts_backends::{BackendTtsRequest, BackendTtsResponse, TtsBackend, TtsResult},
        voice_manager::{VoiceDestination, VoiceManager, VoiceReference},
        CharacterName, CharacterVoice, Gender, TtsResponse, TtsSystemHandle, Voice, VoiceLine,
    },
};
use eyre::ContextCompat;
use itertools::Itertools;
use path_abs::PathOps;
use rand::{prelude::IteratorRandom, thread_rng};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};
use tokio::{
    io::BufWriter,
    sync::{broadcast::error::RecvError, Mutex},
};
use tokio::sync::broadcast;
use crate::system::TtsVoice;

const CONFIG_NAME: &str = "config.json";
const LINES_NAME: &str = "lines.json";

pub type GameSessionActorHandle = tokio::sync::mpsc::UnboundedSender<GameSessionMessage>;

#[derive(Clone, Debug)]
pub struct GameSessionHandle {
    send: tokio::sync::mpsc::Sender<GameSessionMessage>,
}

impl GameSessionHandle {
    pub async fn new(
        game_name: &str,
        voice_man: Arc<VoiceManager>,
        tts: TtsBackend,
        config: SharedConfig,
    ) -> eyre::Result<Self> {
        // Small amount before we exert back-pressure
        let (send, recv) = tokio::sync::mpsc::channel(10);
        let (send_b, recv_b) = tokio::sync::broadcast::channel(100);
        let (notify_send, notify_recv) = tokio::sync::mpsc::channel(1);
        let (game_data, line_cache) = GameSessionActor::create_or_load_from_file(game_name, &config).await?;
        let line_cache = Arc::new(Mutex::new(line_cache));
        let shared_queue = Arc::new(Mutex::new(VecDeque::new()));

        let shared_data = Arc::new(GameSharedData {
            config,
            voice_manager: voice_man,
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
            data: shared_data,
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

        Ok(Self {
            send,
        })
    }
    
    /// Request a handle to the stream of [TtsResponse]s from the generation queue for this session.
    pub async fn broadcast_handle(&self) -> eyre::Result<broadcast::Receiver<Arc<TtsResponse>>> {
        let (send, recv) = tokio::sync::oneshot::channel();
        self.send.send(GameSessionMessage::BroadcastHandle(send)).await?;

        Ok(recv.await?)
    }
    
    /// Will add the given items onto the queue for TTS generation.
    /// 
    /// These items will be prioritised over previous queue items
    pub async fn add_all_to_queue(&self, items: Vec<VoiceLine>) -> eyre::Result<()> {
        Ok(self.send.send(GameSessionMessage::AddToQueue(items)).await?)
    }
    
    /// Request a single voice line with the highest priority.
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
    /// Retrieve
    Single(VoiceLine, tokio::sync::oneshot::Sender<Arc<TtsResponse>>),
    /// Request a direct handle to the TtsResponse stream
    BroadcastHandle(tokio::sync::oneshot::Sender<broadcast::Receiver<Arc<TtsResponse>>>)
}

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
    queue: Arc<Mutex<VecDeque<VoiceLine>>>,
    line_cache: Arc<Mutex<LineCache>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GameData {
    pub game_name: String,
    pub character_map: papaya::HashMap<CharacterName, VoiceReference>,
    /// The voices which should be in the random pool of assignment
    pub male_voices: Vec<VoiceReference>,
    pub female_voices: Vec<VoiceReference>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct LineCache {
    /// Voice -> Line voiced -> file name
    pub voice_to_line: HashMap<VoiceReference, HashMap<String, String>>,
}

impl GameSessionActor {
    pub async fn run(mut self) -> eyre::Result<()> {
        loop {
            let Some(msg) = self.recv.recv().await else {
                tracing::trace!("Stopping local actor as channel was closed");
                break;
            };
            self.handle_message(msg).await?;
        }

        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn handle_message(&mut self, message: GameSessionMessage) -> eyre::Result<()> {
        match message {
            GameSessionMessage::AddToQueue(new_lines) => {
                // Reverse iterator to ensure the push_front will leave us with the correct order in the queue
                let mut queue = self.data.queue.lock().await;
                for line in new_lines.into_iter().rev() {
                    queue.retain(|v| v != &line);
                    queue.push_front(line);
                }
                // Notify the queue worker that we have added new items
                let _ = self.notify.try_send(());
            }
            GameSessionMessage::Single(req, response) => {
                // First check if the cache already contains the required data
                let to_send = if let Some(tts_response) = self.data.try_cache_retrieve(&req).await? {
                    Arc::new(tts_response)
                } else {
                    // Make sure this lock is dropped here, otherwise deadlock!
                    self.data.queue.lock().await.push_front(req.clone());
                    let _ = self.notify.try_send(());

                    // Wait on the queue actor to broadcast the completion of our request
                    loop {
                        match self.b_recv.recv().await {
                            Ok(resp) if resp.line == req => break resp,
                            Ok(ignored) => {
                                tracing::trace!(?ignored, "Ignoring previous generation")
                            }
                            Err(RecvError::Lagged(_)) => {
                                // Fine to ignore, this is expected
                            }
                            Err(e) => eyre::bail!(e),
                        }
                    }
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

    pub async fn create_or_load_from_file(
        game_name: &str,
        config: &SharedConfig,
    ) -> eyre::Result<(GameData, LineCache)> {
        let dir = super::dirs::game_dir(config, game_name);

        if tokio::fs::try_exists(&dir).await? {
            Self::load_from_dir(&dir).await
        } else {
            Self::create(game_name, &dir).await
        }
    }

    async fn create(game_name: &str, dir: &Path) -> eyre::Result<(GameData, LineCache)> {
        let data = GameData {
            game_name: game_name.into(),
            character_map: Default::default(),
            male_voices: vec![],
            female_voices: vec![],
        };
        let out = serde_json::to_vec_pretty(&data)?;

        tokio::fs::create_dir_all(dir).await?;
        tokio::fs::write(dir.join(CONFIG_NAME), &out).await?;

        Ok((data, Default::default()))
    }

    async fn load_from_dir(dir: &Path) -> eyre::Result<(GameData, LineCache)> {
        let game_data = tokio::fs::read(dir.join(CONFIG_NAME)).await?;
        let data = serde_json::from_slice(&game_data)?;
        // If the below doesn't exist we can just re-create it.
        let line_file = super::dirs::game_dir_lines_cache(dir).join(LINES_NAME);
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
                tracing::trace!(?self.data.game_data.game_name, "Stopping GameQueueActor actor as notify channel was closed");
                break;
            };

            while self.pop_queue().await? {}
        }

        // Save our line cache before dropping this session.
        self.data.save_cache().await?;

        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn pop_queue(&mut self) -> eyre::Result<bool> {
        let Some(next_item) = self.data.queue.lock().await.pop_front() else {
            // Can just return as the queue won't be modified
            return Ok(false);
        };

        let game_response = Arc::new(self.cache_or_request(next_item).await?);
        // Don't care whether there are receivers
        let _ = self.broadcast.send(game_response);

        Ok(true)
    }

    /// Either use a cached TTS line, or generate a new one based on the given `voice_line`.
    #[tracing::instrument(skip(self))]
    async fn cache_or_request(&mut self, voice_line: VoiceLine) -> eyre::Result<TtsResponse> {
        // First check if we have a cache reference
        if let Some(response) = self.data.try_cache_retrieve(&voice_line).await? {
            return Ok(response);
        }
        let voice_to_use = match &voice_line.person {
            TtsVoice::ForceVoice(forced) => {
                forced.clone()
            }
            TtsVoice::CharacterVoice(character) => {
                self.data.map_character(character)?
            }
        };

        // TODO: Line emotion detection
        let voice = self.data.voice_manager.get_voice(&voice_to_use)?;
        let sample = voice.random_sample()?;

        // TODO: Configurable language
        let request = BackendTtsRequest {
            gen_text: voice_line.line.clone(),
            language: "en".to_string(),
            voice_reference: vec![sample],
            speed: None,
        };
        let response = self.tts.tts_request(voice_line.model, request).await?;
        
        let out = self.transform_response(&voice_to_use, voice_line, response).await?;
        // Once in a while save our line cache in case it crashes.
        self.generations_count += 1;
        if self.generations_count > 20 {
            self.data.save_cache().await?
        }

        Ok(out)
    }

    /// Transfer a TTS file from its temporary directory to a permanent one and track its contents
    async fn transform_response(
        &mut self,
        voice: &VoiceReference,
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
                self.data
                    .line_cache
                    .lock()
                    .await
                    .voice_to_line
                    .entry(voice.clone())
                    .or_default()
                    .insert(line.line.clone(), file_name);

                Ok(TtsResponse {
                    file_path: target_voice_file,
                    line,
                    voice_used: voice.name.clone(),
                })
            }
            TtsResult::Stream => unimplemented!("Implement stream handling (still want to cache the output as well!)"),
        }
    }
}

impl GameSharedData {
    #[tracing::instrument(skip(self))]
    async fn try_cache_retrieve(&self, voice_line: &VoiceLine) -> eyre::Result<Option<TtsResponse>> {
        let voice_to_use = match &voice_line.person {
            TtsVoice::ForceVoice(forced) => {
                forced.clone()
            }
            TtsVoice::CharacterVoice(character) => {
                self.map_character(character)?
            }
        };;

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
                let target_voice_file = self.line_cache_path().join(&voice_to_use.name).join(&file_name);

                return Ok(Some(TtsResponse {
                    file_path: target_voice_file,
                    line: voice_line.clone(),
                    voice_used: voice_to_use.name.clone(),
                }));
            }
        } else {
            tracing::debug!("Forcefully regenerating line")
        }

        Ok(None)
    }

    /// Try map the given character to a voice in our backend.
    fn map_character(&self, character: &CharacterVoice) -> eyre::Result<VoiceReference> {
        let pin = self.game_data.character_map.pin();

        if let Some(voice) = pin.get(&character.name) {
            Ok(voice.clone())
        } else {
            // First check if a game specific voice exists with the same name as the given character
            let voices = self.voice_manager.get_game_voices(&self.game_data.game_name);
            let voice_to_use = if let Some(matched) = voices.iter().find(|v| v.name == character.name) {
                VoiceReference::game(matched.name.clone(), self.game_data.game_name.clone())
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

            Ok(pin.get_or_insert(character.name.clone(), voice_to_use).clone())
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

    fn line_cache_path(&self) -> PathBuf {
        super::dirs::game_lines_cache(&self.config, &self.game_data.game_name)
    }
}
