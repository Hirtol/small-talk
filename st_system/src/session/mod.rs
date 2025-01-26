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
use order_channel::OrderedSender;
use queue_actor::{GameQueueActor, SingleRequest};
use crate::{CharacterName, CharacterVoice, Gender, PostProcessing, TtsModel, TtsResponse, TtsVoice, VoiceLine};
use crate::config::TtsSystemConfig;
use crate::emotion::EmotionBackend;
use crate::error::GameSessionError;
use crate::playback::PlaybackEngineHandle;
use crate::postprocessing::AudioData;
use crate::rvc_backends::{BackendRvcRequest, RvcBackend, RvcResult};
use crate::tts_backends::{BackendTtsRequest, BackendTtsResponse, TtsBackend, TtsResult};
use crate::voice_manager::{FsVoiceData, VoiceDestination, VoiceManager, VoiceReference};

const CONFIG_NAME: &str = "config.json";
const LINES_NAME: &str = "lines.json";

type GameResult<T> = std::result::Result<T, GameSessionError>;

mod queue_actor;
mod order_channel;

#[derive(Clone)]
pub struct GameSessionHandle {
    pub playback: PlaybackEngineHandle,
    game_tts: Arc<GameTts>,
    voice_man: Arc<VoiceManager>,
}

impl GameSessionHandle {
    #[tracing::instrument(skip(config, tts, rvc, emotion, voice_man))]
    pub async fn new(
        game_name: &str,
        voice_man: Arc<VoiceManager>,
        tts: TtsBackend,
        rvc: RvcBackend,
        emotion: EmotionBackend,
        config: Arc<TtsSystemConfig>,
    ) -> eyre::Result<Self> {
        tracing::info!("Starting: {}", game_name);
        // Small amount before we exert back-pressure
        let (game_data, line_cache) = GameData::create_or_load_from_file(game_name, &config).await?;
        let line_cache = Arc::new(Mutex::new(line_cache));
        let (q_send, q_recv) = order_channel::ordered_channel();
        let (p_send, p_recv) = order_channel::ordered_channel();

        let shared_data = Arc::new(GameSharedData {
            config,
            voice_manager: voice_man.clone(),
            game_data,
            line_cache,
        });

        let queue_actor = GameQueueActor {
            tts,
            rvc,
            emotion,
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
    /// If this future is dropped prematurely the request will still be handled.
    /// This will be done even if this future is _not_ dropped.
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
        self.data.invalidate_cache_lines(items.iter().filter(|v| v.force_generate).cloned().collect()).await?;
        // Reverse iterator to ensure the push_front will leave us with the correct order in the queue
        self.queue
            .change_queue(|queue| {
                for line in items.into_iter().rev() {
                    queue.retain(|v| v.0 != line || v.1.is_some());
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
        if request.force_generate {
            self.data.invalidate_cache_line(&request).await?;
        }
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

    #[tracing::instrument(skip(self))]
    pub async fn request_tts_with_channel(&self, request: VoiceLine, send: tokio::sync::oneshot::Sender<Arc<TtsResponse>>) -> eyre::Result<()> {
        if request.force_generate {
            self.data.invalidate_cache_line(&request).await?;
        }
        // First check if the cache already contains the required data
         if let Some(tts_response) = self.data.try_cache_retrieve(&request).await? {
            let _ = send.send(Arc::new(tts_response));
        } else {
            // Otherwise send a priority request to our queue
            self.priority.change_queue(move |queue| queue.push_front((request, Some(send)))).await?;
        };

        Ok(())
    }
}

struct GameSharedData {
    config: Arc<TtsSystemConfig>,
    voice_manager: Arc<VoiceManager>,
    game_data: GameData,
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
    async fn invalidate_cache_lines(&self, items: Vec<VoiceLine>) -> eyre::Result<()> {
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

    async fn invalidate_cache_line(&self, line: &VoiceLine) -> eyre::Result<()> {
        let mut cache = self.line_cache.lock().await;

        let voice_to_use = match &line.person {
            TtsVoice::ForceVoice(forced) => forced.clone(),
            TtsVoice::CharacterVoice(character) => self.map_character(character).await?,
        };
        cache.voice_to_line
            .entry(voice_to_use)
            .or_default()
            .remove(&line.line);

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
        let config_save = self.config.game_dir(&self.game_data.game_name).join(CONFIG_NAME);
        let writer = std::io::BufWriter::new(std::fs::File::create(config_save)?);

        Ok(serde_json::to_writer_pretty(writer, &self.game_data)?)
    }

    fn line_cache_path(&self) -> PathBuf {
        self.config.game_lines_cache(&self.game_data.game_name)
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

