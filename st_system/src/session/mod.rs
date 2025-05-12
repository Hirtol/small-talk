use crate::{
    config::TtsSystemConfig, data::TtsModel, emotion::EmotionBackend, error::GameSessionError, playback::PlaybackEngineHandle, postprocessing::AudioData, rvc_backends::{BackendRvcRequest, RvcCoordinator, RvcResult},
    session::db::{DatabaseGender, DbEnumHelper, SessionDb},
    tts_backends::{BackendTtsRequest, BackendTtsResponse, TtsCoordinator, TtsResult},
    voice_manager::{FsVoiceData, VoiceDestination, VoiceManager, VoiceReference},
    CharacterName,
    CharacterVoice,
    Gender,
    PostProcessing,
    TtsResponse,
    TtsVoice,
    VoiceLine,
};
use eyre::{Context, ContextCompat};
use itertools::Itertools;
use linecache::LineCache;
use order_channel::OrderedSender;
use path_abs::PathOps;
use queue_actor::{GameQueueActor, SingleRequest};
use rand::prelude::IteratorRandom;
use sea_orm::{sea_query, ActiveEnum, ActiveModelTrait, ColumnTrait, EntityTrait, IntoActiveValue, QueryFilter, QuerySelect, QueryTrait};
use serde::{de::Error, Deserialize, Deserializer, Serialize, Serializer};
use st_db::{ReadConnection, SelectExt, WriteConnection, WriteTransaction};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    num::NonZeroU32,
    path::{Path, PathBuf},
    sync::{atomic::AtomicBool, Arc},
    time::SystemTime,
};
use tokio::sync::{broadcast, broadcast::error::RecvError, mpsc::error::TrySendError, Mutex, Notify};
use crate::session::queue_actor::VoiceLineRequest;

const CONFIG_NAME: &str = "config.json";
const DB_NAME: &str = "database.db";
const LINES_NAME: &str = "lines.json";

type GameResult<T> = std::result::Result<T, GameSessionError>;

pub mod db;
pub mod linecache;
mod order_channel;
mod queue_actor;

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
        tts: TtsCoordinator,
        rvc: RvcCoordinator,
        emotion: EmotionBackend,
        config: Arc<TtsSystemConfig>,
    ) -> eyre::Result<Self> {
        tracing::info!("Starting: {}", game_name);
        // Small amount before we exert back-pressure
        let (game_data, line_cache, db) = GameData::create_or_load_from_file(game_name, &config).await?;
        // let line_cache = Arc::new(Mutex::new(line_cache));
        let (q_send, q_recv) = order_channel::ordered_channel();
        let (p_send, p_recv) = order_channel::ordered_channel();

        let shared_data = Arc::new(GameSharedData {
            game_db: db,
            config,
            voice_manager: voice_man.clone(),
            game_data,
            // line_cache,
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

    /// Retrieve the name of this session
    pub fn name(&self) -> &str {
        &self.game_tts.data.game_data.game_name
    }

    /// Check whether this session is still alive, or was somehow taken offline.
    pub fn is_alive(&self) -> bool {
        !self.game_tts.priority.is_closed()
    }

    /// Force the character mapping to use the given voice.
    pub async fn force_character_voice(&self, character: CharacterVoice, voice: VoiceReference) -> eyre::Result<()> {
        tracing::debug!(?character, ?voice, "Forced voice mapping");
        use st_db::entity::characters::*;

        let to_update = ActiveModel {
            id: Default::default(),
            character_name: character.name.into_active_value(),
            character_gender: character.gender.unwrap_or(Gender::default()).to_db().to_value().into_active_value(),
            voice_name: voice.name.into_active_value(),
            voice_location: voice.location.to_string_value().into_active_value(),
        };

        Entity::insert(to_update)
            .on_conflict(
                sea_query::OnConflict::columns([Column::CharacterName, Column::CharacterGender])
                    .update_columns([Column::VoiceName, Column::VoiceLocation])
                    .to_owned(),
            )
            .exec(self.game_tts.data.game_db.reader())
            .await?;
        Ok(())
    }

    /// Return all current character voice mappings
    pub async fn character_voices(&self) -> eyre::Result<HashMap<CharacterVoice, VoiceReference>> {
        use st_db::entity::characters::*;

        let entities = Entity::find().all(self.game_tts.data.game_db.reader()).await?;

        Ok(entities
            .into_iter()
            .map(|val| {
                let character = CharacterVoice {
                    name: val.character_name,
                    gender: DatabaseGender::try_from_value(&val.character_gender)
                        .map(|g| g.into())
                        .ok()
                };

                let voice = VoiceReference {
                    name: val.voice_name,
                    location: val.voice_location.into(),
                };

                (character, voice)
            })
            .collect())
    }

    /// Return all available voices for this particular game, including global voices.
    pub async fn available_voices(&self) -> eyre::Result<Vec<FsVoiceData>> {
        Ok(self.voice_man.get_voices(&self.game_tts.data.game_data.game_name))
    }

    /// Return all text lines voiced by the given [VoiceReference]
    pub async fn voice_lines(&self, voice: &VoiceReference) -> eyre::Result<Vec<String>> {
        let voice_ref: Vec<String> = db::voice_lines::Entity::find()
            .select_only()
            .columns([db::voice_lines::Column::DialogueText])
            .filter(db::lines_table_voice_reference_condition(voice))
            .into_tuple()
            .all(self.game_tts.data.game_db.reader())
            .await?;

        Ok(voice_ref)
    }

    /// Will add the given items onto the queue for TTS generation.
    ///
    /// These items will be prioritised over previous queue items
    pub async fn add_all_to_queue(&self, items: Vec<VoiceLine>) -> eyre::Result<()> {
        self.game_tts.add_all_to_queue(items).await
    }

    /// Request a single voice line
    ///
    /// If this future is dropped prematurely the request will still be handled.
    /// This will be done even if this future is _not_ dropped.
    #[tracing::instrument(skip(self))]
    pub async fn request_tts(&self, request: VoiceLine) -> eyre::Result<Arc<TtsResponse>> {
        let (snd, rcv) = tokio::sync::oneshot::channel();

        self.game_tts.request_tts_with_channel(request, snd).await?;

        Ok(rcv.await?)
    }
}


pub struct GameTts {
    /// Database containing character voice mappings and dialogue
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
        self.data
            .invalidate_cache_lines(items.iter().filter(|v| v.force_generate).cloned().collect())
            .await?;
        // Reverse iterator to ensure the push_front will leave us with the correct order in the queue
        self.queue
            .change_queue(|queue| {
                for line in items.into_iter().rev() {
                    queue.retain(|v| v.0 != line || v.1.is_some());
                    queue.push_front((line, None, tracing::Span::current()));
                }
            })
            .await
    }

    /// Request a single voice line with normal priority.
    ///
    /// Returns the completed request.
    pub async fn add_to_queue(&self, item: VoiceLineRequest) -> eyre::Result<Arc<TtsResponse>> {
        if item.force_generate {
            let tx = self.data.game_db.writer().begin().await?;
            self.data.invalidate_cache_line(&tx, &item).await?;
            tx.commit().await?;
        }
        let (snd, rcv) = tokio::sync::oneshot::channel();

        self.queue
            .change_queue(|queue| {
                queue.push_front((item, Some(snd), tracing::Span::current()));
            })
            .await?;

        Ok(rcv.await?)
    }

    /// Request a single voice line with the highest priority.
    ///
    /// Any previous request(s) on the highest priority channel are demoted to back of the regular queue.
    #[tracing::instrument(skip(self))]
    pub async fn request_tts_with_channel(
        &self,
        request: VoiceLineRequest,
        send: tokio::sync::oneshot::Sender<Arc<TtsResponse>>,
    ) -> eyre::Result<()> {
        let tx = self.data.game_db.writer().begin().await?;
        if request.force_generate {
            self.data.invalidate_cache_line(&tx, &request).await?;
        }
        // First check if the cache already contains the required data
        if let Some(tts_response) = self.data.try_cache_retrieve(&tx, &request).await? {
            let _ = send.send(Arc::new(tts_response));
        } else {
            // Otherwise send a priority request to our queue, clear any previous urgent requests.
            let to_queue = self
                .priority
                .change_queue(move |priority| {
                    let old_values = std::mem::take(priority);
                    priority.push_front((request, Some(send), tracing::Span::current()));
                    old_values
                })
                .await?;

            if !to_queue.is_empty() {
                self.queue
                    .change_queue(move |queue| {
                        queue.extend(to_queue);
                    })
                    .await?;
            }
        };

        tx.commit().await?;

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GameData {
    /// The name of the game for which this data is associated.
    game_name: String,
    /// A mapping from character names to their assigned voice references.
    character_map: papaya::HashMap<CharacterName, VoiceReference>,
    /// The voices which should be in the random pool of assignment for male characters.
    male_voices: Vec<VoiceReference>,
    /// The voices which should be in the random pool of assignment for female characters.
    female_voices: Vec<VoiceReference>,
}

impl GameData {
    pub async fn create_or_load_from_file(
        game_name: &str,
        config: &TtsSystemConfig,
    ) -> eyre::Result<(GameData, LineCache, SessionDb)> {
        if tokio::fs::try_exists(config.game_dir(game_name)).await? {
            Self::load_from_dir(config, game_name).await
        } else {
            Self::create(game_name, config).await
        }
    }

    pub async fn create(game_name: &str, config: &TtsSystemConfig) -> eyre::Result<(GameData, LineCache, SessionDb)> {
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
        let db_conf = db::DbConfig {
            db_path: dir.join(DB_NAME),
            in_memory: false,
            max_connections_reader: NonZeroU32::new(8).unwrap(),
            max_connections_writer: NonZeroU32::new(1).unwrap(),
        };
        let db = db_conf.initialise_database().await?;

        Ok((data, Default::default(), db))
    }

    pub async fn load_from_dir(
        conf: &TtsSystemConfig,
        game_name: &str,
    ) -> eyre::Result<(GameData, LineCache, SessionDb)> {
        let dir = conf.game_dir(game_name);
        let game_data = tokio::fs::read(dir.join(CONFIG_NAME)).await?;
        let data = serde_json::from_slice(&game_data)?;
        // If the below doesn't exist we can just re-create it.
        let line_file = conf.game_dir_lines_cache(&dir).join(LINES_NAME);
        let line_cache = tokio::fs::read(line_file).await.unwrap_or_default();
        let lines = serde_json::from_slice(&line_cache).unwrap_or_default();

        let db_conf = db::DbConfig {
            db_path: dir.join(DB_NAME),
            in_memory: false,
            max_connections_reader: NonZeroU32::new(8).unwrap(),
            max_connections_writer: NonZeroU32::new(1).unwrap(),
        };
        let db = db_conf.initialise_database().await?;

        Ok((data, lines, db))
    }
}

pub struct GameSharedData {
    pub game_db: SessionDb,
    // pub line_cache: Arc<Mutex<LineCache>>,
    pub config: Arc<TtsSystemConfig>,
    pub voice_manager: Arc<VoiceManager>,
    pub game_data: GameData,
}

impl GameSharedData {
    #[tracing::instrument(skip_all)]
    async fn try_cache_retrieve(&self, tx: &impl WriteConnection, voice_line: &VoiceLine) -> eyre::Result<Option<TtsResponse>> {
        let voice_to_use = match &voice_line.person {
            TtsVoice::ForceVoice(forced) => forced.clone(),
            TtsVoice::CharacterVoice(character) => self.map_character(tx, character).await?,
        };

        tracing::trace!(?voice_to_use, "Will try to use voice for cache");

        if !voice_line.force_generate {
            let out = db::voice_lines::Entity::find()
                .filter(db::lines_table_voice_line_condition(&voice_line.line, &voice_to_use))
                .one(tx)
                .await?;

            let result = out.map(|v| {
                let target_voice_file = self.lines_voice_path(&voice_to_use).join(v.file_name);

                TtsResponse {
                    file_path: target_voice_file,
                    line: voice_line.clone(),
                    voice_used: voice_to_use.name.clone(),
                }
            });

            Ok(result)
        } else {
            Ok(None)
        }
    }

    /// Remove all cached lines matching the given `items`.
    async fn invalidate_cache_lines(&self, items: Vec<VoiceLine>) -> eyre::Result<()> {
        let tx = self.game_db.writer().begin().await?;

        // N queries, could be more efficient...
        for item in items {
            self.invalidate_cache_line(&tx, &item).await?;
        }

        tx.commit().await?;

        Ok(())
    }

    async fn invalidate_cache_line(&self, tx: &impl WriteConnection, line: &VoiceLine) -> eyre::Result<()> {
        use st_db::entity::*;

        let voice_to_use = match &line.person {
            TtsVoice::ForceVoice(forced) => forced.clone(),
            TtsVoice::CharacterVoice(character) => self.map_character(tx, character).await?,
        };

        voice_lines::Entity::delete_many()
            .filter(
                voice_lines::Column::Id.in_subquery(
                    voice_lines::Entity::find()
                        .select_only()
                        .column(voice_lines::Column::Id)
                        .filter(db::lines_table_voice_line_condition(&line.line, &voice_to_use))
                        .into_query(),
                ),
            )
            .exec(tx)
            .await?;

        Ok(())
    }

    /// Try map the given character to a voice in our backend.
    async fn map_character(
        &self,
        tx: &impl WriteConnection,
        character: &CharacterVoice,
    ) -> eyre::Result<VoiceReference> {
        // First check if the character exists in our database
        let existing_voice = db::characters::Entity::find()
            .filter(db::characters::Column::CharacterName.eq(&character.name))
            .filter(db::characters::Column::CharacterGender.eq(character.gender.map(|g| g.to_db())))
            .one(tx)
            .await?;

        if let Some(voice) = existing_voice {
            Ok(voice.into())
        } else {
            // First check if a game specific voice exists with the same name as the given character
            let voice_ref = VoiceReference::game(&character.name, self.game_data.game_name.clone());

            let voice_to_use = if let Some(matched) = self.voice_manager.get_voice(voice_ref).ok() {
                matched.reference
            } else {
                let voice_counts: Vec<(String, String, u32)> = db::characters::Entity::find()
                    .select_only()
                    .columns([db::characters::Column::VoiceName, db::characters::Column::VoiceLocation])
                    .column_as(db::characters::Column::Id.count(), "count")
                    .group_by(db::characters::Column::VoiceName)
                    .group_by(db::characters::Column::VoiceLocation)
                    .into_tuple()
                    .all(tx)
                    .await?;
                let voice_counts = voice_counts
                    .into_iter()
                    .map(|(a, b, c)| (VoiceReference::from_strings(a, b), c))
                    .collect::<HashMap<_, _>>();
                let mut least_used_count = u32::MAX;

                // Otherwise assign a least-used gendered voice
                match character.gender.unwrap_or_default() {
                    // Assume male by default
                    Gender::Male => {
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
                            .choose(&mut rand::rng())
                            .context("No available male voice to assign, please make sure there is at least one!")?;

                        male_voice.clone()
                    }
                    Gender::Female => {
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
                            .choose(&mut rand::rng())
                            .context("No available female voice to assign, please make sure there is at least one!")?;

                        female_voice.clone()
                    }
                }
            };

            let to_insert = db::characters::ActiveModel {
                id: Default::default(),
                character_name: character.name.clone().into_active_value(),
                character_gender: character.gender.unwrap_or(Gender::default()).to_db().to_value().into_active_value(),
                voice_name: voice_to_use.name.into_active_value(),
                voice_location: voice_to_use.location.to_string_value().into_active_value(),
            };

            let out = to_insert.insert(tx).await?;

            Ok(out.into())
        }
    }

    /// Sync function as we can't do async writes to serde writers
    pub async fn save_cache(&self) -> eyre::Result<()> {
        // let target_dir = self.line_cache_path();
        // let json_file = target_dir.join(LINES_NAME);
        // tokio::fs::create_dir_all(&target_dir).await?;
        //
        // let writer = std::io::BufWriter::new(std::fs::File::create(json_file)?);
        // Ok(serde_json::to_writer_pretty(writer, &*self.line_cache.lock().await)?)
        Ok(())
    }

    /// Serialize all variable state (such as character assignments) to disk.
    pub async fn save_state(&self) -> eyre::Result<()> {
        let config_save = self.config.game_dir(&self.game_data.game_name).join(CONFIG_NAME);
        let writer = std::io::BufWriter::new(std::fs::File::create(config_save)?);

        Ok(serde_json::to_writer_pretty(writer, &self.game_data)?)
    }

    /// Returns the path to the directory containing all spoken dialogue by the given [VoiceReference]
    pub fn lines_voice_path(&self, voice: &VoiceReference) -> PathBuf {
        self.line_cache_path().join(&voice.name)
    }

    fn line_cache_path(&self) -> PathBuf {
        self.config.game_lines_cache(&self.game_data.game_name)
    }
}
