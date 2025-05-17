use crate::{
    config::TtsSystemConfig, data::TtsModel, emotion::EmotionBackend, error::GameSessionError, rvc_backends::{BackendRvcRequest, RvcCoordinator, RvcResult},
    session::{
        db::{DatabaseGender, DbEnumHelper, SessionDb},
        linecache::LineCacheEntry,
        queue_actor::VoiceLineRequest,
    },
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
use futures::TryFutureExt;
use itertools::Itertools;
use linecache::LineCache;
use order_channel::OrderedSender;
use path_abs::PathOps;
use queue_actor::{GameQueueActor, SingleRequest};
use rand::prelude::IteratorRandom;
use sea_orm::{
    sea_query, ActiveEnum, ActiveModelTrait, ColumnTrait, DbBackend, EntityTrait, IntoActiveValue, QueryFilter,
    QuerySelect, QueryTrait,
};
use sea_query::OnConflict;
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
use tracing::log;
use crate::audio::playback::PlaybackEngineHandle;
use crate::audio::postprocessing::AudioData;

const CONFIG_NAME: &str = "config.json";
const DB_NAME: &str = "database.db";
const LINES_NAME: &str = "lines.json";

type GameResult<T> = std::result::Result<T, GameSessionError>;
type CharacterRef = db::characters::Model;

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

        let (game_data, db) = GameData::create_or_load_from_file(game_name, &config).await?;
        let line_cache = Arc::new(LineCache::new(game_name.to_string(), config.clone(), db.clone()));

        let (q_send, q_recv) = order_channel::ordered_channel();
        let (p_send, p_recv) = order_channel::ordered_channel();

        let shared_data = Arc::new(GameSharedData {
            game_db: db,
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
                tracing::error!("GameSessionQueue stopped with error: {e:?}");
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
            character_gender: character
                .gender
                .unwrap_or(Gender::default())
                .to_db()
                .to_value()
                .into_active_value(),
            voice_name: voice.name.into_active_value(),
            voice_location: voice.location.to_string_value().into_active_value(),
        };

        Entity::insert(to_update)
            .on_conflict(
                sea_query::OnConflict::columns([Column::CharacterName, Column::CharacterGender])
                    .update_columns([Column::VoiceName, Column::VoiceLocation])
                    .to_owned(),
            )
            .exec(self.game_tts.data.game_db.writer())
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
                        .ok(),
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
    /// Will push the given items to the queue for TTS generation.
    ///
    /// These items will be prioritised over previous queue items
    pub async fn add_all_to_queue(&self, items: Vec<VoiceLine>) -> eyre::Result<()> {
        use futures_lite::stream::StreamExt;
        let tx = self.data.game_db.writer().begin().await?;

        // First invalidate all lines which have a `force_generate` flag.
        let to_invalidate: Vec<_> = futures::stream::iter(items.iter().filter(|v| v.force_generate))
            .then(|x| self.data.voice_line_to_cache(&tx, x))
            .try_collect()
            .await?;
        self.data.line_cache.invalidate_cache_lines(&tx, to_invalidate).await?;

        // Then check and add any dialogue which is new.
        self.data.try_add_new_dialogue(&tx, &items).await?;

        // And map these items to requests
        let requests: Vec<_> = futures::stream::iter(&items)
            .then(|request| {
                self.data
                    .extract_voice_reference(&tx, &request)
                    .map_ok(move |speaker| VoiceLineRequest {
                        speaker,
                        text: request.line.clone(),
                        model: request.model,
                        post: request.post.clone(),
                    })
            })
            .try_collect()
            .await?;

        tx.commit().await?;

        // Reverse iterator to ensure the push_front will leave us with the correct order in the queue
        self.queue
            .change_queue(|queue| {
                for line in requests.into_iter().rev() {
                    queue.retain(|v| v.0 != line || v.1.is_some());
                    queue.push_front((line, None, tracing::Span::current()));
                }
            })
            .await
    }

    /// Request a single voice line with the highest priority.
    ///
    /// Any previous request(s) on the highest priority channel are demoted to back of the regular queue.
    #[tracing::instrument(skip(self))]
    pub async fn request_tts_with_channel(
        &self,
        request: VoiceLine,
        send: tokio::sync::oneshot::Sender<Arc<TtsResponse>>,
    ) -> eyre::Result<()> {
        let tx = self.data.game_db.writer().begin().await?;
        self.data.try_add_new_dialogue(&tx, &[request.clone()]).await?;

        let existing_line = if request.force_generate {
            let cache_entry = self.data.voice_line_to_cache(&tx, &request).await?;
            self.data.line_cache.invalidate_cache_lines(&tx, [cache_entry]).await?;
            None
        } else {
            self.data.try_cache_retrieve(&tx, &request).await?
        };
        // Need to commit here to finalise the cache invalidation
        tx.commit().await?;

        // First check if the cache already contains the required data
        if let Some(tts_response) = existing_line {
            let _ = send.send(Arc::new(tts_response));
        } else {
            // Otherwise send a priority request to our queue, clear any previous urgent requests and return them
            // to the lower priority queue.
            let vl_request = VoiceLineRequest {
                speaker: self.data.extract_voice_reference(self.data.game_db.writer(), &request).await?,
                text: request.line,
                model: request.model,
                post: request.post,
            };

            let lower_priority = self
                .priority
                .change_queue(move |priority| {
                    let old_values = std::mem::take(priority);
                    priority.push_front((vl_request, Some(send), tracing::Span::current()));
                    old_values
                })
                .await?;

            if !lower_priority.is_empty() {
                self.queue
                    .change_queue(move |queue| {
                        queue.extend(lower_priority);
                    })
                    .await?;
            }
        };

        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GameData {
    /// The name of the game to which this data is associated.
    game_name: String,
    /// The voices which should be in the random pool of assignment for male characters.
    male_voices: Vec<VoiceReference>,
    /// The voices which should be in the random pool of assignment for female characters.
    female_voices: Vec<VoiceReference>,
}

impl GameData {
    pub async fn create_or_load_from_file(
        game_name: &str,
        config: &TtsSystemConfig,
    ) -> eyre::Result<(GameData, SessionDb)> {
        if tokio::fs::try_exists(config.game_dir(game_name)).await? {
            Self::load_from_dir(config, game_name).await
        } else {
            Self::create(game_name, config).await
        }
    }

    pub async fn create(game_name: &str, config: &TtsSystemConfig) -> eyre::Result<(GameData, SessionDb)> {
        let data = GameData {
            game_name: game_name.into(),
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

        Ok((data, db))
    }

    pub async fn load_from_dir(conf: &TtsSystemConfig, game_name: &str) -> eyre::Result<(GameData, SessionDb)> {
        let dir = conf.game_dir(game_name);
        let game_data = tokio::fs::read(dir.join(CONFIG_NAME)).await?;
        let data = serde_json::from_slice(&game_data)?;

        let db_conf = db::DbConfig {
            db_path: dir.join(DB_NAME),
            in_memory: false,
            max_connections_reader: NonZeroU32::new(8).unwrap(),
            max_connections_writer: NonZeroU32::new(1).unwrap(),
        };
        let db = db_conf.initialise_database().await?;

        Ok((data, db))
    }
}

pub struct GameSharedData {
    pub game_db: SessionDb,
    pub line_cache: Arc<LineCache>,
    pub config: Arc<TtsSystemConfig>,
    pub voice_manager: Arc<VoiceManager>,
    pub game_data: GameData,
}

impl GameSharedData {
    #[tracing::instrument(skip_all)]
    async fn try_cache_retrieve(
        &self,
        tx: &impl WriteConnection,
        voice_line: &VoiceLine,
    ) -> eyre::Result<Option<TtsResponse>> {
        if !voice_line.force_generate {
            let cache_entry = self.voice_line_to_cache(tx, voice_line).await?;
            self.line_cache.try_retrieve(tx, cache_entry).await
        } else {
            Ok(None)
        }
    }

    pub async fn voice_line_to_cache(
        &self,
        tx: &impl WriteConnection,
        line: &VoiceLine,
    ) -> eyre::Result<LineCacheEntry> {
        let voice = self.extract_voice_reference(tx, &line).await?;
        Ok(LineCacheEntry {
            text: line.line.clone(),
            voice,
        })
    }

    pub async fn extract_voice_reference(
        &self,
        tx: &impl WriteConnection,
        line: &VoiceLine,
    ) -> eyre::Result<VoiceReference> {
        Ok(match &line.person {
            TtsVoice::ForceVoice(forced) => forced.clone(),
            TtsVoice::CharacterVoice(character) => self.map_character(tx, character).await?.into(),
        })
    }

    async fn try_add_new_dialogue(&self, tx: &impl WriteConnection, voice_lines: &[VoiceLine]) -> eyre::Result<()> {
        use futures_lite::stream::StreamExt;
        let all_dialogue = voice_lines.into_iter().flat_map(|x| {
            if let TtsVoice::CharacterVoice(voice) = &x.person {
                Some((&x.line, voice))
            } else {
                None
            }
        });
        let all_characters: Vec<_> = futures::stream::iter(all_dialogue)
            .then(|(line, voice)| self.map_character(tx, voice).map_ok(move |x| (line, x)))
            .try_collect()
            .await?;

        if all_characters.is_empty() {
            // Only forced dialogue/failed character maps
            // Need to early return, or we get issues with the `insert_many` later.
            return Ok(());
        }

        let to_insert = all_characters
            .into_iter()
            .map(|(line, character)| db::dialogue::ActiveModel {
                id: Default::default(),
                character_id: character.id.into_active_value(),
                dialogue_text: line.clone().into_active_value(),
            });

        let inserted_lines = db::dialogue::Entity::insert_many(to_insert)
            .on_conflict(
                OnConflict::columns([db::dialogue::Column::CharacterId, db::dialogue::Column::DialogueText])
                    .do_nothing()
                    .to_owned(),
            )
            .do_nothing()
            .exec(tx)
            .await?;

        tracing::trace!(?inserted_lines, "Inserted lines");

        Ok(())
    }

    /// Try map the given character to a voice in our backend.
    async fn map_character(&self, tx: &impl WriteConnection, character: &CharacterVoice) -> eyre::Result<CharacterRef> {
        // Assume male
        let char_gender = character.gender.unwrap_or_default();
        let char_name = &character.name;

        // First check if the character exists in our database
        let existing_voice = db::characters::Entity::find()
            .filter(db::characters::Column::CharacterName.eq(char_name))
            .filter(db::characters::Column::CharacterGender.eq(char_gender.to_db()))
            .one(tx)
            .await?;

        if let Some(voice) = existing_voice {
            Ok(voice)
        } else {
            // First check if a game specific voice exists with the same name as the given character
            let voice_ref = VoiceReference::game(char_name, self.game_data.game_name.clone());

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
                match char_gender {
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
                character_name: char_name.clone().into_active_value(),
                character_gender: char_gender.to_db().to_value().into_active_value(),
                voice_name: voice_to_use.name.into_active_value(),
                voice_location: voice_to_use.location.to_string_value().into_active_value(),
            };

            let out = to_insert.insert(tx).await?;

            Ok(out)
        }
    }
}
