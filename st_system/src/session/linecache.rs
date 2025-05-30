use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use sea_orm::{ColumnTrait, EntityTrait, IntoActiveValue, QuerySelect, QueryTrait};
use serde::de::Error;
use st_db::{ReadConnection, WriteConnection};
use crate::config::TtsSystemConfig;
use crate::session::db;
use crate::session::db::SessionDb;
use crate::TtsResponse;
use crate::voice_manager::{VoiceDestination, VoiceReference};
use sea_orm::QueryFilter;

#[derive(Debug)]
pub struct LineCacheEntry {
    pub text: String,
    pub voice: VoiceReference,
}

#[derive(Debug, Clone)]
pub struct LineCache {
    game_db: SessionDb,
    game_name: String,
    config: Arc<TtsSystemConfig>
}

impl LineCache {
    pub fn new(game_name: String, config: Arc<TtsSystemConfig>, game_db: SessionDb) -> Self {
        Self {
            game_db,
            game_name,
            config,
        }
    }

    /// Attempt to retrieve an existing TTS response from the database
    ///
    /// If no cached line is found will return `Ok(None)`.
    pub async fn try_retrieve(&self, tx: &impl ReadConnection, entry: LineCacheEntry) -> eyre::Result<Option<TtsResponse>> {
        let out = db::voice_lines::Entity::find()
            .filter(db::lines_table_voice_line_condition(&entry.text, &entry.voice))
            .one(tx)
            .await?;

        Ok(out.map(|v| {
            let target_voice_file = self.lines_voice_path(&entry.voice).join(v.file_name);

            TtsResponse {
                file_path: target_voice_file,
                line: entry.text,
                voice_used: entry.voice,
            }
        }))
    }

    /// Remove all cached lines matching the given `items`.
    pub async fn invalidate_cache_lines(&self, tx: &impl WriteConnection, items: impl IntoIterator<Item=LineCacheEntry>) -> eyre::Result<()> {
        // N queries, could be more efficient...
        for item in items {
            self.invalidate_cache_line(tx, &item).await?;
        }

        Ok(())
    }

    async fn invalidate_cache_line(&self, tx: &impl WriteConnection, line: &LineCacheEntry) -> eyre::Result<()> {
        use st_db::entity::*;
        tracing::debug!(?line, "Invalidating line");
        let deleted_models = voice_lines::Entity::delete_many()
            .filter(
                voice_lines::Column::Id.in_subquery(
                    voice_lines::Entity::find()
                        .select_only()
                        .column(voice_lines::Column::Id)
                        .filter(db::lines_table_voice_line_condition(&line.text, &line.voice))
                        .into_query(),
                ),
            )
            .exec_with_returning(tx)
            .await?;
        // Delete old voice files that are no longer needed.
        for model in deleted_models {
            let target_voice_file = self.lines_voice_path(&line.voice).join(model.file_name);
            if let Err(e) = tokio::fs::remove_file(&target_voice_file).await {
                tracing::warn!(?target_voice_file, ?e, "Failed to delete invalidated voice line")
            }
        }

        Ok(())
    }

    /// Update the given cache entry with a new file name.
    pub async fn update_cache_line_path(&self, entry: LineCacheEntry, new_file_name: String) -> eyre::Result<()> {
        let model = db::voice_lines::ActiveModel {
            file_name: new_file_name.into_active_value(),
            .. Default::default()
        };

        db::voice_lines::Entity::update_many()
            .set(model)
            .filter(db::lines_table_voice_line_condition(&entry.text, &entry.voice))
            .exec(self.game_db.writer())
            .await?;

        Ok(())
    }

    /// Return all lines saved in this [LineCache].
    pub async fn all_lines(&self) -> eyre::Result<HashMap<VoiceReference, Vec<db::voice_lines::Model>>> {
        let lines = db::voice_lines::Entity::find().all(self.game_db.reader()).await?;

        let mut map: HashMap<VoiceReference, Vec<db::voice_lines::Model>> = HashMap::new();

        for line in lines {
            let key = VoiceReference {
                name: line.voice_name.clone(),
                location: line.voice_location.clone().into(),
            };

            map.entry(key)
                .or_default()
                .push(line);
        }

        Ok(map)
    }

    /// Returns the path to the directory containing all spoken dialogue by the given [VoiceReference]
    pub fn lines_voice_path(&self, voice: &VoiceReference) -> PathBuf {
        self.line_cache_path().join(&voice.name)
    }

    fn line_cache_path(&self) -> PathBuf {
        self.config.game_lines_cache(&self.game_name)
    }
}