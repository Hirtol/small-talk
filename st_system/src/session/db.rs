use crate::voice_manager::VoiceReference;
use eyre::Context;
use sea_orm::{sea_query::StringLen, ActiveEnum, ColumnTrait, DeriveActiveEnum, EnumIter};
use sea_query::{Condition, IntoCondition};
use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqliteSynchronous},
    SqlitePool,
};
use st_db::DatabasePool;
use std::{num::NonZeroU32, path::PathBuf, time::Duration};

pub use st_db::entity::*;
use crate::VoiceLine;

pub type SessionDb = DatabasePool;

pub fn lines_table_voice_line_condition(line: &str, voice: &VoiceReference) -> Condition {
    voice_lines::Column::DialogueText.eq(line)
        .into_condition()
        .add(lines_table_voice_reference_condition(voice))
}

pub fn lines_table_voice_reference_condition(voice: &VoiceReference) -> Condition {
    use st_db::entity::voice_lines::*;
    Column::VoiceName
        .eq(&voice.name)
        .and(Column::VoiceLocation.eq(voice.location.to_string_value()))
        .into_condition()
}

pub trait DbEnumHelper<V: ActiveEnum> {
    fn to_db_enum_value(self) -> V::Value;
}

pub trait DbEnumOptionalHelper<V: ActiveEnum> {
    fn to_db_enum_value(self) -> Option<V::Value>;
}

impl<V: ActiveEnum, P: Into<V>> DbEnumHelper<V> for P {
    fn to_db_enum_value(self) -> V::Value {
        let target_db: V = self.into();
        target_db.to_value()
    }
}

#[derive(EnumIter, DeriveActiveEnum, Copy, Clone, Debug)]
#[sea_orm(rs_type = "String", db_type = "String(StringLen::None)", rename_all = "camelCase")]
pub enum DatabaseGender {
    Male,
    Female,
}

impl DatabaseGender {
    pub fn to_string(&self) -> String {
        self.to_value()
    }
}

#[derive(Clone, Debug, Hash, PartialOrd, PartialEq, Eq)]
pub struct DbConfig {
    /// Full path to the DB file.
    pub db_path: PathBuf,
    pub in_memory: bool,
    /// The amount of connections to the database.
    pub max_connections_reader: NonZeroU32,
    /// In theory Sqlx should've fixed 'database locked' errors, but those still seem to occur with multiple mutations.
    /// Hence why we use just a single connection by default.
    pub max_connections_writer: NonZeroU32,
}

impl DbConfig {
    /// Turn the config settings into a valid DB url.
    pub fn database_url(&self) -> String {
        if self.in_memory {
            "sqlite::memory:".to_string()
        } else {
            format!(
                "sqlite://{}?mode=rwc",
                self.db_path
                    .to_str()
                    .expect("Invalid database path specified in config or ENV")
            )
        }
    }

    pub async fn initialise_database(self) -> eyre::Result<SessionDb> {
        std::fs::create_dir_all(self.db_path.parent().unwrap())?;

        let options = self
            .database_url()
            .parse::<SqliteConnectOptions>()?
            .foreign_keys(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Normal) // Since we're in WAL mode
            .pragma("wal_autocheckpoint", "1000")
            .busy_timeout(Duration::from_secs(10));

        let pool = DatabasePool::new_sqlite(
            options,
            self.max_connections_writer.get(),
            self.max_connections_reader.get(),
        )
        .await?;

        setup_db_schema(pool.get_sqlx_sqlite_writer()).await?;

        Ok(pool)
    }
}

async fn setup_db_schema(db: &SqlitePool) -> eyre::Result<()> {
    tracing::info!("Running game database migrations");

    st_db::migrate()
        .run(db)
        .await
        .context("Error running database migrations")?;

    tracing::info!("Completed game database setup");

    Ok(())
}
