use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Duration;
use eyre::Context;
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqliteSynchronous};
use sqlx::SqlitePool;
use st_db::DatabasePool;

pub type SessionDb = DatabasePool;

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
}

pub async fn initialise_database(db_cfg: DbConfig) -> eyre::Result<SessionDb> {
    std::fs::create_dir_all(db_cfg.db_path.parent().unwrap())?;

    let options = db_cfg
        .database_url()
        .parse::<SqliteConnectOptions>()?
        .foreign_keys(true)
        .journal_mode(SqliteJournalMode::Wal)
        .synchronous(SqliteSynchronous::Normal) // Since we're in WAL mode
        .pragma("wal_autocheckpoint", "1000")
        .busy_timeout(Duration::from_secs(10));

    let pool = DatabasePool::new_sqlite(
        options,
        db_cfg.max_connections_writer.get(),
        db_cfg.max_connections_reader.get(),
    )
        .await?;

    setup_db_schema(pool.get_sqlx_sqlite_writer()).await?;

    Ok(pool)
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