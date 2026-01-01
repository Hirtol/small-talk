use crate::error::FfiError;
use st_application::SmallTalkApplication;
use std::{sync::Arc, time::Duration};
use std::path::PathBuf;
use tracing_subscriber::util::SubscriberInitExt;
use st_application::config::SmallTalkConfig;
use system::StSystemFfi;

uniffi::setup_scaffolding!();

pub type Result<T> = std::result::Result<T, FfiError>;

pub mod error;
pub mod records;
pub mod session;
pub mod system;
mod telemetry;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, Default)]
pub struct SmallTalkUniffiConfig {
    #[serde(default)]
    pub small_talk: SmallTalkConfig,
}

#[uniffi::export(async_runtime = "tokio")]
pub async fn create_st_system() -> Result<StSystemFfi> {
    let possible_conf = st_application::config::initialise_config::<SmallTalkUniffiConfig>();
    tracing::debug!("Final config: {:#?}", possible_conf);
    let conf = Arc::new(possible_conf?.small_talk);
    let sys = SmallTalkApplication::new(&conf).await?;

    Ok(StSystemFfi {
        config: conf,
        system: sys,
    })
}

#[uniffi::export]
pub fn initialise_file_log(path: String) {
    let subscriber = telemetry::create_subscriber(
        "WARN,reqwest=DEBUG,st_uniffi=TRACE,st_system=TRACE,st_http=TRACE,st_ml=TRACE,sqlx=WARN,hyper=WARN",
        path
    );
    let _ = subscriber.try_init();
}
