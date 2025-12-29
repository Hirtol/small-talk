use crate::error::FfiError;
use std::sync::Arc;
use std::time::Duration;
use system::StSystemFfi;

uniffi::setup_scaffolding!();

pub type Result<T> = std::result::Result<T, FfiError>;

pub mod error;
pub mod session;
pub mod system;
pub mod records;

#[uniffi::export(async_runtime="tokio")]
pub async fn create_st_system() -> Result<StSystemFfi> {
    let conf = Arc::new(st_http::config::initialise_config()?);
    let sys = system::create_tts_system(conf.clone())?;

    tokio::time::sleep(Duration::from_millis(1)).await;
    tracing::info!("Finishing create_st_system");

    Ok(StSystemFfi {
        config: conf,
        system: sys,
    })
}
