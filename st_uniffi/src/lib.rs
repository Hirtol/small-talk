use crate::error::FfiError;
use st_application::SmallTalkApplication;
use std::{sync::Arc, time::Duration};
use system::StSystemFfi;

uniffi::setup_scaffolding!();

pub type Result<T> = std::result::Result<T, FfiError>;

pub mod error;
pub mod records;
pub mod session;
pub mod system;

#[uniffi::export(async_runtime = "tokio")]
pub async fn create_st_system() -> Result<StSystemFfi> {
    let conf = Arc::new(st_application::config::initialise_config()?);
    let sys = SmallTalkApplication::new(&conf).await?;

    tokio::time::sleep(Duration::from_millis(1)).await;
    tracing::info!("Finishing create_st_system");

    Ok(StSystemFfi {
        config: conf,
        system: sys,
    })
}
