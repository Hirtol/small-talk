use crate::error::FfiError;
use std::sync::Arc;
use system::StSystemFfi;

uniffi::setup_scaffolding!();

pub type Result<T> = std::result::Result<T, FfiError>;

pub mod error;
pub mod session;
pub mod system;
pub mod records;

#[uniffi::export]
pub fn create_st_system() -> Result<StSystemFfi> {
    let conf = Arc::new(st_http::config::initialise_config()?);
    let sys = system::create_tts_system(conf.clone())?;

    Ok(StSystemFfi {
        config: conf,
        system: sys,
    })
}
