use std::sync::Arc;
use platform_dirs::AppDirs;
use tokio::sync::Notify;

pub mod setup;
pub mod telemetry;
pub mod config;
pub mod api;
pub mod system;

pub fn get_app_dirs() -> AppDirs {
    platform_dirs::AppDirs::new("SmallTalk".into(), false).expect("Couldn't find a home directory for config!")
}

/// A notifier to be able to shut down all systems appropriately, and in time.
pub fn get_quit_notifier() -> Arc<Notify> {
    Arc::new(Notify::new())
}