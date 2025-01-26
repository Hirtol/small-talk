//! All content related to the back-end systems such as voice generation

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use platform_dirs::AppDirs;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use crate::config::TtsSystemConfig;
use crate::rvc_backends::RvcBackend;
use crate::session::GameSessionHandle;
use crate::tts_backends::{TtsBackend, BackendTtsRequest, BackendTtsResponse};
use crate::voice_manager::VoiceManager;

pub use crate::data::*;
use crate::emotion::EmotionBackend;

pub mod tts_backends;
pub mod rvc_backends;
pub mod data;
pub mod session;
pub mod voice_manager;
pub mod utils;
pub mod playback;
pub mod config;
pub mod timeout;
pub mod emotion;
mod postprocessing;
mod error;

pub type TtsSystemHandle = Arc<TtsSystem>;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub enum TtsModel {
    E2,
    Xtts,
}

/// Single place collating all active backends of our system.
pub struct TtsSystem {
    config: Arc<TtsSystemConfig>,
    // We don't use papaya here to prevent race conditions
    sessions: Arc<Mutex<HashMap<String, GameSessionHandle>>>,
    voice_man: Arc<VoiceManager>,
    tts: TtsBackend,
    rvc: RvcBackend,
    emotion: EmotionBackend,
}

impl TtsSystem {
    pub fn new(config: Arc<TtsSystemConfig>, tts_backend: TtsBackend, rvc_backend: RvcBackend, emotion_backend: EmotionBackend) -> Self {
        Self {
            emotion: emotion_backend,
            config: config.clone(),
            sessions: Arc::new(Default::default()),
            voice_man: Arc::new(VoiceManager::new(config)),
            tts: tts_backend,
            rvc: rvc_backend,
        }
    }

    #[tracing::instrument(skip(self))]
    pub async fn get_or_start_session(&self, game: &str) -> eyre::Result<GameSessionHandle> {
        let mut pin = self.sessions.lock().await;

        if let Some(game_ses) = pin.get(game) {
            if game_ses.is_alive() {
                return Ok(game_ses.clone())
            }
        }
        let new_session = GameSessionHandle::new(game, self.voice_man.clone(), self.tts.clone(), self.rvc.clone(), self.emotion.clone(), self.config.clone()).await?;
        pin.insert(game.into(), new_session.clone());

        Ok(new_session)
    }

    /// Stop the given session if it was started
    ///
    /// Does nothing if no session for `game` was currently operational.
    #[tracing::instrument(skip(self))]
    pub async fn stop_session(&self, game: &str) -> eyre::Result<()> {
        let mut pin = self.sessions.lock().await;
        let _ = pin.remove(game);

        Ok(())
    }

    /// Shut the entire TTS backend down.
    pub async fn shutdown(&self) -> eyre::Result<()> {
        self.sessions.lock().await.clear();
        // TODO: Add a 'shutdown' message to the actors for proper shutdown and remove the below
        tokio::time::sleep(Duration::from_secs(1)).await;
        Ok(())
    }
}

pub fn get_app_dirs() -> AppDirs {
    platform_dirs::AppDirs::new("SmallTalk".into(), false).expect("Couldn't find a home directory for config!")
}
