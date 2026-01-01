//! All content related to the back-end systems such as voice generation

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use platform_dirs::AppDirs;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use crate::config::TtsSystemConfig;
use crate::rvc_backends::RvcCoordinator;
use crate::session::GameSessionHandle;
use crate::tts_backends::TtsCoordinator;
use crate::voice_manager::VoiceManager;

pub use st_data::*;
use crate::emotion::{EmotionBackend, EmotionCoordinator};

pub mod tts_backends;
pub mod rvc_backends;
pub mod session;
pub mod voice_manager;
pub mod utils;
pub mod config;
pub mod timeout;
pub mod emotion;
pub mod error;

pub type TtsSystemHandle = Arc<TtsSystem>;

/// Single place collating all active backends of our system.
pub struct TtsSystem {
    config: Arc<TtsSystemConfig>,
    // We don't use papaya here to prevent race conditions
    sessions: Arc<Mutex<HashMap<String, GameSessionHandle>>>,
    voice_man: Arc<VoiceManager>,
    tts: TtsCoordinator,
    rvc: RvcCoordinator,
    emotion: EmotionCoordinator,
    root_cancel: CancellationToken,
}

impl TtsSystem {
    pub fn new(config: Arc<TtsSystemConfig>, tts_backend: TtsCoordinator, rvc_backend: RvcCoordinator, emotion_backend: EmotionCoordinator, token: CancellationToken) -> Self {
        Self {
            emotion: emotion_backend,
            config: config.clone(),
            sessions: Arc::new(Default::default()),
            voice_man: Arc::new(VoiceManager::new(config)),
            tts: tts_backend,
            rvc: rvc_backend,
            root_cancel: token,
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
        let session_token = self.root_cancel.child_token();
        let new_session = GameSessionHandle::new(game, self.voice_man.clone(), self.tts.clone(), self.rvc.clone(), self.emotion.clone(), self.config.clone(), session_token).await?;
        pin.insert(game.into(), new_session.clone());

        Ok(new_session)
    }

    /// Stop the given session if it was started
    ///
    /// Does nothing if no session for `game` was currently operational.
    #[tracing::instrument(skip(self))]
    pub async fn stop_session(&self, game: &str) -> eyre::Result<()> {
        let mut pin = self.sessions.lock().await;
        let session = pin.remove(game);

        if let Some(session) = session {
            session.token.cancel()
        }

        Ok(())
    }

    /// Shut the entire TTS backend down.
    pub async fn shutdown(&self) -> eyre::Result<()> {
        self.sessions.lock().await.clear();
        Ok(())
    }
}

pub fn get_app_dirs() -> AppDirs {
    platform_dirs::AppDirs::new("SmallTalk".into(), false).expect("Couldn't find a home directory for config!")
}
