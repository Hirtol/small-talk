//! All content related to the back-end systems such as voice generation

use std::collections::HashMap;
pub use data::*;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use crate::config::SharedConfig;
use crate::system::session::GameSessionHandle;
use crate::system::tts_backends::{TtsBackend, BackendTtsRequest, BackendTtsResponse};
use crate::system::voice_manager::VoiceManager;

pub mod tts_backends;
pub mod data;
pub mod session;
pub mod voice_manager;
pub mod utils;

pub type TtsSystemHandle = Arc<TtsSystem>;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub enum TtsModel {
    F5,
    Xtts,
}

/// Single place collating all active backends of our system.
pub struct TtsSystem {
    config: SharedConfig,
    // We don't use papaya here to prevent race conditions
    sessions: Arc<Mutex<HashMap<String, GameSessionHandle>>>,
    voice_man: Arc<VoiceManager>,
    tts: TtsBackend,
}

impl TtsSystem {
    pub fn new(config: SharedConfig, tts_backend: TtsBackend) -> Self {
        Self {
            config: config.clone(),
            sessions: Arc::new(Default::default()),
            voice_man: Arc::new(VoiceManager::new(config)),
            tts: tts_backend,
        }
    }

    pub async fn get_or_start_session(&self, game: &str) -> eyre::Result<GameSessionHandle> {
        let mut pin = self.sessions.lock().await;
        
        if let Some(game_ses) = pin.get(game) {
            Ok(game_ses.clone())
        } else {
            let new_session = GameSessionHandle::new(game, self.voice_man.clone(), self.tts.clone(), self.config.clone()).await?;
            pin.insert(game.into(), new_session.clone());
            
            Ok(new_session)
        }
    }
    /// Send a TTS request to the given model.
    pub async fn tts_request(&self, model: TtsModel, req: BackendTtsRequest) -> eyre::Result<BackendTtsResponse> {
        todo!()
    }
}

pub mod dirs {
    use std::path::{Path, PathBuf};
    use path_abs::PathOps;
    use crate::config::Config;

    pub fn game_dir(conf: &Config, game_name: &str) -> PathBuf {
        conf.dirs.game_data_path().join(game_name)
    }
    
    pub fn game_dir_lines_cache(game_dir: &Path) -> PathBuf {
        game_dir.join("lines")
    }
    
    pub fn game_lines_cache(conf: &Config, game_name: &str) -> PathBuf {
        game_dir_lines_cache(&game_dir(conf, game_name))
    }
    
    pub fn game_voice(conf: &Config, game_name: &str) -> PathBuf {
        game_dir(conf, game_name).join("voices")
    }
    
    pub fn global_voice(conf: &Config) -> PathBuf {
        conf.dirs.appdata.join("global").join("voices")
    }

    pub fn game_output(conf: &Config, game_name: &str) -> PathBuf {
        game_dir(conf, game_name).join("output")
    }
}