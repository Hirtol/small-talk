//! All content related to the back-end systems such as voice generation

use std::collections::HashMap;
pub use data::*;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::system::tts_backends::alltalk::AllTalkTTS;

mod tts_backends;
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

pub struct TtsSystem {
    backends: HashMap<TtsModel, AllTalkTTS>,
}

impl TtsSystem {
    pub fn new() -> Self {
        Self {
            backends: Default::default(),
        }
    }
}

pub mod dirs {
    use std::path::PathBuf;
    use path_abs::PathOps;
    use crate::config::Config;

    pub fn game_dir(conf: &Config, game_name: &str) -> PathBuf {
        conf.dirs.game_data_path().join(game_name)
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