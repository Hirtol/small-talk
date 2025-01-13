use std::path::{Path, PathBuf};
use path_abs::PathOps;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TtsSystemConfig {
    /// Directory storing all game data, including global voices and game specific data.
    pub appdata_dir: PathBuf,
    /// Path to the Whisper model. Should be a valid GGUF/GGML model.
    pub whisper_model: PathBuf,
    /// Path to the emotion classifier model
    pub emotion_classifier_model: PathBuf,
    /// Path to the BERT-based model providing text embeddings.
    ///
    /// Should be GGUF/GGML.
    pub bert_embeddings_model: PathBuf,
}

impl Default for TtsSystemConfig {
    fn default() -> Self {
        let app_dir = crate::get_app_dirs().config_dir;
        let appdata_dir = app_dir.join("appdata");
        let models_dir = appdata_dir.join("models");
        Self {
            whisper_model: models_dir.join("whisper").join("ggml-medium-q5_0.bin"),
            emotion_classifier_model: models_dir.join("text_emotion_classifier").join("classifier_head"),
            bert_embeddings_model: models_dir.join("text_emotion_classifier").join("ggml-model-Q4_k.gguf"),
            appdata_dir,
        }
    }
}

impl TtsSystemConfig {
    pub fn game_dir(&self, game_name: &str) -> PathBuf {
        self.appdata_dir.join("game_data").join(game_name)
    }

    pub fn game_dir_lines_cache(&self, game_dir: &Path) -> PathBuf {
        game_dir.join("lines")
    }

    pub fn game_lines_cache(&self, game_name: &str) -> PathBuf {
        self.game_dir_lines_cache(&self.game_dir(game_name))
    }

    pub fn game_voice(&self, game_name: &str) -> PathBuf {
        self.game_dir(game_name).join("voices")
    }

    pub fn global_voice(&self) -> PathBuf {
        // We pretend 'global' is a game everyone can see.
        self.game_voice("global")
    }
}