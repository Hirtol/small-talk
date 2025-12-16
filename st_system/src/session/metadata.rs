use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoicelineMetadata {
    pub model_used: String,
    pub source_file: String,
}

impl VoicelineMetadata {
    pub fn new(model_used: impl Into<String>, source_file: impl Into<String>) -> VoicelineMetadata {
        VoicelineMetadata {
            model_used: model_used.into(),
            source_file: source_file.into(),
        }
    }

    pub fn to_db(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }

    pub fn from_db(db: serde_json::Value) -> eyre::Result<Self> {
        Ok(serde_json::from_value(db)?)
    }
}