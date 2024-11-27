use std::path::PathBuf;
use std::time::Duration;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use routes::config;
use crate::system::{TtsModel, TtsVoice, VoiceLine};

pub mod routes;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ApiTtsRequest {
    pub line: String,
    /// The person who ought to voice the line
    pub person: TtsVoice,
    pub model: TtsModel,
    /// Force the generation of a new line, even if it already existed in the cache.
    pub force_generate: bool,
}

impl From<ApiTtsRequest> for VoiceLine {
    fn from(value: ApiTtsRequest) -> Self {
        Self {
            line: value.line,
            person: value.person,
            model: value.model,
            force_generate: value.force_generate,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ApiTtsResponse {
    pub file_path: PathBuf,
}