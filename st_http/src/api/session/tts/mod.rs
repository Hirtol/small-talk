use std::path::PathBuf;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use routes::config;
use st_system::{PostProcessing, RvcModel, RvcOptions, TtsVoice, VoiceLine};
use st_system::data::TtsModel;

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
            // For now assume these defaults.
            // TODO make these configurable.
            post: Some(PostProcessing {
                verify_percentage: None,
                trim_silence: true,
                normalise: true,
                rvc: Some(RvcOptions {
                    model: RvcModel::SeedVc,
                    high_quality: true,
                }),
            }),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ApiTtsResponse {
    pub file_path: PathBuf,
}