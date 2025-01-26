use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use routes::config;

pub mod routes;
pub mod tts;

#[derive(Debug, JsonSchema, Serialize, Deserialize)]
pub struct Session {
    /// The game name for this particular session.
    pub id: String,
}