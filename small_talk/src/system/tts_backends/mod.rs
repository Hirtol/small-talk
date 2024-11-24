use std::path::PathBuf;
use std::time::Duration;

pub mod alltalk;

#[derive(Debug, Clone)]
pub struct TtsRequest {
    /// Text to generate
    pub gen_text: String,
    /// Language of the generation task
    pub language: String,
    /// Path reference(s) to the voice samples to use for generating.
    /// If only one sample is needed simply pick the first
    ///
    /// These should not be moved/deleted, if needed simply hardlink these to a new location 
    pub voice_reference: Vec<PathBuf>,
    /// The playback speed of the voice
    pub speed: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct TtsResponse {
    /// How long it took to generate the response
    pub gen_time: Duration,
    pub result: TtsResult
}

#[derive(Debug, Clone)]
pub enum TtsResult {
    /// FS location of the output
    File(PathBuf),
    /// TODO, maybe
    Stream
}