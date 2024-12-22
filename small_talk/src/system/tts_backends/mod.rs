use std::path::PathBuf;
use std::time::Duration;
use crate::system::tts_backends::alltalk::local::LocalAllTalkHandle;
use crate::system::{TtsModel};
use crate::system::voice_manager::FsVoiceSample;

pub mod alltalk;

/// The collection of TTS backend handles.
#[derive(Debug, Clone)]
pub struct TtsBackend {
    pub xtts: LocalAllTalkHandle,
    pub f5: LocalAllTalkHandle,
}

impl TtsBackend {
    pub fn new(xtts_all_talk: LocalAllTalkHandle, f5_all_talk: LocalAllTalkHandle) -> Self {
        Self {
            xtts: xtts_all_talk,
            f5: f5_all_talk,
        }
    }

    /// Send a TTS request to the given model.
    #[tracing::instrument(skip(self))]
    pub async fn tts_request(&self, model: TtsModel, req: BackendTtsRequest) -> eyre::Result<BackendTtsResponse> {
        match model {
            TtsModel::F5 => {
                self.f5.submit_tts_request(req).await
            }
            TtsModel::Xtts => {
                self.xtts.submit_tts_request(req).await
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BackendTtsRequest {
    /// Text to generate
    pub gen_text: String,
    /// Language of the generation task
    pub language: String,
    /// Path reference(s) to the voice samples to use for generating.
    /// If only one sample is needed simply pick the first
    ///
    /// These should not be moved/deleted, if needed simply hardlink these to a new location 
    pub voice_reference: Vec<FsVoiceSample>,
    /// The playback speed of the voice
    pub speed: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct BackendTtsResponse {
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