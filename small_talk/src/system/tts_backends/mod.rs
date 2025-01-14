use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use small_talk_ml::stt::WhisperTranscribe;
use crate::system::tts_backends::alltalk::local::LocalAllTalkHandle;
use crate::system::{TtsModel};
use crate::system::voice_manager::FsVoiceSample;

pub mod alltalk;

/// The collection of TTS backend handles.
#[derive(Clone)]
pub struct TtsBackend {
    pub xtts: LocalAllTalkHandle,
    pub e2: LocalAllTalkHandle,
    pub whisper: Arc<Mutex<WhisperTranscribe>>,
}

impl TtsBackend {
    pub fn new(xtts_all_talk: LocalAllTalkHandle, f5_all_talk: LocalAllTalkHandle, whisper_transcribe: WhisperTranscribe) -> Self {
        Self {
            xtts: xtts_all_talk,
            e2: f5_all_talk,
            whisper: Arc::new(Mutex::new(whisper_transcribe)),
        }
    }

    /// Send a TTS request to the given model.
    #[tracing::instrument(skip(self))]
    pub async fn tts_request(&self, model: TtsModel, req: BackendTtsRequest) -> eyre::Result<BackendTtsResponse> {
        match model {
            TtsModel::E2 => {
                self.e2.submit_tts_request(req).await
            }
            TtsModel::Xtts => {
                self.xtts.submit_tts_request(req).await
            }
        }
    }

    /// Check whether the given `wav` file contains speech data matching the `original_prompt`.
    /// We calculate the Levenshtein distance and calculate its ratio compared to the original prompt-length
    ///
    /// # Returns
    ///
    /// A score in the range [0..1], where a higher score is a closer match.
    pub async fn verify_prompt(&self, wav_file: impl Into<PathBuf>, original_prompt: &str) -> eyre::Result<f32> {
        let whisp_clone = self.whisper.clone();
        let wav_file = wav_file.into();

        let output = tokio::task::spawn_blocking(move || {
            let mut whisp = whisp_clone.blocking_lock();

            whisp.transcribe_file(wav_file)
        }).await??;
        let leven = strsim::levenshtein(&output, original_prompt);
        let ratio = leven as f32 / original_prompt.chars().count() as f32;
        Ok(1.0 - ratio)
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