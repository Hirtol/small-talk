use std::ops::DerefMut;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::Duration;
use tokio::sync::Mutex;
use st_ml::stt::WhisperTranscribe;
use crate::error::TtsError;
use crate::tts_backends::alltalk::local::LocalAllTalkHandle;
use crate::timeout::DroppableState;
use crate::data::TtsModel;
use crate::voice_manager::FsVoiceSample;

pub mod alltalk;

pub type Result<T> = std::result::Result<T, TtsError>;

/// The collection of TTS backend handles.
#[derive(Clone)]
pub struct TtsCoordinator {
    pub xtts: Option<LocalAllTalkHandle>,
    whisper: Arc<Mutex<Option<WhisperTranscribe>>>,
    whisper_path: PathBuf,
}

impl TtsCoordinator {
    /// Create a new [TtsCoordinator]
    ///
    /// If no TtsBackend model is provided all requests will return with [TtsError::ModelNotInitialised].
    pub fn new(xtts_all_talk: Option<LocalAllTalkHandle>, whisper_path: PathBuf) -> Self {
        Self {
            xtts: xtts_all_talk,
            whisper: Arc::new(Mutex::new(None)),
            whisper_path,
        }
    }

    /// Send a TTS request to the given model.
    #[tracing::instrument(skip(self))]
    pub async fn tts_request(&self, model: TtsModel, req: BackendTtsRequest) -> Result<BackendTtsResponse> {
        match model {
            TtsModel::Xtts => {
                let Some(xtts) = &self.xtts else {
                    return Err(TtsError::ModelNotInitialised {
                        model
                    })
                };
                Ok(xtts.submit_tts_request(req).await?)
            }
        }
    }

    /// Check whether the given `wav` file contains speech data matching the `original_prompt`.
    /// We calculate the Levenshtein distance and calculate its ratio compared to the original prompt-length
    ///
    /// # Returns
    ///
    /// A score in the range [0..1], where a higher score is a closer match.
    pub async fn verify_prompt(&self, wav_file: impl Into<PathBuf>, original_prompt: &str) -> Result<f32> {
        let whisp_clone = self.whisper.clone();
        let wav_file = wav_file.into();
        let whisp_path = self.whisper_path.clone();

        let output = tokio::task::spawn_blocking(move || {
            let mut whisp = whisp_clone.blocking_lock();

            match whisp.deref_mut() {
                None => {
                    let cpu_threads = std::thread::available_parallelism()?.get() / 2;
                    let mut model = WhisperTranscribe::new(whisp_path, cpu_threads as u16)?;
                    let output = model.transcribe_file(wav_file);
                    *whisp = Some(model);
                    output
                }
                Some(model) => model.transcribe_file(wav_file)
            }
        }).await.map_err(|e| eyre::eyre!(e))??;
        // Can cause problems if we don't remove these for short quotes.
        let original_without_quotes = original_prompt.trim_start_matches('"').trim_end_matches('"');
        let leven = strsim::levenshtein(&output, original_without_quotes);
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