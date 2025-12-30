use crate::{error::TtsError, timeout::DroppableState, voice_manager::FsVoiceSample};
use echo_tts::EchoTtsHandle;
use eyre::Context;
use indextts::IndexTtsHandle;
use st_audio::AudioData;
use st_data::TtsModel;
use st_ml::stt::WhisperTranscribe;
use std::{ops::DerefMut, path::PathBuf, sync::Arc, time::Duration};
use tokio::sync::Mutex;

pub mod chunking;
pub mod docker_backend;
pub mod echo_tts;
pub mod generic_backend;
pub mod indextts;

pub type Result<T> = std::result::Result<T, TtsError>;

/// The collection of TTS backend handles.
#[derive(Clone)]
pub struct TtsCoordinator {
    pub index_tts: Option<IndexTtsHandle>,
    pub echo_tts: Option<EchoTtsHandle>,
    pub whisper: Option<Arc<Mutex<WhisperTranscribe>>>,
}

impl TtsCoordinator {
    /// Create a new [TtsCoordinator]
    ///
    /// If no TtsBackend model is provided all requests will return with [TtsError::ModelNotInitialised].
    pub fn new(
        index_tts: Option<IndexTtsHandle>,
        echo_tts: Option<EchoTtsHandle>,
        whisper: Option<Arc<Mutex<WhisperTranscribe>>>,
    ) -> Self {
        Self {
            index_tts,
            echo_tts,
            whisper,
        }
    }

    /// Send a TTS request to the given model.
    #[tracing::instrument(skip(self))]
    pub async fn tts_request(&self, model: TtsModel, req: BackendTtsRequest) -> Result<BackendTtsResponse> {
        match model {
            TtsModel::IndexTts => {
                let Some(index) = &self.index_tts else {
                    return Err(TtsError::ModelNotInitialised { model });
                };

                Ok(index.submit_tts_request(req).await?)
            }

            TtsModel::EchoTts => {
                let Some(echo) = &self.echo_tts else {
                    return Err(TtsError::ModelNotInitialised { model });
                };

                Ok(echo.submit_tts_request(req).await?)
            }
        }
    }

    /// Check whether the given `wav` file contains speech data matching the `original_prompt`.
    /// We calculate the Levenshtein distance and calculate its ratio compared to the original prompt-length
    ///
    /// # Returns
    ///
    /// A score in the range [0..1], where a higher score is a closer match.
    pub async fn verify_prompt_path(&self, wav_file: impl Into<PathBuf>, original_prompt: &str) -> Result<f32> {
        let wav_file = wav_file.into();
        let mut reader: wavers::Wav<f32> = wavers::Wav::from_path(wav_file).context("Failed to read WAV file")?;

        self.verify_prompt(AudioData::new(&mut reader)?, original_prompt).await
    }

    /// Check whether the given `wav` file contains speech data matching the `original_prompt`.
    /// We calculate the Levenshtein distance and calculate its ratio compared to the original prompt-length
    ///
    /// # Returns
    ///
    /// A score in the range [0..1], where a higher score is a closer match.
    pub async fn verify_prompt(&self, audio_data: AudioData, original_prompt: &str) -> Result<f32> {
        let whisp_clone = self
            .whisper
            .clone()
            .ok_or_else(|| eyre::eyre!("Whisper is disabled in the config, can't verify"))?;

        let output = tokio::task::spawn_blocking(move || {
            let mut whisp = whisp_clone.blocking_lock();

            whisp.infer(&audio_data.samples, audio_data.n_channels, audio_data.sample_rate)
        })
        .await
        .map_err(|e| eyre::eyre!(e))??;

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
    pub result: TtsResult,
}

#[derive(Debug, Clone)]
pub enum TtsResult {
    /// FS location of the output
    File(PathBuf),
    Audio(AudioData),
    /// TODO, maybe
    Stream,
}
