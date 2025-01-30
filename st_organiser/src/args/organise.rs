use eyre::ContextCompat;
use futures::StreamExt;
use itertools::Itertools;
use path_abs::PathInfo;
use st_http::{
    config::{Config, SharedConfig},
};
use st_ml::{
    embeddings::LLamaEmbedder,
    emotion_classifier::{BasicEmotion, BasicEmotionClassifier},
};
use std::{
    collections::HashMap,
    ffi::{c_char, c_void, CString},
    fs::File,
    io::BufReader,
    path::PathBuf,
};
use st_system::voice_manager::{VoiceDestination, VoiceManager, VoiceSample};

#[derive(clap::Args, Debug)]
pub struct OrganiseCommand {
    /// The path in which to find voice samples.
    ///
    /// Either expects directories of WAV files (where the directory name is the 'voice' name,
    /// or single WAV files which will be organised later.
    sample_path: PathBuf,
    /// Destination, 'global' for a global voice available to all games.
    #[clap(short, default_value = "global")]
    destination: String,
}

impl OrganiseCommand {
    #[tracing::instrument(skip_all, fields(self.sample_path))]
    pub async fn run(self, config: SharedConfig) -> eyre::Result<()> {
        let mut voice_man = VoiceManager::new(config.dirs.clone());

        let destination = if self.destination == "global" {
            VoiceDestination::Global
        } else {
            VoiceDestination::Game(self.destination)
        };
        let mut queue: HashMap<String, Vec<PathBuf>> = HashMap::new();

        for parent_item in std::fs::read_dir(&self.sample_path)?.flatten() {
            if parent_item.file_type()?.is_file() {
                let is_wav = parent_item.path().extension().unwrap().to_string_lossy() == "wav";

                if is_wav {
                    let voice_name = parent_item.path().file_stem().unwrap().to_string_lossy().to_string();
                    tracing::debug!(?voice_name, path=?parent_item.path(), "Queueing voice sample");
                    queue.entry(voice_name).or_default().push(parent_item.path())
                } else {
                    tracing::debug!("Skipping: {:?} as it's not a WAV or directory", parent_item.path())
                }
            } else {
                let voice_name = parent_item.path().file_stem().unwrap().to_string_lossy().to_string();
                for item in std::fs::read_dir(parent_item.path())?.flatten() {
                    if item.file_type()?.is_file() {
                        let is_wav = item.path().extension().unwrap().to_string_lossy() == "wav";

                        if is_wav {
                            tracing::debug!(?voice_name, path=?item.path(), "Queueing voice sample");
                            queue.entry(voice_name.clone()).or_default().push(item.path())
                        } else {
                            tracing::debug!("Skipping: {:?} as it's not a WAV", item.path())
                        }
                    }
                }
            }
        }

        tracing::warn!("Using Whisper emotion detection, this is not perfect");

        let device = st_ml::burn::backend::ndarray::NdArrayDevice::default();
        let mut emotion_classifier: BasicEmotionClassifier<st_ml::CpuBackend> = BasicEmotionClassifier::new(
            &config.dirs.emotion_classifier_model,
            &config.dirs.bert_embeddings_model,
            device,
        )?;

        let whisper_path = &config.dirs.whisper_model;
        let mut whisper = st_ml::stt::WhisperTranscribe::new(whisper_path, 12)?;

        let total_samples_to_process = queue.values().map(|d| d.len()).sum::<usize>();

        tracing::info!(total_samples_to_process, "Will process samples");

        for (voice_name, samples) in queue {
            tracing::info!("Starting processing of Voice: {:?}", voice_name);
            for sample in samples {
                tracing::debug!("Handling sample: {:?}", sample);
                let existing_transcript = sample.with_extension("txt");
                let full_text = if existing_transcript.exists() {
                    tracing::trace!("Found existing transcription, using it instead of Whisper");
                    std::fs::read_to_string(existing_transcript)?
                } else {
                    whisper.transcribe_file(&sample)?
                };

                let emotion = emotion_classifier
                    .infer([&full_text.trim()])?
                    .into_iter()
                    .next()
                    .context("Impossible")?;

                tracing::debug!("Finished sample, emotion: {emotion:?} for text: {full_text:?}");

                let sam = VoiceSample {
                    emotion,
                    spoken_text: Some(full_text.trim().into()),
                    data: std::fs::read(sample)?,
                };

                voice_man.store_voice_samples(destination.clone(), &voice_name, vec![sam])?;
            }
        }

        Ok(())
    }
}
