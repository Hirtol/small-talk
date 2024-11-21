use eyre::ContextCompat;
use futures::StreamExt;
use itertools::Itertools;
use kalosm::sound::{rodio::Decoder, WhisperBuilder, WhisperSource};
use kalosm_common::{Cache, ModelLoadingProgress};
use path_abs::PathInfo;
use small_talk::{
    config::{Config, SharedConfig},
    system::voice_manager::{VoiceDestination, VoiceManager, VoiceSample},
};
use small_talk_ml::{
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
use faster_whisper_rs::config::{WhisperConfig, WhisperConfigBuilder};

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

pub struct ExternalWhisperLib {
    whisper_create: libloading::Symbol<'static, unsafe extern "C" fn(*const c_char) -> *mut c_void>,
    whisper_parse: libloading::Symbol<'static, unsafe extern "C" fn(*mut c_void, *const i16, usize) -> CString>,
    whisper_free: libloading::Symbol<'static, unsafe extern "C" fn(*const c_void)>,
}

impl ExternalWhisperLib {
    pub fn new(path: &str) -> eyre::Result<ExternalWhisperLib> {
        unsafe {
            let whisper_lib = libloading::Library::new(path)?;
            let whisper_create: libloading::Symbol<unsafe extern "C" fn(*const c_char) -> *mut c_void> =
                whisper_lib.get(b"create_whisper")?;
            let whisper_parse: libloading::Symbol<unsafe extern "C" fn(*mut c_void, *const i16, usize) -> CString> =
                whisper_lib.get(b"parse_tokens")?;
            let whisper_free: libloading::Symbol<unsafe extern "C" fn(*const c_void)> =
                whisper_lib.get(b"free_whisper")?;

            let out = ExternalWhisperLib {
                whisper_create: std::mem::transmute(whisper_create),
                whisper_parse: std::mem::transmute(whisper_parse),
                whisper_free: std::mem::transmute(whisper_free),
            };

            // We can just forget it to make them static
            std::mem::forget(whisper_lib);
            Ok(out)
        }
    }
}

impl OrganiseCommand {
    #[tracing::instrument(skip_all, fields(self.sample_path))]
    pub async fn run(self, config: SharedConfig) -> eyre::Result<()> {
        let external_whisp = ExternalWhisperLib::new("./small_talk_whisper.dll")?;
        let mut voice_man = VoiceManager::new(config.clone());

        let destination = if self.destination == "global" {
            VoiceDestination::Global
        } else {
            VoiceDestination::Game(&self.destination)
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

        let classifier_path = config.dirs.model_path().join("text_emotion_classifier");
        let device = small_talk_ml::burn::backend::ndarray::NdArrayDevice::default();
        let mut emotion_classifier: BasicEmotionClassifier<small_talk_ml::CpuBackend> = BasicEmotionClassifier::new(
            classifier_path.join("classifier_head"),
            classifier_path.join("ggml-model-Q4_k.gguf"),
            device,
        )?;
        
        let cfg = WhisperConfigBuilder::default().language("en".to_string()).prefix("".to_string()).build()?;
        let faster_whisper = faster_whisper_rs::WhisperModel::new("distil-small.en".to_string(), "cuda".to_string(), "int8_float16".to_string(), cfg).unwrap();

        let total_samples_to_process = queue.values().map(|d| d.len()).sum::<usize>();

        tracing::info!(total_samples_to_process, "Will process samples");

        for (voice_name, samples) in queue {
            tracing::info!("Starting processing of Voice: {:?}", voice_name);
            for sample in samples {
                tracing::debug!("Handling sample: {:?}", sample);

                let data = hound::WavReader::open(&sample)?
                    .into_samples::<i16>()
                    .flatten()
                    .collect_vec();

                if data.is_empty() {
                    tracing::error!("Empty sample set");
                    continue;
                }

                let full_text = faster_whisper.transcribe(sample.to_str().context("Fail")?.to_string()).unwrap().to_string();

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
