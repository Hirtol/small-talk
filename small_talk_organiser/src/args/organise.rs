use kalosm::sound::{WhisperBuilder, WhisperSource};
use kalosm_common::{Cache, ModelLoadingProgress};
use small_talk::{config::{Config, SharedConfig}, system::voice_manager::VoiceManager, BasicEmotion};
use std::{collections::HashMap, path::PathBuf};
use std::fs::File;
use std::io::BufReader;
use futures::StreamExt;
use itertools::Itertools;
use kalosm::sound::rodio::Decoder;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use small_talk::system::voice_manager::{VoiceDestination, VoiceSample};

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
                    queue
                        .entry(voice_name)
                        .or_default()
                        .push(parent_item.path())
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
                            queue
                                .entry(voice_name.clone())
                                .or_default()
                                .push(item.path())
                        } else {
                            tracing::debug!("Skipping: {:?} as it's not a WAV", item.path())
                        }
                    }
                }
            }
        }

        tracing::warn!("Using Whisper emotion detection, this is not perfect");

        // let cache = Cache::new(config.dirs.model_path().join("whisper"));
        // let whisper = WhisperBuilder::default()
        //     .with_source(WhisperSource::QuantizedDistilMediumEn)
        //     .with_cache(cache)
        //     .build_with_loading_handler(|progress| match progress {
        //         ModelLoadingProgress::Downloading {
        //             source,
        //             start_time,
        //             progress,
        //         } => {
        //             let progress = (progress * 100.0) as u32;
        //             let elapsed = start_time.elapsed().as_secs_f32();
        //             tracing::debug!("Downloading file {source} {progress}% ({elapsed}s)");
        //         }
        //         ModelLoadingProgress::Loading { progress } => {
        //             let progress = (progress * 100.0) as u32;
        //             tracing::debug!("Loading model {progress}%");
        //         }
        //     }).await.unwrap();
        
        let real_whisper = config.dirs.model_path().join("whisper").join("ggml").join("ggml-tiny.en-q8_0.bin");
        let ctx = WhisperContext::new_with_params(&real_whisper.to_string_lossy(), WhisperContextParameters::default())?;
        let mut state = ctx.create_state()?;
        let mut params = FullParams::new(SamplingStrategy::Greedy {best_of: 1});
        params.set_language(Some("en"));

        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        
        let total_samples_to_process = queue.values().map(|d| d.len()).sum::<usize>();
        
        tracing::info!(total_samples_to_process, "Will process samples");
        
        for (voice_name, samples) in queue {
            tracing::info!("Starting processing of Voice: {:?}", voice_name);
            for sample in samples {
                tracing::debug!("Handling sample: {:?}", sample);
                // let reader = BufReader::new(File::open(sample)?);
                // let decoder = Decoder::new_wav(reader)?;
                // let result = whisper.transcribe(decoder).unwrap();
                // let full_text = result.map(|seg| {
                //     tracing::trace!("Segment: {:?} - Conf: {:?}", seg.text(), seg.confidence());
                //     seg.text().to_string()
                // }).collect::<Vec<_>>().await.join("");
                
                let data = hound::WavReader::open(&sample)?.into_samples::<i16>().flatten().collect_vec();
                // we must convert to 16KHz mono f32 samples for the model
                // some utilities exist for this
                // note that you don't need to use these, you can do it yourself or any other way you want
                // these are just provided for convenience
                // SIMD variants of these functions are also available, but only on nightly Rust: see the docs
                let mut inter_samples = vec![Default::default(); data.len()];

                whisper_rs::convert_integer_to_float_audio(&data, &mut inter_samples)
                    .expect("failed to convert audio data");
                let samples = whisper_rs::convert_stereo_to_mono_audio(&inter_samples).unwrap_or(inter_samples);

                // now we can run the model
                // note the key we use here is the one we created above
                state
                    .full(params.clone(), &samples[..])
                    .expect("failed to run model");

                // fetch the results
                let num_segments = state
                    .full_n_segments()
                    .expect("failed to get number of segments");
                let mut full_text = String::new();
                for i in 0..num_segments {
                    let segment = state
                        .full_get_segment_text(i)
                        .expect("failed to get segment");
                    full_text += &segment;
                    let start_timestamp = state
                        .full_get_segment_t0(i)
                        .expect("failed to get segment start timestamp");
                    let end_timestamp = state
                        .full_get_segment_t1(i)
                        .expect("failed to get segment end timestamp");
                    tracing::info!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
                }
                
                tracing::debug!("Finished sample: {:?}", full_text);
                let sam = VoiceSample {
                    emotion: BasicEmotion::Neutral,
                    spoken_text: Some(full_text),
                    data: std::fs::read(sample)?,
                };
                voice_man.store_voice_samples(destination.clone(), &voice_name, vec![sam])?;
            }
        }
        
        Ok(())
    }
}
