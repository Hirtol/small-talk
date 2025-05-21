use crate::{
    data::TtsModel, emotion::EmotionBackend, error::GameSessionError,
    rvc_backends::{BackendRvcRequest, RvcCoordinator, RvcResult},
    session::{
        db, db::DbEnumHelper, linecache::LineCacheEntry, order_channel::OrderedReceiver, GameResult, GameSharedData,
    },
    tts_backends::{BackendTtsRequest, BackendTtsResponse, TtsCoordinator, TtsResult},
    voice_manager::VoiceReference,
    PostProcessing,
    TtsResponse,
    TtsVoice,
    VoiceLine,
};
use eyre::{ContextCompat, WrapErr};
use itertools::Itertools;
use path_abs::PathOps;
use rand::prelude::IteratorRandom;
use sea_orm::{ActiveModelTrait, IntoActiveValue};
use st_db::{DbId, WriteConnection, WriteTransaction};
use std::{format, path::PathBuf, sync::Arc, time::SystemTime, unimplemented, vec};
use tracing::Instrument;
use crate::audio::postprocessing;
use crate::audio::audio_data::AudioData;

pub type SingleRequest = (
    VoiceLineRequest,
    Option<tokio::sync::oneshot::Sender<Arc<TtsResponse>>>,
    tracing::Span,
);

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
pub struct VoiceLineRequest {
    pub text: String,
    pub speaker: VoiceReference,
    pub model: TtsModel,
    /// Optional audio post-processing
    pub post: Option<PostProcessing>,
}

impl VoiceLineRequest {
    pub fn to_line_cache(&self) -> LineCacheEntry {
        LineCacheEntry {
            text: self.text.clone(),
            voice: self.speaker.clone(),
        }
    }
}

pub(super) struct GameQueueActor {
    pub tts: TtsCoordinator,
    pub rvc: RvcCoordinator,
    pub emotion: EmotionBackend,
    pub data: Arc<GameSharedData>,
    pub queue: OrderedReceiver<SingleRequest>,
    pub priority: OrderedReceiver<SingleRequest>,

    pub generations_count: usize,
}

impl GameQueueActor {
    #[tracing::instrument(skip(self))]
    pub async fn run(mut self) -> eyre::Result<()> {
        // Ignore failed reads.
        let _ = self.read_queue().await;

        loop {
            tokio::select! {
                biased;

                Some(next_item) = self.priority.recv() => {
                    self.handle_request_err(next_item).await?
                },
                Some(next_item) = self.queue.recv() => {
                    tracing::trace!("Remaining items in queue: {}", self.queue.len().await);
                    self.handle_request_err(next_item).await?
                },
                else => break
            }
        }

        self.save_queue().await?;

        Ok(())
    }

    async fn handle_request_err(&mut self, (next_item, respond, span): SingleRequest) -> eyre::Result<()> {
        match self.handle_request(next_item, respond).instrument(span).await {
            Err(e) => match e {
                GameSessionError::VoiceDoesNotExist { voice } => {
                    tracing::warn!("Ignoring request which requested non-existent voice: {voice}");
                    Ok(())
                }
                GameSessionError::NoVoiceSamples { voice } => {
                    tracing::warn!("Ignoring request which requested voice with no samples: {voice}");
                    Ok(())
                }
                GameSessionError::IncorrectGeneration => {
                    tracing::warn!("Skipping line request after too many generation failure");
                    Ok(())
                }
                GameSessionError::Timeout => {
                    tracing::warn!("Skipping line request due to timeout");
                    Ok(())
                }
                GameSessionError::InvalidText { txt } => {
                    tracing::warn!(?txt, "Received invalid text in request");
                    Ok(())
                }
                GameSessionError::ModelNotInitialised { model } => {
                    tracing::warn!(
                        ?model,
                        "A model was requested, but no provider is available to service it"
                    );
                    Ok(())
                }
                GameSessionError::RvcNotInitialised => {
                    tracing::warn!("A RVC post-process step was requested, but no provider is available to service it");
                    Ok(())
                }
                e => {
                    // First persist our data
                    tracing::error!(game=?self.data.game_data.game_name, "Stopping GameQueueActor actor due to unknown error");
                    self.save_queue().await?;
                    // Then bail
                    eyre::bail!(e)
                }
            },
            _ => Ok(()),
        }
    }

    #[tracing::instrument(skip(self, respond))]
    async fn handle_request(
        &mut self,
        next_item: VoiceLineRequest,
        respond: Option<tokio::sync::oneshot::Sender<Arc<TtsResponse>>>,
    ) -> GameResult<()> {
        // First check if we have a cache reference
        let tts_response = if let Some(cache) = self
            .data
            .line_cache
            .try_retrieve(self.data.game_db.reader(), next_item.to_line_cache())
            .await?
        {
            cache
        } else {
            self.execute_request(next_item).await?
        };

        if let Some(response_channel) = respond {
            // If the consumer drops the other end we don't care
            let _ = response_channel.send(Arc::new(tts_response));
        }

        Ok(())
    }

    /// Generate a new line based on the given `voice_line`.
    #[tracing::instrument(skip(self))]
    async fn execute_request(&mut self, voice_line: VoiceLineRequest) -> GameResult<TtsResponse> {
        // If we want to use RVC we'll try and warm it up before the TTS request to save time
        if let Some(post) = &voice_line.post {
            if let Some(rvc) = &post.rvc {
                self.rvc.prepare_instance(rvc.high_quality).await?;
            }
        }

        let voice = self.data.voice_manager.get_voice(voice_line.speaker.clone())?;

        let emotion = self.emotion.classify_emotion([&voice_line.text])?[0];
        tracing::debug!(?emotion, "Identified emotion in line");

        let sample = voice
            .try_emotion_sample(emotion)?
            .next()
            .ok_or_else(|| GameSessionError::NoVoiceSamples {
                voice: voice.reference.name,
            })?
            .into_iter()
            .choose(&mut rand::rng())
            .context("No sample")?;

        let sample_path = sample.sample.clone();
        // TODO: Configurable language
        let request = BackendTtsRequest {
            gen_text: voice_line.text.clone(),
            language: "en".to_string(),
            voice_reference: vec![sample],
            speed: None,
        };

        let mut response = None;
        for i in 0..3 {
            let response_gen = self.tts.tts_request(voice_line.model, request.clone()).await?;
            response = if let Some(post) = voice_line.post.as_ref() {
                match self
                    .postprocess(&voice_line, sample_path.clone(), post, response_gen)
                    .await
                {
                    Ok(response) => Some(response),
                    Err(GameSessionError::IncorrectGeneration) => {
                        tracing::trace!(attempt = i, "Failed to generate voice line, retrying");
                        // Retry with a new generation
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            } else {
                Some(response_gen)
            };

            break;
        }
        let Some(response) = response else {
            return Err(GameSessionError::IncorrectGeneration);
        };

        let out = self
            .finalise_response(self.data.game_db.writer(), voice_line.speaker, voice_line.text, response)
            .await?;

        Ok(out)
    }

    /// Perform post-processing on the newly generated raw TTS files.
    ///
    /// This includes but is not limited to, silence trimming, low/high-pass filters.
    #[tracing::instrument(skip_all)]
    async fn postprocess(
        &mut self,
        voice_line: &VoiceLineRequest,
        voice_sample: PathBuf,
        post_processing: &PostProcessing,
        response: BackendTtsResponse,
    ) -> Result<BackendTtsResponse, GameSessionError> {
        let should_trim = post_processing.trim_silence;
        let should_normalise = post_processing.normalise;

        let timer = std::time::Instant::now();

        let mut original_audio_data = match response.result.clone() {
            TtsResult::Audio(audio_data) => {
                audio_data
            }
            TtsResult::File(temp_path) => {
                let mut raw_audio_data = wavers::Wav::<f32>::from_path(&temp_path).context("Failed to read TTS file")?;
                AudioData::new(&mut raw_audio_data)?
            }
            TtsResult::Stream => unimplemented!("Todo")
        };

        let mut new_audio = {
            // First we check with Whisper (if desired) matches our prompt.
            if let Some(percent) = post_processing.verify_percentage {
                let score = self.tts.verify_prompt(original_audio_data.clone(), &voice_line.text).await?;
                tracing::trace!(?score, "Whisper TTS match");
                // There will obviously be transcription errors, so we choose a relatively
                if score < (percent as f32 / 100.0) {
                    return Err(GameSessionError::IncorrectGeneration);
                }
            }

            // Then we run our audio post-processing to clean it up for human ears.
            tokio::task::spawn_blocking(move || {
                let mut sample_data: &mut [f32] = &mut original_audio_data.samples;

                if should_trim {
                    // Basically any signal should count.
                    sample_data = postprocessing::trim_lead(sample_data, original_audio_data.n_channels, 0.01);
                }
                if should_normalise {
                    postprocessing::loudness_normalise(
                        sample_data,
                        original_audio_data.sample_rate,
                        original_audio_data.n_channels,
                    );
                }

                Ok::<_, eyre::Error>(original_audio_data)
            })
                .await
                .context("Failed to join")??
        };

        if let Some(rvc) = &post_processing.rvc {
            let req = BackendRvcRequest {
                audio: new_audio,
                target_voice: voice_sample,
            };
            let out = self.rvc.rvc_request(req, rvc.high_quality).await?;

            match out.result {
                RvcResult::Wav(mut data) => {
                    // Silence is still cut out, but we might need to re-normalise.
                    if should_normalise {
                        postprocessing::loudness_normalise(&mut data.samples, data.sample_rate, data.n_channels);
                    }
                    new_audio = data;
                }
                RvcResult::Stream => unimplemented!("Streams are not yet supported"),
            }
        }

        let took = timer.elapsed();
        tracing::debug!(?took, "Finished post-processing");

        Ok(BackendTtsResponse {
            gen_time: response.gen_time + took,
            result: TtsResult::Audio(new_audio),
        })
    }

    /// Transfer a TTS file from its temporary directory to a permanent one and track its contents
    async fn finalise_response(
        &self,
        tx: &impl WriteConnection,
        voice: VoiceReference,
        text: String,
        response: BackendTtsResponse,
    ) -> eyre::Result<TtsResponse> {
        let target_dir = self.data.line_cache.lines_voice_path(&voice);
        tokio::fs::create_dir_all(&target_dir).await?;

        let (target_voice_file, file_name) = match response.result {
            TtsResult::Audio(data) => {
                let current_time = SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis();
                let file_name = {
                    let mut new_name = std::ffi::OsString::from(current_time.to_string());
                    new_name.push(".wav");
                    new_name.to_string_lossy().into_owned()
                };
                let target_voice_file = target_dir.join(&*file_name);

                data.write_to_wav_file(&target_voice_file)?;

                (target_voice_file, file_name)
            }
            TtsResult::File(temp_path) => {
                // TODO: Perhaps think of a better method to naming the generated lines
                let current_time = SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis();
                let file_name = {
                    let ext = temp_path.extension();
                    let mut new_name = std::ffi::OsString::from(current_time.to_string());
                    new_name.push(".");
                    if let Some(ext) = ext {
                        new_name.push(ext);
                    } else {
                        // Assume wav
                        new_name.push("wav");
                    }

                    new_name.to_string_lossy().into_owned()
                };
                let target_voice_file = target_dir.join(&*file_name);

                // Move the file to its permanent spot, and add it to the tracking
                tokio::fs::rename(&temp_path, &target_voice_file).await?;

                (target_voice_file, file_name)
            }
            TtsResult::Stream => unimplemented!("Implement stream handling (still want to cache the output as well!)"),
        };

        let voice_line_db = db::voice_lines::ActiveModel {
            id: Default::default(),
            dialogue_text: text.clone().into_active_value(),
            voice_name: voice.name.clone().into_active_value(),
            voice_location: voice.location.clone().to_string_value().into_active_value(),
            file_name: file_name.into_active_value(),
        };

        // DB Constraint replaces line if it already exists TODO: Reap unreferenced voice files
        voice_line_db.insert(tx).await?;

        Ok(TtsResponse {
            file_path: target_voice_file,
            line: text,
            voice_used: voice,
        })
    }

    async fn save_queue(&self) -> eyre::Result<()> {
        let q_path = self
            .data
            .config
            .game_dir(&self.data.game_data.game_name)
            .join(QUEUE_DATA);
        let to_serialize = self
            .queue
            .modify_contents(|data| data.iter().map(|v| &v.0).cloned().collect_vec())
            .await;

        let writer = std::io::BufWriter::new(std::fs::File::create(q_path)?);
        Ok(serde_json::to_writer_pretty(writer, &to_serialize)?)
    }

    async fn read_queue(&self) -> eyre::Result<()> {
        let q_path = self
            .data
            .config
            .game_dir(&self.data.game_data.game_name)
            .join(QUEUE_DATA);

        self.queue
            .modify_contents(|data| {
                let to_save: Vec<VoiceLineRequest> = serde_json::from_slice(&std::fs::read(q_path)?)?;
                data.extend(to_save.into_iter().map(|v| (v, None, tracing::Span::current())));
                Ok::<_, eyre::Error>(())
            })
            .await
    }
}

const QUEUE_DATA: &str = "queue_backup.json";
