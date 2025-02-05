use crate::{
    data::TtsModel,
    emotion::EmotionBackend,
    error::GameSessionError,
    postprocessing,
    postprocessing::AudioData,
    rvc_backends::{BackendRvcRequest, RvcCoordinator, RvcResult},
    session::{order_channel::OrderedReceiver, GameResult, GameSharedData},
    tts_backends::{BackendTtsRequest, BackendTtsResponse, TtsCoordinator, TtsResult},
    voice_manager::VoiceReference,
    PostProcessing, TtsResponse, TtsVoice, VoiceLine,
};
use eyre::{ContextCompat, WrapErr};
use itertools::Itertools;
use path_abs::PathOps;
use rand::{prelude::IteratorRandom, thread_rng};
use std::{format, path::PathBuf, sync::Arc, time::SystemTime, unimplemented, vec};
use tracing::Instrument;

pub type SingleRequest = (
    VoiceLine,
    Option<tokio::sync::oneshot::Sender<Arc<TtsResponse>>>,
    tracing::Span,
);

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

        self.data.save_cache().await?;
        self.data.save_state().await?;
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
                    tracing::warn!(
                        "A RVC post-process step was requested, but no provider is available to service it"
                    );
                    Ok(())
                }
                e => {
                    // First persist our data
                    tracing::error!(game=?self.data.game_data.game_name, "Stopping GameQueueActor actor due to unknown error");
                    self.data.save_cache().await?;
                    self.data.save_state().await?;
                    self.save_queue().await?;
                    // Then bail
                    eyre::bail!(e)
                }
            },
            _ => Ok(()),
        }
    }

    #[tracing::instrument(skip(self))]
    async fn handle_request(
        &mut self,
        next_item: VoiceLine,
        respond: Option<tokio::sync::oneshot::Sender<Arc<TtsResponse>>>,
    ) -> GameResult<()> {
        let game_response = Arc::new(self.cache_or_request(next_item).await?);
        if let Some(response_channel) = respond {
            // If the consumer drops the other end we don't care
            let _ = response_channel.send(game_response.clone());
        }

        Ok(())
    }

    /// Either use a cached TTS line, or generate a new one based on the given `voice_line`.
    #[tracing::instrument(skip(self))]
    async fn cache_or_request(&mut self, voice_line: VoiceLine) -> GameResult<TtsResponse> {
        // First check if we have a cache reference
        if let Some(response) = self.data.try_cache_retrieve(&voice_line).await? {
            return Ok(response);
        }

        tracing::debug!("No cache available, requesting from TTS");
        // If we want to use RVC we'll try and warm it up before the TTS request to save time
        if let Some(post) = &voice_line.post {
            if let Some(rvc) = &post.rvc {
                self.rvc.prepare_instance(rvc.high_quality).await?;
            }
        }

        let voice_to_use = match &voice_line.person {
            TtsVoice::ForceVoice(forced) => forced.clone(),
            TtsVoice::CharacterVoice(character) => self.data.map_character(character).await?,
        };
        let voice = self.data.voice_manager.get_voice(voice_to_use.clone())?;

        let emotion = self.emotion.classify_emotion([&voice_line.line])?[0];
        tracing::debug!(?emotion, "Identified emotion in line");

        let sample = voice
            .try_emotion_sample(emotion)?
            .next()
            .ok_or_else(|| GameSessionError::NoVoiceSamples {
                voice: voice.reference.name,
            })?
            .into_iter()
            .choose(&mut thread_rng())
            .context("No sample")?;

        let sample_path = sample.sample.clone();
        // TODO: Configurable language
        let request = BackendTtsRequest {
            gen_text: voice_line.line.clone(),
            language: "en".to_string(),
            voice_reference: vec![sample],
            speed: None,
        };
        tracing::info!("Going to request");
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
        tracing::info!("Transforming response");
        let out = self.transform_response(voice_to_use, voice_line, response).await?;
        // Once in a while save our line cache in case it crashes.
        self.generations_count += 1;
        if self.generations_count > 20 {
            self.generations_count = 0;
            self.save_queue().await?;
            self.data.save_cache().await?
        }

        Ok(out)
    }

    /// Perform post-processing on the newly generated raw TTS files.
    ///
    /// This includes but is not limited to, silence trimming, low/high-pass filters.
    #[tracing::instrument(skip_all)]
    async fn postprocess(
        &mut self,
        voice_line: &VoiceLine,
        voice_sample: PathBuf,
        post_processing: &PostProcessing,
        response: BackendTtsResponse,
    ) -> Result<BackendTtsResponse, GameSessionError> {
        let should_trim = post_processing.trim_silence;
        let should_normalise = post_processing.normalise;

        let timer = std::time::Instant::now();

        let (new_audio, destination_path) = match response.result.clone() {
            TtsResult::File(temp_path) => {
                // First we check with Whisper (if desired) matches our prompt.
                if let Some(percent) = post_processing.verify_percentage {
                    let score = self.tts.verify_prompt(&temp_path, &voice_line.line).await?;
                    tracing::trace!(?score, "Whisper TTS match");
                    // There will obviously be transcription errors, so we choose a relatively
                    if score < (percent as f32 / 100.0) {
                        return Err(GameSessionError::IncorrectGeneration);
                    }
                }
                tracing::info!("Post process");
                // Then we run our audio post-processing to clean it up for human ears.
                tokio::task::spawn_blocking(move || {
                    let mut raw_audio_data = wavers::Wav::<f32>::from_path(&temp_path)?;
                    let mut audio = AudioData::new(&mut raw_audio_data)?;
                    let mut sample_data: &mut [f32] = &mut audio.samples;

                    if should_trim {
                        // Basically any signal should count.
                        sample_data = postprocessing::trim_lead(sample_data, raw_audio_data.n_channels(), 0.01);
                    }
                    if should_normalise {
                        postprocessing::loudness_normalise(
                            sample_data,
                            raw_audio_data.sample_rate() as u32,
                            raw_audio_data.n_channels(),
                        );
                    }
                    audio.samples = sample_data.to_vec();

                    Ok::<_, eyre::Error>((audio, temp_path))
                })
                .await
                .context("Failed to join")??
            }
            TtsResult::Stream => unimplemented!("Streams are not yet supported"),
        };
        tracing::info!("Post process RVC");
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
                    data.write_to_wav_file(&destination_path)?;
                }
                RvcResult::Stream => unimplemented!("Streams are not yet supported"),
            }
        }

        let took = timer.elapsed();
        tracing::debug!(?took, "Finished post-processing");

        Ok(BackendTtsResponse {
            gen_time: response.gen_time + took,
            result: TtsResult::File(destination_path),
        })
    }

    /// Transfer a TTS file from its temporary directory to a permanent one and track its contents
    async fn transform_response(
        &mut self,
        voice: VoiceReference,
        line: VoiceLine,
        response: BackendTtsResponse,
    ) -> eyre::Result<TtsResponse> {
        let target_dir = self.data.lines_voice_path(&voice);
        tokio::fs::create_dir_all(&target_dir).await?;

        match response.result {
            TtsResult::File(temp_path) => {
                // TODO: Perhaps think of a better method to naming the generated lines
                let current_time = SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis();

                let file_name = format!("{}.wav", current_time);
                let target_voice_file = target_dir.join(&file_name);

                // Move the file to its permanent spot, and add it to the tracking
                tokio::fs::rename(&temp_path, &target_voice_file).await?;

                let voice_used = voice.name.clone();

                let old_value = self
                    .data
                    .line_cache
                    .lock()
                    .await
                    .voice_to_line
                    .entry(voice)
                    .or_default()
                    .insert(line.line.clone(), file_name);

                // Delete old lines if they existed (e.g., this was forcefully regenerated)
                if let Some(old_line) = old_value {
                    tracing::debug!(?old_line, "Deleting old line for new generation");
                    let target_voice_file = target_dir.join(&old_line);
                    // If it fails we don't care.
                    let _ = tokio::fs::remove_file(target_voice_file).await;
                }

                Ok(TtsResponse {
                    file_path: target_voice_file,
                    line,
                    voice_used,
                })
            }
            TtsResult::Stream => unimplemented!("Implement stream handling (still want to cache the output as well!)"),
        }
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
                let to_save: Vec<VoiceLine> = serde_json::from_slice(&std::fs::read(q_path)?)?;
                data.extend(to_save.into_iter().map(|v| (v, None, tracing::Span::current())));
                Ok::<_, eyre::Error>(())
            })
            .await
    }
}

const QUEUE_DATA: &str = "queue_backup.json";
