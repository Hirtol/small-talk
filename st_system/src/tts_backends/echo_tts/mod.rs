use crate::tts_backends::{echo_tts::api::{EchoTtsAPI}, BackendTtsRequest, BackendTtsResponse, TtsResult};
use std::time::Duration;
use eyre::{ContextCompat, WrapErr};
use local::LocalEchoTtsState;
use crate::error::TtsError;
use crate::timeout::GcCell;
use crate::tts_backends::chunking::Chunk;
use crate::tts_backends::echo_tts::api::EchoTtsRequest;
use crate::tts_backends::echo_tts::text_processing::TextProcessor;
use crate::tts_backends::generic_backend::{ActiveTtsState, ActiveTtsStateConfig, ReadyTtsApi, TtsApi, TtsBackendMessage};

type EchoTts = ReadyTtsApi<EchoTtsAPI>;

pub mod api;
pub mod local;

pub use crate::tts_backends::echo_tts::{api::EchoTtsApiConfig, local::LocalEchoTtsConfig};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EchoTtsConfig {
    pub state: ActiveTtsStateConfig<LocalEchoTtsConfig, EchoTtsApiConfig>,
    pub state_timeout: Duration,
}

impl Default for EchoTtsConfig {
    fn default() -> Self {
        Self {
            state: ActiveTtsStateConfig::Local {
                local_config: Default::default(),
            },
            state_timeout: Duration::from_secs(1800),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EchoTtsHandle {
    pub send: tokio::sync::mpsc::UnboundedSender<TtsBackendMessage>,
}

impl EchoTtsHandle {
    /// Create and start a new [EchoTtsActor] actor, returning the cloneable handle to the actor in the process.
    pub fn new(config: EchoTtsConfig) -> eyre::Result<Self> {
        let term = papaya::HashMap::from([
            ("Kenabres".to_string(), "Kenaabres".to_string()),
            ("worldwound".to_string(), "world wound".to_string()),
            ("Worldwound".to_string(), "World wound".to_string()),
        ]);

        let (send, recv) = tokio::sync::mpsc::unbounded_channel();
        let actor = EchoTtsActor {
            text_processor: TextProcessor::new(term),
            state: GcCell::new(config.state_timeout),
            config,
            recv,
        };

        tokio::task::spawn(async move {
            if let Err(e) = actor.run().await {
                tracing::error!("LocalEchoTts stopped with error: {e}");
            }
        });

        Ok(Self { send })
    }

    pub async fn start_instance(&self) -> eyre::Result<()> {
        Ok(self.send.send(TtsBackendMessage::StartInstance)?)
    }

    pub async fn stop_instance(&self) -> eyre::Result<()> {
        Ok(self.send.send(TtsBackendMessage::StopInstance)?)
    }

    pub async fn submit_tts_request(&self, request: BackendTtsRequest) -> eyre::Result<BackendTtsResponse> {
        let (send, recv) = tokio::sync::oneshot::channel();
        self.send.send(TtsBackendMessage::TtsRequest(request, send))?;

        Ok(recv.await?)
    }
}

struct EchoTtsActor {
    text_processor: TextProcessor,
    config: EchoTtsConfig,
    state: GcCell<ActiveTtsState<LocalEchoTtsState, EchoTts>>,
    recv: tokio::sync::mpsc::UnboundedReceiver<TtsBackendMessage>,
}

impl EchoTtsActor {

    /// Start the actor, this future should be `tokio::spawn`ed.
    ///
    /// It will automatically drop the internal state if it hasn't been accessed in a while to preserve memory.
    #[tracing::instrument(skip(self))]
    pub async fn run(mut self) -> Result<(), TtsError> {
        loop {
            tokio::select! {
                msg = self.recv.recv() => {
                    // Have to pattern match here, as we want this `select!` to stop if the channel is closed, and not hang
                    // on our timeout
                    match msg {
                        Some(msg) => match self.handle_message(msg).await {
                            Ok(_) => {}
                            e => return e
                        },
                        None => {
                            tracing::trace!("Stopping LocalEchoTts actor as channel was closed");
                            self.state.kill_state().await?;
                            break
                        },
                    }
                },
                _ = self.state.timeout_future() => {
                    tracing::debug!("Timeout expired, dropping local EchoTts state");
                    // Drop the state, killing the sub-process
                    // Safe to do as we know that it won't be generating for us since we have exclusive access.
                    self.state.kill_state().await?
                }
                else => break,
            }
        }

        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn handle_message(&mut self, message: TtsBackendMessage) -> Result<(), TtsError> {
        match message {
            TtsBackendMessage::StartInstance => {
                self.state.get_state(&self.config.state).await?;
            }
            TtsBackendMessage::StopInstance => {
                self.state.kill_state().await?;
            }
            TtsBackendMessage::TtsRequest(mut request, response) => {
                let state = self.state.get_state(&self.config.state).await?;
                let voice_sample = request.voice_reference.pop().context("No voice sample")?;
                let voice_data = voice_sample.data().await?;
                let preprocessed_text = self.text_processor.process(request.gen_text);
                let chunked = crate::tts_backends::chunking::chunk_text(&preprocessed_text, 70, 500);

                let mut all_chunks = Vec::new();
                let now = std::time::Instant::now();

                for chunk in chunked {
                    let req = EchoTtsRequest {
                        sequence_length: Some(Self::get_expected_sequence_length(&chunk)),
                        num_steps: Some(30),
                        text: chunk.text,
                        wav_file_bytes: voice_data.clone(),
                    };
                    tracing::trace!(?req, "Sending the following request to echo");
                    let tts_response = tokio::time::timeout(Duration::from_secs(40), state.tts(req)).await.context("Timeout elapsed")??;
                    all_chunks.push(tts_response);
                }

                let took = now.elapsed();
                let mut stitched = crate::audio::stitching::stitch_with_gaps(all_chunks.into_iter(), Duration::from_millis(100))?;
                stitched.lowpass_filter(10500.);

                let _ = response.send(BackendTtsResponse {
                    gen_time: took,
                    result: TtsResult::Audio(stitched),
                });

                tracing::trace!(?took, "Finished handling of TTS request");
            }
        }
        Ok(())
    }

    /// For really low-cost chunks we can optimise inference substantially by shortening the length of the inferred audio.
    /// The max of `640` is equivalent to ~30 seconds of audio, and decreases proportionally.
    ///
    /// This is rather pessimistic, lower values are possible, but then you run the risk of cut-off audio
    fn get_expected_sequence_length(chunk: &Chunk) -> usize {
        match chunk.cost {
            0..50 => 130,
            50..100 => 213,
            100..250 => 416,
            250.. => 640
        }
    }
}


mod text_processing {
    use papaya::HashMap;

    pub struct TextProcessor {
        replace_tokens: HashMap<String, String>,
    }

    impl TextProcessor {
        pub fn new(tokens: HashMap<String, String>) -> Self {
            Self { replace_tokens: tokens }
        }

        pub fn process(&self, text: impl Into<String>) -> String {
            let mut stack = text.into();

            // TODO: For now a _very_ inefficient replacement, but later on use [AhoCorasick::replace_all]
            for (token, replacement) in self.replace_tokens.pin().iter() {
                stack = stack.replace(token, replacement)
            }

            stack
        }
    }
}