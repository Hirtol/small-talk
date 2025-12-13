use crate::{
    error::TtsError,
    timeout::GcCell,
    tts_backends::{
        generic_backend::{ActiveTtsState, ActiveTtsStateConfig, ReadyTtsApi, TtsApi, TtsBackendMessage}, indextts::{
            api::{IndexTtsAPI, IndexTtsRequest},
            local::LocalIndexState,
            text_processing::TextProcessor,
        }, BackendTtsRequest,
        BackendTtsResponse,
        TtsResult,
    },
};
use eyre::{ContextCompat, WrapErr};
use std::time::Duration;

pub mod api;
mod local;

pub use crate::tts_backends::indextts::{api::IndexTtsApiConfig, local::LocalIndexTtsConfig};

type IndexTts = ReadyTtsApi<IndexTtsAPI>;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexTtsConfig {
    pub state: ActiveTtsStateConfig<LocalIndexTtsConfig, IndexTtsApiConfig>,
    pub state_timeout: Duration,
}

impl Default for IndexTtsConfig {
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
pub struct IndexTtsHandle {
    pub send: tokio::sync::mpsc::UnboundedSender<TtsBackendMessage>,
}

impl IndexTtsHandle {
    /// Create and start a new [IndexTtsActor] actor, returning the cloneable handle to the actor in the process.
    pub fn new(config: IndexTtsConfig) -> eyre::Result<Self> {
        let term = papaya::HashMap::from([
            ("tiefling".to_string(), "teefling".to_string()),
            ("No.".into(), "No .".into()),
        ]);

        let (send, recv) = tokio::sync::mpsc::unbounded_channel();
        let actor = IndexTtsActor {
            text_processor: TextProcessor::new(term),
            state: GcCell::new(config.state_timeout),
            config,
            recv,
        };

        tokio::task::spawn(async move {
            if let Err(e) = actor.run().await {
                tracing::error!("LocalIndexTts stopped with error: {e}");
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

struct IndexTtsActor {
    text_processor: TextProcessor,
    config: IndexTtsConfig,
    state: GcCell<ActiveTtsState<LocalIndexState, IndexTts>>,
    recv: tokio::sync::mpsc::UnboundedReceiver<TtsBackendMessage>,
}

impl IndexTtsActor {
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
                            tracing::trace!("Stopping LocalIndexTts actor as channel was closed");
                            self.state.kill_state().await?;
                            break
                        },
                    }
                },
                _ = self.state.timeout_future() => {
                    tracing::debug!("Timeout expired, dropping local IndexTts state");
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

                let req = IndexTtsRequest {
                    text: self.text_processor.process(request.gen_text),
                    wav_file_bytes: voice_sample.data().await?,
                };

                let now = std::time::Instant::now();
                let mut tts_response = tokio::time::timeout(Duration::from_secs(40), state.tts(req))
                    .await
                    .context("Timeout elapsed")??;
                let took = now.elapsed();

                // IndexTTS generates a high-pitch crackle at and above the ~11Khz range. We apply a 10500 Hz low-pass filter to remove this crackle.
                // (10500 instead of 11000 as our filtering crate isn't great)
                tts_response.lowpass_filter(10500.);

                let _ = response.send(BackendTtsResponse {
                    gen_time: took,
                    result: TtsResult::Audio(tts_response),
                });

                tracing::trace!(?took, "Finished handling of TTS request");
            }
        }
        Ok(())
    }
}

mod text_processing {
    //! Index-TTS has a few pronunciation peculiarities which we need to handle by preprocessing text:
    //! 1. Conjunctions with a dash (e.g., 'barely-there') should have the dash removed or the pronunciation will have a long pause.
    //! 2. Certain words need a literal writing (e.g., 'tieflings' -> 'teeflings') in order to have a correct pronunciation.

    use papaya::HashMap;

    pub struct TextProcessor {
        replace_tokens: HashMap<String, String>,
        dash_replace: regex::Regex,
        apostrophe_replace: regex::Regex,
    }

    impl TextProcessor {
        pub fn new(tokens: HashMap<String, String>) -> Self {
            Self {
                replace_tokens: tokens,
                dash_replace: regex::Regex::new(r"(\w+)-(\w+)").unwrap(),
                apostrophe_replace: regex::Regex::new(r"(?i)\b(there|where)'s\b").unwrap(),
            }
        }

        pub fn process(&self, text: impl AsRef<str>) -> String {
            let stack = text.as_ref();

            let dash_replaced = self.dash_replace.replace_all(stack, "$1 $2").into_owned();
            let mut dash_replaced = self
                .apostrophe_replace
                .replace_all(&dash_replaced, "$1 is")
                .into_owned();

            // TODO: For now a _very_ inefficient replacement, but later on use [AhoCorasick::replace_all]
            for (token, replacement) in self.replace_tokens.pin().iter() {
                dash_replaced = dash_replaced.replace(token, replacement)
            }

            dash_replaced
        }
    }
}
