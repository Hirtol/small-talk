use eyre::{Context, ContextCompat};
use std::{
    path::{Path, PathBuf},
    process::Stdio,
};
use std::time::Duration;
use bollard::container::StartContainerOptions;
use bollard::Docker;
use bollard::models::ContainerSummary;
use process_wrap::tokio::TokioChildWrapper;
use tokio::{
    process::{Child, Command},
};
use tokio::time::error::Elapsed;
use crate::error::{RvcError, TtsError};
use crate::timeout::{DroppableState, GcCell};
use crate::tts_backends::{BackendTtsRequest, BackendTtsResponse, TtsResult};
use crate::tts_backends::indextts::api::{IndexTtsApiConfig, IndexTtsRequest};
use crate::tts_backends::indextts::IndexTts;
use crate::tts_backends::indextts::text_processing::TextProcessor;

const INDEX_TTS_DEFAULT_PORT: u16 = 11996;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LocalIndexTtsConfig {
    pub image_name: String,
    pub timeout: Duration
}

impl Default for LocalIndexTtsConfig {
    fn default() -> Self {
        Self {
            image_name: "hirtol/index-tts-llvm:latest".to_string(),
            timeout: std::time::Duration::from_secs(1800),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LocalIndexHandle {
    pub send: tokio::sync::mpsc::UnboundedSender<IndexMessage>,
}

#[derive(Debug)]
pub enum IndexMessage {
    /// Request the immediate start of the child process
    StartInstance,
    /// Request the immediate stop of the child process
    StopInstance,
    TtsRequest(BackendTtsRequest, tokio::sync::oneshot::Sender<BackendTtsResponse>),
}

impl LocalIndexHandle {
    /// Create and start a new [LocalIndexTts] actor, returning the cloneable handle to the actor in the process.
    pub fn new(config: LocalIndexTtsConfig) -> eyre::Result<Self> {
        let term = papaya::HashMap::from([("tiefling".to_string(), "teefling".to_string())]);


        // Small amount before we exert back-pressure
        let (send, recv) = tokio::sync::mpsc::unbounded_channel();
        let actor = LocalIndexTts {
            text_processor: TextProcessor::new(term),
            state: GcCell::new(config.timeout),
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
        Ok(self.send.send(IndexMessage::StartInstance)?)
    }

    pub async fn stop_instance(&self) -> eyre::Result<()> {
        Ok(self.send.send(IndexMessage::StopInstance)?)
    }

    pub async fn submit_tts_request(&self, request: BackendTtsRequest) -> eyre::Result<BackendTtsResponse> {
        let (send, recv) = tokio::sync::oneshot::channel();
        self.send.send(IndexMessage::TtsRequest(request, send))?;

        Ok(recv.await?)
    }
}

struct LocalIndexTts {
    text_processor: TextProcessor,
    config: LocalIndexTtsConfig,
    state: GcCell<TemporaryState>,
    recv: tokio::sync::mpsc::UnboundedReceiver<IndexMessage>,
}

struct TemporaryState {
    tts: IndexTts,
    daemon: Docker,
    docker_container: ContainerSummary,
}

impl LocalIndexTts {

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
    async fn handle_message(&mut self, message: IndexMessage) -> Result<(), TtsError> {
        match message {
            IndexMessage::StartInstance => {
                self.state.get_state(&self.config).await?;
            }
            IndexMessage::StopInstance => {
                self.state.kill_state().await?;
            }
            IndexMessage::TtsRequest(mut request, response) => {
                let state = self.state.get_state(&self.config).await?;
                let voice_sample = request.voice_reference.pop().context("No voice sample")?;

                let req = IndexTtsRequest {
                    text: self.text_processor.process(request.gen_text),
                    wav_file_bytes: voice_sample.data().await?,
                };

                let now = std::time::Instant::now();
                let mut tts_response = tokio::time::timeout(Duration::from_secs(40), state.tts.api.tts(req)).await.context("Timeout elapsed")??;
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

impl DroppableState for TemporaryState {
    type Context = LocalIndexTtsConfig;

    async fn initialise_state(context: &Self::Context) -> eyre::Result<Self> {
        #[tracing::instrument]
        async fn start_indextts(daemon: &Docker) -> eyre::Result<ContainerSummary> {
            tracing::debug!("Attempting to start IndexTts process");
            let container = docker::find_or_create_container(daemon, "small-talk-index-tts-vllm").await?;

            daemon.start_container(container.id.as_deref().unwrap(), None::<StartContainerOptions<String>>).await?;
            // Need to query again as we might get a randomly assigned IP address
            let final_container = docker::find_or_create_container(daemon, "small-talk-index-tts-vllm").await?;

            Ok(final_container)
        }

        let daemon = bollard::Docker::connect_with_local_defaults()?;
        let container = start_indextts(&daemon).await?;

        let container_port = if let Some(ports) = &container.ports {
            ports.first().and_then(|p| p.public_port).unwrap_or(INDEX_TTS_DEFAULT_PORT)
        } else {
            INDEX_TTS_DEFAULT_PORT
        };
        let api_address = format!("http://localhost:{container_port}");
        tracing::debug!(?api_address, "Started IndexTts container");

        let api = IndexTts::new(IndexTtsApiConfig {
            address: url::Url::parse(&api_address)?,
        }).await?;

        Ok(TemporaryState {
            tts: api,
            daemon,
            docker_container: container,
        })
    }

    async fn on_kill(&mut self) -> eyre::Result<()> {
        self.daemon.stop_container(self.docker_container.id.as_deref().unwrap(), None).await?;
        Ok(())
    }
}

mod docker {
    use std::collections::HashMap;
    use bollard::container::{Config, CreateContainerOptions, ListContainersOptions};
    use bollard::Docker;
    use bollard::image::CreateImageOptions;
    use bollard::models::{ContainerSummary, DeviceRequest, HostConfig};
    use eyre::{ContextCompat};
    use crate::tts_backends::indextts::local::INDEX_TTS_DEFAULT_PORT;

    const INDEX_DOCKER_IMAGE: &str = "hirtol/index-tts-llvm:latest";

    macro_rules! hashmap {
        ($( $key: expr => $val: expr ),* $(,)?) => {{
            let mut map = std::collections::HashMap::new();
            $( map.insert($key, $val); )*
            map
        }};
    }

    pub async fn find_or_create_container(daemon: &Docker, name: &str) -> eyre::Result<ContainerSummary> {
        use futures::stream::StreamExt;
        let container = find_container(daemon, name).await?;

        if let Some(container) = container {
            Ok(container)
        } else {
            // First pull the image if it doesn't exist. TODO: Verify this is done correctly
            let _ = daemon.create_image(Some(CreateImageOptions {
                from_image: INDEX_DOCKER_IMAGE,
                .. Default::default()
            }), None, None).next().await;

            let create_options = CreateContainerOptions {
                name,
                platform: None,
            };
            // Randomly assign a port
            let host_config: HostConfig = HostConfig {
                extra_hosts: Some(vec!["host.docker.internal:host-gateway".into()]),
                port_bindings: Some(hashmap! {
                    INDEX_TTS_DEFAULT_PORT.to_string() => None,
                }),
                device_requests: Some(vec![DeviceRequest {
                    driver: Some("".into()),
                    count: Some(-1),
                    device_ids: None,
                    capabilities: Some(vec![vec!["gpu".into()]]),
                    options: Some(HashMap::new()),
                }]),
                ..Default::default()
            };

            let empty = HashMap::<(), ()>::new();
            let mut exposed_ports = HashMap::new();
            let exposed_port = format!("{INDEX_TTS_DEFAULT_PORT}");
            exposed_ports.insert(&*exposed_port, empty);
            let config = Config {
                image: Some(INDEX_DOCKER_IMAGE),
                cmd: None,
                exposed_ports: Some(exposed_ports),
                host_config: Some(host_config),
                ..Default::default()
            };

            let _container = daemon.create_container(Some(create_options), config).await?;

            find_container(daemon, name).await?.context("Failed to create container")
        }
    }

    pub async fn find_container(daemon: &Docker, name: &str) -> eyre::Result<Option<ContainerSummary>> {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        map.insert("name".to_string(), vec![name.to_string()]);
        let opts = ListContainersOptions {
            all: true,
            limit: None,
            size: false,
            filters: map,
        };

        Ok(daemon
            .list_containers(Some(opts))
            .await?
            .into_iter()
            .next())
    }
}


#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Duration;
    use biquad::DirectForm2Transposed;
    use st_ml::emotion_classifier::BasicEmotion;
    use crate::audio::audio_data::AudioData;
    use crate::tts_backends::{BackendTtsRequest, TtsResult};
    use crate::tts_backends::indextts::api::{IndexTtsAPI, IndexTtsApiConfig, IndexTtsRequest};
    use crate::tts_backends::indextts::IndexTts;
    use crate::tts_backends::indextts::local::{LocalIndexHandle, LocalIndexTtsConfig};
    use crate::voice_manager::FsVoiceSample;

    #[tokio::test]
    #[tracing_test::traced_test]
    async fn test_index_api() -> eyre::Result<()> {
        let thing = LocalIndexTtsConfig {
            image_name: "hirtol/index-tts-llvm:latest".to_string(),
            timeout: Duration::from_secs(60),
        };
        let api = LocalIndexHandle::new(thing)?;

        let wav = std::fs::read(r"G:\TTS\small-talk-data\game_data\Pathfinder-WOTR\voices\Regill\Neutral_13.wav")?;
        let out = api.submit_tts_request(BackendTtsRequest {
            gen_text: "At the beginning of every test, the macro injects span opening code.".to_string(),
            language: "en".to_string(),
            voice_reference: vec![FsVoiceSample {
                emotion: BasicEmotion::Neutral,
                spoken_text: None,
                sample: PathBuf::from(r"G:\TTS\small-talk-data\game_data\Pathfinder-WOTR\voices\Regill\Neutral_13.wav"),
            }],
            speed: None,
        }).await?;

        match out.result {
            TtsResult::File(_) => {}
            TtsResult::Audio(out) => {
                out.write_to_wav_file("regil.wav".as_ref())?;
            }
            TtsResult::Stream => {}
        }

        Ok(())
    }

}