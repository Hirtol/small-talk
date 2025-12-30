use crate::{
    timeout::DroppableState,
    tts_backends::{
        docker_backend::{DockerTemporaryState, DockerTtsCreateConfig},
        echo_tts::{
            api::{EchoTtsApiConfig, EchoTtsRequest},
            EchoTts,
        },
        generic_backend::TtsApi,
    },
};
use st_audio::AudioData;
const ECHO_TTS_DOCKER_PORT: u16 = 8000;
const ECHO_TTS_DOCKER_IMAGE: &str = "hirtol/echo-tts:latest";
const ECHO_TTS_CONTAINER_NAME: &str = "small-talk-echo-tts";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LocalEchoTtsConfig {
    pub image_name: String,
}

impl Default for LocalEchoTtsConfig {
    fn default() -> Self {
        Self {
            image_name: ECHO_TTS_DOCKER_IMAGE.to_string(),
        }
    }
}

pub struct LocalEchoTtsState {
    tts: EchoTts,
    docker: DockerTemporaryState,
}

impl DroppableState for LocalEchoTtsState {
    type Context = LocalEchoTtsConfig;

    async fn initialise_state(context: &Self::Context) -> eyre::Result<Self> {
        let docker_config = DockerTtsCreateConfig {
            container_name: ECHO_TTS_CONTAINER_NAME.to_string(),
            image_name: context.image_name.clone(),
            internal_port: ECHO_TTS_DOCKER_PORT,
        };
        let docker_state = DockerTemporaryState::initialise_state(&docker_config).await?;

        let api = EchoTts::initialise_state(&EchoTtsApiConfig {
            address: url::Url::parse(&docker_state.api_address)?,
        })
        .await?;

        Ok(LocalEchoTtsState {
            tts: api,
            docker: docker_state,
        })
    }

    async fn on_kill(&mut self) -> eyre::Result<()> {
        self.docker.on_kill().await
    }
}

impl TtsApi for LocalEchoTtsState {
    type Request = EchoTtsRequest;

    async fn ready(&self) -> eyre::Result<bool> {
        self.tts.ready().await
    }

    async fn tts(&self, request: Self::Request) -> eyre::Result<AudioData> {
        self.tts.tts(request).await
    }
}
