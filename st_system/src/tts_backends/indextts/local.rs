use crate::{
    audio::AudioData,
    timeout::DroppableState,
    tts_backends::{
        docker_backend::{docker_utils::DockerTtsCreateConfig, DockerTemporaryState},
        generic_backend::{ReadyTtsApi, TtsApi},
        indextts::{
            api::{IndexTtsAPI, IndexTtsApiConfig, IndexTtsRequest},
            IndexTts,
        },
    },
};

const INDEX_TTS_DEFAULT_PORT: u16 = 11996;
const INDEX_DOCKER_IMAGE: &str = "hirtol/index-tts-llvm:latest";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LocalIndexTtsConfig {
    pub image_name: String,
}

impl Default for LocalIndexTtsConfig {
    fn default() -> Self {
        Self {
            image_name: INDEX_DOCKER_IMAGE.to_string(),
        }
    }
}

pub struct LocalIndexState {
    tts: IndexTts,
    docker: DockerTemporaryState,
}

impl TtsApi for LocalIndexState {
    type Request = IndexTtsRequest;

    async fn ready(&self) -> eyre::Result<bool> {
        self.tts.ready().await
    }

    async fn tts(&self, request: Self::Request) -> eyre::Result<AudioData> {
        self.tts.tts(request).await
    }
}

impl DroppableState for LocalIndexState {
    type Context = LocalIndexTtsConfig;

    async fn initialise_state(context: &Self::Context) -> eyre::Result<Self> {
        let docker_config = DockerTtsCreateConfig {
            container_name: "small-talk-index-tts".to_string(),
            image_name: context.image_name.clone(),
            internal_port: INDEX_TTS_DEFAULT_PORT,
        };
        let docker_state = DockerTemporaryState::initialise_state(&docker_config).await?;

        let api = ReadyTtsApi::new(IndexTtsAPI::new(IndexTtsApiConfig {
            address: url::Url::parse(&docker_state.api_address)?,
        })?)
        .await?;

        Ok(LocalIndexState {
            tts: api,
            docker: docker_state,
        })
    }

    async fn on_kill(&mut self) -> eyre::Result<()> {
        self.docker.on_kill().await
    }
}
