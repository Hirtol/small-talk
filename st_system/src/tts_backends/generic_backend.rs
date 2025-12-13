use std::time::Duration;
use serde::{Deserialize, Serialize};
use crate::audio::AudioData;
use crate::timeout::{DroppableState, GcCell};
use crate::tts_backends::{BackendTtsRequest, BackendTtsResponse};

#[derive(Debug)]
pub enum TtsBackendMessage {
    /// Request the immediate start of the child process
    StartInstance,
    /// Request the immediate stop of the child process
    StopInstance,
    TtsRequest(BackendTtsRequest, tokio::sync::oneshot::Sender<BackendTtsResponse>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActiveTtsStateConfig<LocalConf, RemoteConf = RemoteTtsConfig> {
    Local {
        local_config: LocalConf
    },
    Remote {
        remote_config: RemoteConf,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteTtsConfig {
    pub api_address: String,
}

#[derive(Debug)]
pub enum ActiveTtsState<Local, Remote> {
    Local(Local),
    Remote(Remote),
}

impl<LocalConf, RemoteConf, Local: DroppableState<Context=LocalConf>, Remote: DroppableState<Context=RemoteConf>> DroppableState for ActiveTtsState<Local, Remote> {
    type Context = ActiveTtsStateConfig<LocalConf, RemoteConf>;

    async fn initialise_state(context: &Self::Context) -> eyre::Result<Self> {
        match context {
            ActiveTtsStateConfig::Local { local_config } => Local::initialise_state(local_config).await.map(ActiveTtsState::Local),
            ActiveTtsStateConfig::Remote { remote_config } => Remote::initialise_state(remote_config).await.map(ActiveTtsState::Remote),
        }
    }

    async fn on_kill(&mut self) -> eyre::Result<()> {
        match self {
            ActiveTtsState::Local(local) => local.on_kill().await,
            ActiveTtsState::Remote(remote) => remote.on_kill().await
        }
    }
}

impl<Req, Local: TtsApi<Request=Req>, Remote: TtsApi<Request=Req>> TtsApi for ActiveTtsState<Local, Remote> {
    type Request = Req;

    async fn ready(&self) -> eyre::Result<bool> {
        match self {
            ActiveTtsState::Local(local) => local.ready().await,
            ActiveTtsState::Remote(remote) => remote.ready().await
        }
    }

    async fn tts(&self, request: Self::Request) -> eyre::Result<AudioData> {
        match self {
            ActiveTtsState::Local(local) => local.tts(request).await,
            ActiveTtsState::Remote(remote) => remote.tts(request).await
        }
    }
}

pub trait TtsApi {
    type Request;

    async fn ready(&self) -> eyre::Result<bool>;

    /// Send a request for a generation to the given API.
    async fn tts(&self, request: Self::Request) -> eyre::Result<AudioData>;
}

pub struct ReadyTtsApi<T>(T);

impl<T: TtsApi> ReadyTtsApi<T> {
    pub async fn new(api: T) -> eyre::Result<Self> {
        // Wait for it to be ready
        tokio::time::timeout(Duration::from_secs(120), async {
            while !api.ready().await? {
                tracing::trace!("TTS API not ready yet, waiting");
                tokio::time::sleep(Duration::from_secs(1)).await
            }

            Ok::<_, eyre::Report>(())
        }).await??;
        tracing::trace!("TTS Api ready!");

        Ok(Self(api))
    }
}

impl<T: TtsApi> TtsApi for ReadyTtsApi<T> {
    type Request = T::Request;

    async fn ready(&self) -> eyre::Result<bool> {
        self.0.ready().await
    }

    async fn tts(&self, request: Self::Request) -> eyre::Result<AudioData> {
        self.0.tts(request).await
    }
}

impl<T: DroppableState> DroppableState for ReadyTtsApi<T> {
    type Context = T::Context;

    async fn initialise_state(context: &Self::Context) -> eyre::Result<Self> {
        T::initialise_state(context).await.map(Self)
    }

    async fn on_kill(&mut self) -> eyre::Result<()> {
        self.0.on_kill().await
    }
}