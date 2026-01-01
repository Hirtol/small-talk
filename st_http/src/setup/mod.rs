use crate::{
    api::AppState,
    config::{SmallTalkHttpConfig},
};
use axum::{
    error_handling::HandleErrorLayer, http::{StatusCode},
    BoxError,
    Router,
};
use st_system::{
    emotion::EmotionBackend, rvc_backends::{
        seedvc::local::{LocalSeedHandle, LocalSeedVcConfig},
        RvcCoordinator,
    },
    tts_backends::{
        indextts::{
            api::IndexTtsApiConfig,
            LocalIndexTtsConfig,
        },
        TtsCoordinator,
    },
    TtsSystem,
    TtsSystemHandle,
};
use std::{
    sync::{Arc, LazyLock},
    time::Duration,
};
use std::path::PathBuf;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tower::ServiceBuilder;
use tower_http::{compression::CompressionLayer, services::ServeFile, trace::TraceLayer};
use st_application::SmallTalkApplication;
use st_ml::stt::WhisperTranscribe;
use st_system::emotion::EmotionCoordinator;
use st_system::tts_backends::echo_tts::EchoTtsHandle;
use st_system::tts_backends::indextts::IndexTtsHandle;

mod first_time;

pub struct Application {
    pub tcp: TcpListener,
    pub config: Arc<SmallTalkHttpConfig>,
    pub small_talk: SmallTalkApplication,
}

impl Application {
    #[tracing::instrument(name = "Create application", skip(config), fields(addr = config.app.host, port = config.app.port))]
    pub async fn new(config: SmallTalkHttpConfig) -> eyre::Result<Self> {
        let tcp = TcpListener::bind(config.app.bind_address()).await?;

        first_time::first_time_setup(&config).await?;
        let small_talk = SmallTalkApplication::new(&config.small_talk).await?;
        let config = Arc::new(config);

        let result = Application {
            tcp,
            config,
            small_talk,
        };

        Ok(result)
    }

    /// Start running the Axum server, consuming `Application`.
    /// The future completes when the Tokio-Runtime has been shut down (due to f.e a SIGINT).
    ///
    /// # Arguments
    ///
    /// * `quitter` - A way to inform the spawned runtime to shut down. Especially useful for tests
    /// where we won't provide a signal for shutdown.
    pub async fn run(self) -> eyre::Result<()> {
        tracing::info!("Setup complete, starting server...");

        let app = construct_server(self.config.clone(), self.small_talk.tts_system.clone()).await?;

        tracing::info!("Listening on {:?}", self.tcp.local_addr()?);

        let server = axum::serve(self.tcp, app.into_make_service());

        let result = tokio::select! {
            _ = self.small_talk.root_cancel.cancelled() => Ok(()),
            res = tokio::signal::ctrl_c() => {
                tracing::trace!("Received CTRL-C notification, exiting...");

                res.map_err(|e| eyre::eyre!(e))
            },
            res = server => res.map_err(|e| eyre::eyre!(e))
        };

        self.small_talk.shutdown().await?;

        result
    }

    pub fn port(&self) -> &TcpListener {
        &self.tcp
    }
}

async fn construct_server(config: Arc<SmallTalkHttpConfig>, system: TtsSystemHandle) -> eyre::Result<Router> {
    let state = AppState { config, system };

    let app_layers = ServiceBuilder::new()
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new().br(true).gzip(true).deflate(true));

    let app = api_router().layer(app_layers).with_state(state);

    Ok(apply_security_middleware(app))
}

fn api_router() -> Router<AppState> {
    crate::api::config()
}

fn apply_security_middleware(router: Router) -> Router {
    let security = ServiceBuilder::new()
        .layer(HandleErrorLayer::new(generic_error_handler))
        .load_shed()
        .concurrency_limit(512)
        .layer(tower_http::timeout::TimeoutLayer::new(Duration::from_secs(60)));

    router.layer(security)
}

async fn generic_error_handler(_error: BoxError) -> impl axum::response::IntoResponse {
    tracing::trace!(error=?_error, "Error occurred in normal response handler");
    (StatusCode::INTERNAL_SERVER_ERROR, "Internal Error")
}
