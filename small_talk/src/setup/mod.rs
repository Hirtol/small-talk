
use std::sync::Arc;
use std::time::Duration;
use axum::error_handling::HandleErrorLayer;
use axum::http::{header, HeaderValue, StatusCode};
use axum::routing::{get_service, MethodRouter};
use axum::{BoxError, Router};
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::compression::CompressionLayer;
use tower_http::services::{ServeFile};
use tower_http::trace::TraceLayer;


use crate::api::AppState;
use crate::config::{Config, SharedConfig};
use crate::system::{TtsSystem, TtsSystemHandle};
use crate::system::tts_backends::alltalk::AllTalkConfig;
use crate::system::tts_backends::alltalk::local::{LocalAllTalkConfig, LocalAllTalkHandle};
use crate::system::tts_backends::TtsBackend;

mod first_time;

pub struct Application {
    pub tcp: TcpListener,
    pub config: SharedConfig,
    pub voice: TtsSystemHandle,
}

impl Application {
    #[tracing::instrument(name = "Create application", skip(config), fields(addr = config.app.host, port = config.app.port))]
    pub async fn new(config: Config) -> eyre::Result<Self> {
        let tcp = TcpListener::bind(config.app.bind_address()).await?;

        first_time::first_time_setup(&config).await?;
        let config = Arc::new(config);


        let all_talk_cfg = LocalAllTalkConfig {
            instance_path: config.xtts.local_all_talk.clone(),
            timeout: config.xtts.timeout,
            api: config.xtts.alltalk_cfg.clone(),
        };
        let xtts = LocalAllTalkHandle::new(all_talk_cfg)?;
        let all_talk_cfg = LocalAllTalkConfig {
            instance_path: config.f5_tts.local_all_talk.clone(),
            timeout: config.f5_tts.timeout,
            api: config.f5_tts.alltalk_cfg.clone(),
        };
        let f5 = LocalAllTalkHandle::new(all_talk_cfg)?;
        let backends = TtsBackend::new(xtts, f5);
        let handle = Arc::new(TtsSystem::new(config.clone(), backends));
        
        let result = Application {
            tcp,
            config,
            voice: handle
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
    pub async fn run(self, quitter: Arc<tokio::sync::Notify>) -> eyre::Result<()> {
        tracing::info!("Setup complete, starting server...");
        
        let app = construct_server(
            self.config.clone(),
            self.voice.clone(),
        )
            .await?;

        tracing::info!("Listening on {:?}", self.tcp.local_addr()?);
        
        let server = axum::serve(self.tcp, app.into_make_service());

        let result = tokio::select! {
            _ = quitter.notified() => Ok(()),
            res = tokio::signal::ctrl_c() => {
                tracing::trace!("Received CTRL-C notification, exiting...");
                // Should notify all dependant sub-processes.
                quitter.notify_waiters();
                res.map_err(|e| eyre::eyre!(e))
            },
            res = server => res.map_err(|e| eyre::eyre!(e))
        };
        
        self.voice.shutdown().await?;

        result
    }

    pub fn port(&self) -> &TcpListener {
        &self.tcp
    }
}

async fn construct_server(
    config: SharedConfig,
    system: TtsSystemHandle,
) -> eyre::Result<Router> {
    let state = AppState {
        config: config,
        system,
    };

    let app_layers = ServiceBuilder::new()
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new().br(true).gzip(true).deflate(true));

    let app = api_router(state.clone())
        .layer(app_layers)
        .with_state(state);

    Ok(apply_security_middleware(app))
}

fn api_router(app_state: AppState) -> Router<AppState> {
    crate::api::config(app_state)
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