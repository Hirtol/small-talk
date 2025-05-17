use crate::{
    api::AppState,
    config::{Config, SharedConfig},
};
use axum::{
    error_handling::HandleErrorLayer, http::{header, HeaderValue, StatusCode},
    routing::{get_service, MethodRouter},
    BoxError,
    Router,
};
use st_system::{
    emotion::EmotionBackend, rvc_backends::{
        seedvc::local::{LocalSeedHandle, LocalSeedVcConfig},
        RvcCoordinator,
    },
    tts_backends::{
        alltalk::{
            local::{LocalAllTalkConfig, LocalAllTalkHandle},
            AllTalkConfig,
        },
        indextts::{
            api::IndexTtsApiConfig,
            local::{LocalIndexHandle, LocalIndexTtsConfig},
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
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::{compression::CompressionLayer, services::ServeFile, trace::TraceLayer};

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

        let xtts = config
            .xtts
            .as_opt()
            .map(|xtts| {
                let all_talk_cfg = LocalAllTalkConfig {
                    instance_path: xtts.local_all_talk.clone(),
                    timeout: xtts.timeout,
                    api: xtts.alltalk_cfg.clone(),
                };

                LocalAllTalkHandle::new(all_talk_cfg)
            })
            .transpose()?;

        let index = config
            .index_tts
            .as_opt()
            .map(|cfg| LocalIndexHandle::new(cfg.clone()))
            .transpose()?;

        let tts_backend = TtsCoordinator::new(xtts, index, config.dirs.whisper_model.clone());

        let mut seedvc_cfg = config.seed_vc.as_opt().map(|seed_vc| LocalSeedVcConfig {
            instance_path: seed_vc.local_path.clone(),
            timeout: seed_vc.timeout,
            api: seed_vc.config.clone(),
            high_quality: false,
        });
        let seedvc = seedvc_cfg
            .clone()
            .map(|seedvc_cfg| LocalSeedHandle::new(seedvc_cfg.clone()))
            .transpose()?;
        let seedvc_hq = seedvc_cfg
            .map(|mut seedvc_cfg| {
                seedvc_cfg.high_quality = true;
                LocalSeedHandle::new(seedvc_cfg)
            })
            .transpose()?;
        let rvc_backend = RvcCoordinator::new(seedvc, seedvc_hq);

        let emotion_backend = EmotionBackend::new(&config.dirs)?;

        let handle = Arc::new(TtsSystem::new(
            config.dirs.clone(),
            tts_backend,
            rvc_backend,
            emotion_backend,
        ));

        let result = Application {
            tcp,
            config,
            voice: handle,
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

        let app = construct_server(self.config.clone(), self.voice.clone()).await?;

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

async fn construct_server(config: SharedConfig, system: TtsSystemHandle) -> eyre::Result<Router> {
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
