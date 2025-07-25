use eyre::ContextCompat;
use std::{
    path::{Path, PathBuf},
    process::Stdio,
};
use std::time::Duration;
use process_wrap::tokio::TokioChildWrapper;
use tokio::{
    process::{Child, Command},
};
use tokio::time::error::Elapsed;
use crate::error::RvcError;
use crate::rvc_backends::{BackendRvcRequest, BackendRvcResponse, RvcResult};
use crate::rvc_backends::seedvc::api::SeedVcApiConfig;
use crate::rvc_backends::seedvc::SeedRvc;
use crate::timeout::{DroppableState, GcCell};
use crate::tts_backends::{BackendTtsRequest, BackendTtsResponse, TtsResult};

#[derive(Debug, Clone)]
pub struct LocalSeedVcConfig {
    pub instance_path: PathBuf,
    pub timeout: Duration,
    pub api: SeedVcApiConfig,
    pub high_quality: bool,
}

#[derive(Debug, Clone)]
pub struct LocalSeedHandle {
    pub send: tokio::sync::mpsc::UnboundedSender<SeedMessage>,
}

#[derive(Debug)]
pub enum SeedMessage {
    /// Request the immediate start of the child process
    StartInstance,
    /// Request the immediate stop of the child process
    StopInstance,
    RvcRequest(BackendRvcRequest, tokio::sync::oneshot::Sender<BackendRvcResponse>),
}

impl LocalSeedHandle {
    /// Create and start a new [LocalSeedVc] actor, returning the cloneable handle to the actor in the process.
    pub fn new(config: LocalSeedVcConfig) -> eyre::Result<Self> {
        // Small amount before we exert back-pressure
        let (send, recv) = tokio::sync::mpsc::unbounded_channel();
        let actor = LocalSeedVc {
            state: GcCell::new(config.timeout),
            config,
            recv,
        };

        tokio::task::spawn(async move {
            if let Err(e) = actor.run().await {
                tracing::error!("LocalSeedVc stopped with error: {e}");
            }
        });

        Ok(Self { send })
    }

    pub async fn start_instance(&self) -> eyre::Result<()> {
        Ok(self.send.send(SeedMessage::StartInstance)?)
    }

    pub async fn stop_instance(&self) -> eyre::Result<()> {
        Ok(self.send.send(SeedMessage::StopInstance)?)
    }

    /// Send a RVC request to the SeedVc instance.
    pub async fn rvc_request(&self, request: BackendRvcRequest) -> Result<BackendRvcResponse, RvcError> {
        let (send, recv) = tokio::sync::oneshot::channel();
        self.send.send(SeedMessage::RvcRequest(request, send)).map_err(|_| RvcError::Timeout)?;

        recv.await.map_err(|_| RvcError::Timeout)
    }
}

struct LocalSeedVc {
    config: LocalSeedVcConfig,
    state: GcCell<TemporaryState>,
    recv: tokio::sync::mpsc::UnboundedReceiver<SeedMessage>,
}

struct TemporaryState {
    rvc: SeedRvc,
    process: Box<dyn TokioChildWrapper>,
    last_access: std::time::Instant,
}

impl LocalSeedVc {

    /// Start the actor, this future should be `tokio::spawn`ed.
    ///
    /// It will automatically drop the internal state if it hasn't been accessed in a while to preserve memory.
    #[tracing::instrument(skip(self))]
    pub async fn run(mut self) -> Result<(), RvcError> {
        loop {
            tokio::select! {
                msg = self.recv.recv() => {
                    // Have to pattern match here, as we want this `select!` to stop if the channel is closed, and not hang
                    // on our timeout
                    match msg {
                        Some(msg) => match self.handle_message(msg).await {
                            Ok(_) => {}
                            Err(RvcError::Timeout) => {
                                tracing::warn!("SeedVc timed out. Assuming failed state, restarting");
                                // Something went wrong in our underlying state
                                self.state.kill_state().await?;
                            }
                            e => return e
                        },
                        None => {
                            self.state.kill_state().await?;
                            tracing::trace!("Stopping LocalSeedVc actor as channel was closed");
                            break
                        },
                    }
                },
                _ = self.state.timeout_future() => {
                    tracing::debug!("Timeout expired, dropping local SeedVc state");
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
    async fn handle_message(&mut self, message: SeedMessage) -> Result<(), RvcError> {
        match message {
            SeedMessage::StartInstance => {
                self.state.get_state(&self.config).await?;
            }
            SeedMessage::StopInstance => {
                self.state.kill_state().await?;
            }
            SeedMessage::RvcRequest(request, response) => {
                let state = self.state.get_state(&self.config).await?;

                let now = std::time::Instant::now();
                let rvc_response = tokio::time::timeout(Duration::from_secs(40), state.rvc.api.rvc(request)).await??;
                let took = now.elapsed();

                let _ = response.send(BackendRvcResponse {
                    gen_time: took,
                    result: RvcResult::Wav(rvc_response),
                });

                tracing::trace!(?took, "Finished handling of RVC request");
            }
        }
        Ok(())
    }
}

impl DroppableState for TemporaryState {
    type Context = LocalSeedVcConfig;

    async fn initialise_state(context: &Self::Context) -> eyre::Result<Self> {
        #[tracing::instrument]
        async fn start_seedvc(path: &Path, high_quality: bool) -> eyre::Result<Box<dyn TokioChildWrapper>> {
            tracing::debug!("Attempting to start SeedVc process");
            let seed_env = path.join(".venv").join("Scripts");
            let python_exe = seed_env.join("python.exe");
            let log_file = std::fs::File::create(path.join("small_talk.log"))?;

            let mut cmd = Command::new(python_exe);
            cmd.envs(std::env::vars());
            cmd.env("PATH", seed_env);
            cmd.args(["seed_vc_api.py", "--low-vram", "False"])
                .kill_on_drop(true)
                .current_dir(path)
                .stdout(log_file)
                .stderr(Stdio::piped());

            if high_quality {
                cmd.args(["--diffusion-steps", "20", "--f0-condition", "True", "--inference-cfg-rate", "0.7"]);
            } else {
                // A few extra steps for the lower-quality model to make up for the difference
                cmd.args(["--diffusion-steps", "25", "--inference-cfg-rate", "0.7"]);
            }

            let mut wrapped = process_wrap::tokio::TokioCommandWrap::from(cmd);
            wrapped.wrap(process_wrap::tokio::KillOnDrop);

            #[cfg(unix)]
            {
                wrapped.wrap(process_wrap::tokio::ProcessGroup::leader());
            }
            #[cfg(windows)]
            {
                wrapped.wrap(process_wrap::tokio::JobObject);
            }

            match wrapped.spawn() {
                Ok(child) => Ok(child),
                Err(e) => {
                    eyre::bail!("Failed to execute batch file: {}", e);
                }
            }
        }

        let child = start_seedvc(&context.instance_path, context.high_quality).await?;
        let api = SeedRvc::new(context.api.clone()).await?;

        Ok(TemporaryState {
            rvc: api,
            process: child,
            last_access: std::time::Instant::now(),
        })
    }

    async fn on_kill(&mut self) -> eyre::Result<()> {
        let kill_future = Box::into_pin(self.process.kill());
        kill_future.await?;
        Ok(())
    }
}
