use crate::system::tts_backends::alltalk::{api::AllTalkApi, AllTalkConfig, AllTalkTTS};
use eyre::ContextCompat;
use std::{
    path::{Path, PathBuf},
    process::Stdio,
};
use std::time::Duration;
use process_wrap::tokio::TokioChildWrapper;
use tokio::{
    process::{Child, Command},
    sync,
};
use crate::system::rvc_backends::{BackendRvcRequest, BackendRvcResponse, RvcResult};
use crate::system::rvc_backends::seedvc::api::SeedVcApiConfig;
use crate::system::rvc_backends::seedvc::SeedRvc;
use crate::system::tts_backends::{BackendTtsRequest, BackendTtsResponse, TtsResult};

#[derive(Debug, Clone)]
pub struct LocalSeedVcConfig {
    pub instance_path: PathBuf,
    pub timeout: Duration,
    pub api: SeedVcApiConfig,
    pub high_quality: bool,
}

#[derive(Debug, Clone)]
pub struct LocalSeedHandle {
    pub send: tokio::sync::mpsc::Sender<SeedMessage>,
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
        let (send, recv) = sync::mpsc::channel(10);
        let actor = LocalSeedVc {
            config,
            state: None,
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
        Ok(self.send.send(SeedMessage::StartInstance).await?)
    }

    pub async fn stop_instance(&self) -> eyre::Result<()> {
        Ok(self.send.send(SeedMessage::StopInstance).await?)
    }

    /// Send a RVC request to the SeedVc instance.
    pub async fn rvc_request(&self, request: BackendRvcRequest) -> eyre::Result<BackendRvcResponse> {
        let (send, recv) = tokio::sync::oneshot::channel();
        self.send.send(SeedMessage::RvcRequest(request, send)).await?;

        Ok(recv.await?)
    }
}

struct LocalSeedVc {
    config: LocalSeedVcConfig,
    state: Option<TemporaryState>,
    recv: sync::mpsc::Receiver<SeedMessage>,
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
    pub async fn run(mut self) -> eyre::Result<()> {
        loop {
            let instance_last = self.state.as_ref().map(|v| v.last_access);
            // If we have state we need to try to drop it after a while
            if let Some(last_access) = instance_last {
                let timeout = last_access + self.config.timeout;

                tokio::select! {
                    Some(msg) = self.recv.recv() => {
                        self.handle_message(msg).await?;
                    },
                    _ = tokio::time::sleep_until(timeout.into()) => {
                        if self.state.as_ref().map(|v| v.last_access < timeout).unwrap_or_default() {
                            tracing::debug!("Timeout expired, dropping local SeedVc state");
                            // Drop the state, killing the sub-process
                            // Safe to do as we know that it won't be generating for us since we have exclusive access.
                            self.kill_state().await?;
                        }
                    }
                    else => break,
                }
            } else {
                let Some(msg) = self.recv.recv().await else {
                    tracing::trace!("Stopping LocalSeedVc actor as channel was closed");
                    break;
                };
                self.handle_message(msg).await?;
            }
        }

        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn handle_message(&mut self, message: SeedMessage) -> eyre::Result<()> {
        match message {
            SeedMessage::StartInstance => {
                self.verify_state().await?;
            }
            SeedMessage::StopInstance => {
                self.kill_state().await?;
            }
            SeedMessage::RvcRequest(request, response) => {
                let state = self.verify_state().await?;

                let now = std::time::Instant::now();
                let rvc_response = state.rvc.api.rvc(request).await?;
                let took = now.elapsed();

                let _ = response.send(BackendRvcResponse {
                    gen_time: took,
                    result: RvcResult::Wav(rvc_response),
                });

                tracing::trace!(?took, "Finished handling of TTS request");
            }
        }
        Ok(())
    }

    /// Ensure a running SeedVc instance exists.
    async fn verify_state(&mut self) -> eyre::Result<&TemporaryState> {
        // Borrow checker prevents us from doing this nicely...
        if self.state.is_none() {
            self.initialise_state().await
        } else {
            self.state.as_ref().context("Impossible")
        }
    }

    /// Kill the current SeedVc instance.
    async fn kill_state(&mut self) -> eyre::Result<()> {
        let Some(mut val) = self.state.take() else {
            return Ok(());
        };
        let kill_future = Box::into_pin(val.process.kill());
        kill_future.await?;
        Ok(())
    }

    /// Force create and replace the current state by creating a new SeedVc instance.
    #[tracing::instrument(skip(self))]
    async fn initialise_state(&mut self) -> eyre::Result<&TemporaryState> {
        let child = Self::start_seedvc(&self.config.instance_path, self.config.high_quality).await?;
        let api = SeedRvc::new(self.config.api.clone()).await?;

        Ok(self.state.insert(TemporaryState {
            rvc: api,
            process: child,
            last_access: std::time::Instant::now(),
        }))
    }

    /// Start the AllTalk instance located in `path`.
    ///
    /// Note that this spawns a sub-process.
    #[tracing::instrument]
    async fn start_seedvc(path: &Path, high_quality: bool) -> eyre::Result<Box<dyn TokioChildWrapper>> {
        tracing::debug!("Attempting to start SeedVc process");
        let seed_env = path.join(".venv").join("Scripts");
        let python_exe = seed_env.join("python.exe");
        let log_file = std::fs::File::create(path.join("small_talk.log"))?;

        let mut cmd = Command::new(python_exe);
        cmd.envs(std::env::vars());
        cmd.env("PATH", seed_env);
        cmd.args(["seed_vc_api.py"])
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
}

#[cfg(test)]
mod tests {
    use std::process::Stdio;
    use std::time::Duration;
    use process_wrap::tokio::TokioChildWrapper;
    use tokio::io::AsyncWriteExt;
    use tokio::process::Command;
    use crate::system::rvc_backends::seedvc::api::SeedVcApiConfig;
    use crate::system::rvc_backends::seedvc::local::{LocalSeedHandle, LocalSeedVcConfig, SeedMessage};

    #[tokio::test]
    #[tracing_test::traced_test]
    pub async fn basic_test() {
        let cfg = LocalSeedVcConfig {
            instance_path: r"G:\TTS\seed-vc\".into(),
            timeout: Duration::from_secs(2),
            api: SeedVcApiConfig {
                address: "http://localhost:9999".try_into().unwrap()
            }
        };

        let handle = LocalSeedHandle::new(cfg).unwrap();

        handle.send.send(SeedMessage::StartInstance).await.unwrap();

        tokio::time::sleep(Duration::from_secs(5)).await;

        handle.send.send(SeedMessage::StopInstance).await.unwrap();
        tokio::time::sleep(Duration::from_secs(5)).await;
        drop(handle)
    }
}
