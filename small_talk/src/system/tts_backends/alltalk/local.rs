use crate::system::tts_backends::alltalk::{api::AllTalkApi, AllTalkConfig};
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

#[derive(Debug, Clone)]
pub struct LocalAllTalkConfig {
    pub timeout: Duration,
    pub api: AllTalkConfig,
}

#[derive(Debug, Clone)]
pub struct LocalAllTalkHandle {
    pub send: tokio::sync::mpsc::Sender<AllTalkMessage>,
}

#[derive(Debug, Clone)]
pub enum AllTalkMessage {
    /// Request the immediate start of the child process
    StartInstance,
    /// Request the immediate stop of the child process
    StopInstance,
}

impl LocalAllTalkHandle {
    /// Create and start a new [LocalAllTalk] actor, returning the cloneable handle to the actor in the process.
    pub fn new(instance_path: impl Into<PathBuf>, config: LocalAllTalkConfig) -> eyre::Result<Self> {
        // Small amount before we exert back-pressure
        let (send, recv) = sync::mpsc::channel(10);
        let actor = LocalAllTalk {
            instance_path: instance_path.into(),
            config,
            state: None,
            recv,
        };

        tokio::task::spawn(async move {
            if let Err(e) = actor.run().await {
                println!("LocalAllTalk stopped with error: {e}");
            }
        });

        Ok(Self { send })
    }
}

pub struct LocalAllTalk {
    instance_path: PathBuf,
    config: LocalAllTalkConfig,
    state: Option<TemporaryState>,
    recv: sync::mpsc::Receiver<AllTalkMessage>,
}

struct TemporaryState {
    api: AllTalkApi,
    process: Box<dyn TokioChildWrapper>,
    last_access: std::time::Instant,
}

impl LocalAllTalk {

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
                            tracing::debug!("Timeout expired, dropping local AllTalk state");
                            // Drop the state, killing the sub-process
                            // Safe to do as we know that it won't be generating for us since we have exclusive access.
                            self.kill_state().await?;
                        }
                    }
                    else => break,
                }
            } else {
                let Some(msg) = self.recv.recv().await else {
                    tracing::trace!("Stopping local actor as channel was closed");
                    break;
                };
                self.handle_message(msg).await?;
            }
        }

        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn handle_message(&mut self, message: AllTalkMessage) -> eyre::Result<()> {
        match message {
            AllTalkMessage::StartInstance => {
                self.verify_state().await?;
            }
            AllTalkMessage::StopInstance => {
                self.kill_state().await?;
            }
        }
        Ok(())
    }

    /// Ensure a running AllTalk instance exists.
    async fn verify_state(&mut self) -> eyre::Result<()> {
        if self.state.is_none() {
            self.initialise_state().await
        } else {
            Ok(())
        }
    }
    
    /// Kill the current AllTalk instance.
    async fn kill_state(&mut self) -> eyre::Result<()> {
        let Some(mut val) = self.state.take() else {
            return Ok(());
        };
        let kill_future = Box::into_pin(val.process.kill());
        kill_future.await?;
        Ok(())
    }

    /// Force create and replace the current state by creating a new AllTalk instance.
    #[tracing::instrument(skip(self))]
    async fn initialise_state(&mut self) -> eyre::Result<()> {
        let child = Self::start_alltalk(&self.instance_path).await?;

        self.state = Some(TemporaryState {
            api: AllTalkApi::new(self.config.api.clone())?,
            process: child,
            last_access: std::time::Instant::now(),
        });

        Ok(())
    }

    /// Start the AllTalk instance located in `path`.
    ///
    /// Note that this spawns a sub-process.
    #[tracing::instrument]
    async fn start_alltalk(path: &Path) -> eyre::Result<Box<dyn TokioChildWrapper>> {
        tracing::debug!("Attempting to start AllTalk process");
        let start_bat = path.join("start_alltalk.bat");

        let mut cmd = Command::new("cmd");
        cmd.args(["/C", &start_bat.to_string_lossy()])
            .kill_on_drop(true)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
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
    use std::time::Duration;
    use crate::system::tts_backends::alltalk::AllTalkConfig;
    use crate::system::tts_backends::alltalk::local::{AllTalkMessage, LocalAllTalkConfig, LocalAllTalkHandle};

    #[tokio::test]
    pub async fn basic_test() {
        let cfg = LocalAllTalkConfig {
            timeout: Duration::from_secs(2),
            api: AllTalkConfig::new("localhost:7581".try_into().unwrap()),
        };
        let handle = LocalAllTalkHandle::new(r"G:\TTS\alltalk_tts\", ).unwrap();
        
        handle.send.send(AllTalkMessage::StartInstance).await.unwrap();
        
        tokio::time::sleep(Duration::from_secs(5)).await;

        handle.send.send(AllTalkMessage::StopInstance).await.unwrap();
        tokio::time::sleep(Duration::from_secs(5)).await;
        drop(handle)
    }
}
