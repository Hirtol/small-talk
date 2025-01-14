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
use crate::system::tts_backends::{BackendTtsRequest, BackendTtsResponse, TtsResult};

#[derive(Debug, Clone)]
pub struct LocalAllTalkConfig {
    pub instance_path: PathBuf,
    pub timeout: Duration,
    pub api: AllTalkConfig,
}

#[derive(Debug, Clone)]
pub struct LocalAllTalkHandle {
    pub send: tokio::sync::mpsc::Sender<AllTalkMessage>,
}

#[derive(Debug)]
pub enum AllTalkMessage {
    /// Request the immediate start of the child process
    StartInstance,
    /// Request the immediate stop of the child process
    StopInstance,
    TtsRequest(BackendTtsRequest, tokio::sync::oneshot::Sender<BackendTtsResponse>),
}

impl LocalAllTalkHandle {
    /// Create and start a new [LocalAllTalk] actor, returning the cloneable handle to the actor in the process.
    pub fn new(config: LocalAllTalkConfig) -> eyre::Result<Self> {
        // Small amount before we exert back-pressure
        let (send, recv) = sync::mpsc::channel(10);
        let actor = LocalAllTalk {
            config,
            state: None,
            recv,
        };

        tokio::task::spawn(async move {
            if let Err(e) = actor.run().await {
                tracing::error!("LocalAllTalk stopped with error: {e}");
            }
        });

        Ok(Self { send })
    }
    
    /// Send a TTS request to the local AllTalk instance
    pub async fn submit_tts_request(&self, request: BackendTtsRequest) -> eyre::Result<BackendTtsResponse> {
        let (send, recv) = tokio::sync::oneshot::channel();
        self.send.send(AllTalkMessage::TtsRequest(request, send)).await?;

        Ok(recv.await?)
    }
}

struct LocalAllTalk {
    config: LocalAllTalkConfig,
    state: Option<TemporaryState>,
    recv: sync::mpsc::Receiver<AllTalkMessage>,
}

struct TemporaryState {
    tts: AllTalkTTS,
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
                    tracing::trace!("Stopping LocalAllTalk actor as channel was closed");
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
            AllTalkMessage::TtsRequest(request, response) => {
                let voice_path = self.voices_path();
                let state = self.verify_state().await?;
                let output_file = crate::system::utils::random_file_name(24, None);
                // We have to move (hardlink) the sample to the AllTalk voices dir
                let sample_name = crate::system::utils::random_file_name(24, None);
                let input_file = request.voice_reference[0].link_to_name(voice_path, &sample_name)?;
                
                let alltalk_req = super::api::TtsRequest {
                    text_input: request.gen_text,
                    text_filtering: None,
                    character_voice_gen: input_file.sample.file_name()
                        .context("Could not get filename")?
                        .to_string_lossy()
                        .into_owned(),
                    rvccharacter_voice_gen: None,
                    rvccharacter_pitch: None,
                    narrator_enabled: None,
                    narrator_voice_gen: None,
                    rvcnarrator_voice_gen: None,
                    rvcnarrator_pitch: None,
                    text_not_inside: None,
                    language: request.language,
                    output_file_name: output_file,
                    output_file_timestamp: None,
                    autoplay: None,
                    autoplay_volume: None,
                    speed: request.speed,
                    pitch: None,
                    temperature: None,
                    repetition_penalty: None,
                };
                
                let now = std::time::Instant::now();
                let tts_response = state.tts.api.tts_request(alltalk_req).await?;
                let took = now.elapsed();
                let gen_path = PathBuf::from(tts_response.output_file_path);
                
                // Clear up the sample which we linked above.
                tokio::fs::remove_file(input_file.sample).await?;
                if let Some(text) = input_file.spoken_text {
                    tokio::fs::remove_file(text).await?;
                }
                
                let _ = response.send(BackendTtsResponse {
                    gen_time: took,
                    result: TtsResult::File(gen_path),
                });
                
                tracing::trace!(?took, "Finished handling of TTS request");
            }
        }
        Ok(())
    }
    
    fn voices_path(&self) -> PathBuf {
        self.config.instance_path.join("voices")
    }

    /// Ensure a running AllTalk instance exists.
    async fn verify_state(&mut self) -> eyre::Result<&TemporaryState> {
        // Borrow checker prevents us from doing this nicely...
        if self.state.is_none() {
            self.initialise_state().await
        } else {
            self.state.as_ref().context("Impossible")
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
    async fn initialise_state(&mut self) -> eyre::Result<&TemporaryState> {
        let child = Self::start_alltalk(&self.config.instance_path).await?;
        let api = AllTalkTTS::new(self.config.api.clone()).await?;

        Ok(self.state.insert(TemporaryState {
            tts: api,
            process: child,
            last_access: std::time::Instant::now(),
        }))
    }

    /// Start the AllTalk instance located in `path`.
    ///
    /// Note that this spawns a sub-process.
    #[tracing::instrument]
    async fn start_alltalk(path: &Path) -> eyre::Result<Box<dyn TokioChildWrapper>> {
        tracing::debug!("Attempting to start AllTalk process");
        let alltalk_env = path.join("alltalk_environment");
        let conda_env = alltalk_env.join("conda");
        let env_env = alltalk_env.join("env");
        let python_exe = env_env.join("python.exe");

        let mut cmd = Command::new(python_exe);
        cmd.envs(std::env::vars());
        cmd.env("CONDA_ROOT_PREFIX", conda_env);
        cmd.env("INSTALL_ENV_DIR", env_env);
        cmd.args(["script.py"])
            .kill_on_drop(true)
            .current_dir(path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

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
    use crate::system::tts_backends::alltalk::AllTalkConfig;
    use crate::system::tts_backends::alltalk::local::{AllTalkMessage, LocalAllTalkConfig, LocalAllTalkHandle};

    #[tokio::test]
    #[tracing_test::traced_test]
    pub async fn basic_test() {
        let cfg = LocalAllTalkConfig {
            instance_path: r"G:\TTS\alltalk_tts\".into(),
            timeout: Duration::from_secs(2),
            api: AllTalkConfig::new("http://localhost:7581".try_into().unwrap()),
        };

        let handle = LocalAllTalkHandle::new(cfg).unwrap();
        
        handle.send.send(AllTalkMessage::StartInstance).await.unwrap();
        
        tokio::time::sleep(Duration::from_secs(5)).await;

        handle.send.send(AllTalkMessage::StopInstance).await.unwrap();
        tokio::time::sleep(Duration::from_secs(5)).await;
        drop(handle)
    }
}
