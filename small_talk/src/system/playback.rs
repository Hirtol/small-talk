use crate::system::{session::GameSessionMessage, voice_manager::VoiceManager, TtsModel, TtsResponse, VoiceLine};
use rodio::{Decoder, OutputStream, Sink};
use std::{collections::VecDeque, fs::File, io::BufReader, sync::Arc, time::Duration};
use tokio::sync::broadcast;

#[derive(Clone)]
pub struct PlaybackEngineHandle {
    send: tokio::sync::mpsc::Sender<PlaybackMessage>,
}

impl PlaybackEngineHandle {
    /// Start a new playback engine
    pub async fn new(session: tokio::sync::mpsc::Sender<GameSessionMessage>) -> eyre::Result<PlaybackEngineHandle> {
        let (send, recv) = tokio::sync::mpsc::channel(10);
        let (stream, stream_handle) = OutputStream::try_default().expect("No output device for audio available");

        let engine = PlaybackEngine {
            _output_device: SendCpalStream(stream),
            audio_sink: Sink::try_new(&stream_handle).expect("Failed to initialise audio sink."),
            session_handle: session,
            recv,
            current_request: None,
            current_volume: 1.0,
            current_queue: Default::default(),
        };
        let rt = tokio::runtime::Handle::current();
        // We do blocking IO in the actor, so spawn it on the thread pool.
        tokio::task::spawn_blocking(move || {
            rt.block_on(async move {
                if let Err(e) = engine.run().await {
                    tracing::error!("PlaybackEngine stopped with error: {e}");
                }
            })
        });

        Ok(Self { send })
    }

    /// Start the playback of the given line(s).
    ///
    /// If the TTS request hasn't been completed (or requested) the playback engine will wait until it is available.
    /// The playback can be cancelled using [Self::stop], or by simply [Self::start]ing another line.
    /// If the engine was waiting for a different line to be completed then it will simply discard that initial request and wait for the new line instead.
    ///
    /// This method returns immediately, it does not wait for playback to be completed.
    ///
    /// This method treats the whole [Vec] as one [VoiceLine] for the sakes of playback, all lines will be played, or replaced if a new [Self::start] call is issued.
    pub async fn start(&self, lines: VecDeque<PlaybackVoiceLine>) -> eyre::Result<()> {
        Ok(self.send.send(PlaybackMessage::Start(lines)).await?)
    }

    /// Stop the current [VoiceLine] from playing.
    ///
    /// If the engine was waiting for a different line to be completed then it will simply discard that initial request and wait for the new line instead.
    ///
    /// This method returns immediately.
    pub async fn stop(&self) -> eyre::Result<()> {
        Ok(self.send.send(PlaybackMessage::Stop).await?)
    }
}

#[derive(Debug, Clone)]
pub struct PlaybackVoiceLine {
    pub line: VoiceLine,
    pub volume: Option<f32>,
}

#[derive(Debug, Clone)]
pub enum PlaybackMessage {
    Stop,
    Start(VecDeque<PlaybackVoiceLine>),
}

pub struct PlaybackEngine {
    _output_device: SendCpalStream,
    session_handle: tokio::sync::mpsc::Sender<GameSessionMessage>,
    
    recv: tokio::sync::mpsc::Receiver<PlaybackMessage>,
    
    audio_sink: Sink,
    current_volume: f32,
    current_queue: VecDeque<PlaybackVoiceLine>,
    current_request: Option<tokio::sync::oneshot::Receiver<Arc<TtsResponse>>>,
}

impl PlaybackEngine {
    #[tracing::instrument(skip(self))]
    pub async fn run(mut self) -> eyre::Result<()> {
        // There is no callback/future we can use to detect a finished line, so we'll just have to poll it.
        // TODO: Potentially switch to the `kira` crate from the Bevy ecosystem for Reverb/callbacks
        let mut check_interval = tokio::time::interval(Duration::from_millis(100));
        loop {
            let one_shot_future: futures::future::OptionFuture<_> = self.current_request.as_mut().into();
            tokio::select! {
                msg = self.recv.recv() => {
                    let Some(msg) = msg else {
                        break;
                    };

                    self.handle_message(msg).await?;
                },
                Some(Ok(tts)) = one_shot_future => {
                    self.handle_tts_sample(tts).await?;
                },
                _ = check_interval.tick() => {
                    self.handle_queue_tick().await?;
                }
                else => break
            }
        }
        tracing::trace!("Stopping PlaybackEngine as channel was closed");

        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn handle_message(&mut self, message: PlaybackMessage) -> eyre::Result<()> {
        match message {
            PlaybackMessage::Stop => {
                self.current_request = None;
                self.current_queue.clear();
                self.audio_sink.clear();
            }
            PlaybackMessage::Start(lines) => {
                // If we start a new line set we first clear out the old one
                self.audio_sink.clear();
                self.current_request = None;
                self.current_queue = lines;
                if let Some(request) = self.current_queue.pop_front() {
                    self.start_playback_request(request).await?;
                }
                // Add the remaining items to a generation queue so that playbacks after the current one are quick
                if !self.current_queue.is_empty() {
                    let queue_lines = self.current_queue.iter().map(|l| l.line.clone()).collect();
                    self.session_handle
                        .send(GameSessionMessage::AddToQueue(queue_lines))
                        .await?;
                    // As we're preemptively sending this off we should ensure we don't request _another_ regeneration when actually playing this line.
                    self.current_queue.iter_mut().for_each(|l| l.line.force_generate = false);
                }
            }
        }
        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn handle_tts_sample(&mut self, tts: Arc<TtsResponse>) -> eyre::Result<()> {
        self.audio_sink.set_volume(self.current_volume);
        self.current_request = None;

        let audio_file = BufReader::new(File::open(&tts.file_path)?);
        let source = Decoder::new(audio_file)?;
        self.audio_sink.append(source);
        self.audio_sink.play();
        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn handle_queue_tick(&mut self) -> eyre::Result<()> {
        if self.audio_sink.empty() && self.current_request.is_none() {
            if let Some(request) = self.current_queue.pop_front() {
                self.start_playback_request(request).await?;
            }
        }

        Ok(())
    }

    #[tracing::instrument(skip_all)]
    async fn start_playback_request(&mut self, request: PlaybackVoiceLine) -> eyre::Result<()> {
        let (send, recv) = tokio::sync::oneshot::channel();
        self.current_request = Some(recv);
        self.current_volume = request.volume.unwrap_or(1.0);
        
        self.session_handle
            .send(GameSessionMessage::Single(request.line, send))
            .await?;

        Ok(())
    }
}

// Since we only care about Windows we can safely assume that our audio stream in `cpal` is [Send].
// We need this property to make this work as a tokio actor.
struct SendCpalStream(OutputStream);
unsafe impl Send for SendCpalStream {}
