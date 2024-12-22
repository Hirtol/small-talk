use crate::system::{session::{GameSessionMessage}, voice_manager::VoiceManager, TtsModel, TtsResponse, VoiceLine};
use rodio::{Decoder, OutputStream, Sink};
use std::{fs::File, io::BufReader, sync::Arc};
use tokio::sync::{broadcast};

#[derive(Clone)]
pub struct PlaybackEngineHandle {
    send: tokio::sync::mpsc::Sender<PlaybackMessage>,
}

impl PlaybackEngineHandle {
    /// Start a new playback engine
    pub async fn new(session: tokio::sync::mpsc::Sender<GameSessionMessage>, ) -> eyre::Result<PlaybackEngineHandle> {
        let (send, recv) = tokio::sync::mpsc::channel(10);
        let (stream, stream_handle) = OutputStream::try_default().expect("No output device for audio available");

        let engine = PlaybackEngine {
            _output_device: SendCpalStream(stream),
            audio_sink: Sink::try_new(&stream_handle).expect("Failed to initialise audio sink."),
            session_handle: session,
            recv,
            current_request: None,
        };
        let rt = tokio::runtime::Handle::current();
        // We do blocking IO in the actor, so spawn it on the thread pool.
        tokio::task::spawn_blocking(move || rt.block_on(async move {
            if let Err(e) = engine.run().await {
                tracing::error!("PlaybackEngine stopped with error: {e}");
            }
        }));

        Ok(Self { send })
    }
    
    /// Start the playback of the given line.
    /// 
    /// If the TTS request hasn't been completed (or requested) the playback engine will wait until it is available.
    /// The playback can be cancelled using [Self::stop], or by simply [Self::start]ing another line.
    /// If the engine was waiting for a different line to be completed then it will simply discard that initial request and wait for the new line instead.
    /// 
    /// This method returns immediately, it does not wait for playback to be completed
    pub async fn start(&self, line: VoiceLine) -> eyre::Result<()> {
        Ok(self.send.send(PlaybackMessage::Start(line)).await?)
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
pub enum PlaybackMessage {
    Stop,
    Start(VoiceLine),
}

pub struct PlaybackEngine {
    _output_device: SendCpalStream,
    audio_sink: Sink,
    session_handle: tokio::sync::mpsc::Sender<GameSessionMessage>,
    recv: tokio::sync::mpsc::Receiver<PlaybackMessage>,
    current_request: Option<tokio::sync::oneshot::Receiver<Arc<TtsResponse>>>
}

impl PlaybackEngine {
    #[tracing::instrument(skip(self))]
    pub async fn run(mut self) -> eyre::Result<()> {
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
                    self.handle_tts(tts).await?;
                },
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
                self.audio_sink.clear();
            }
            PlaybackMessage::Start(line) => {
                // If we start a new line we first clear out the old one
                self.audio_sink.clear();
                // We just send the request, we then wait for the broadcast channel to give us a result while allowing
                // [Stop] messages to interrupt our playback.
                let (send, recv) = tokio::sync::oneshot::channel();
                self.current_request = Some(recv);
                self.session_handle.send(GameSessionMessage::Single(line, send)).await?;
            }
        }
        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn handle_tts(&mut self, tts: Arc<TtsResponse>) -> eyre::Result<()> {
        self.current_request = None;
        // Xtts tends to produce more quiet outputs, whereas F5 can be a bit deafening.
        match tts.line.model {
            TtsModel::F5 => self.audio_sink.set_volume(0.4),
            TtsModel::Xtts => self.audio_sink.set_volume(1.0),
        }

        let audio_file = BufReader::new(File::open(&tts.file_path)?);
        let source = Decoder::new(audio_file)?;
        self.audio_sink.append(source);
        self.audio_sink.play();
        Ok(())
    }
}

// Since we only care about Windows we can safely assume that our audio stream in `cpal` is [Send].
// We need this property to make this work as a tokio actor.
struct SendCpalStream(OutputStream);
unsafe impl Send for SendCpalStream {}
