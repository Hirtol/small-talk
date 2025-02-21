use crate::{
    TtsResponse, VoiceLine,
};
use eyre::ContextCompat;
use futures::{future::BoxFuture, FutureExt};
use std::{
    collections::VecDeque,
    fs::File,
    io::BufReader,
    sync::{Arc, Weak},
    time::Duration,
};
use kira::{AudioManager, AudioManagerSettings, Decibels, DefaultBackend, Tween};
use kira::effect::filter::FilterBuilder;
use kira::effect::reverb::ReverbBuilder;
use kira::sound::PlaybackState;
use kira::sound::static_sound::{StaticSoundData, StaticSoundHandle};
use kira::track::{TrackBuilder, TrackHandle};
use crate::session::{GameSessionHandle, GameTts};
use crate::voice_manager::VoiceManager;
use tokio::sync::broadcast;
use crate::data::TtsModel;

#[derive(Clone)]
pub struct PlaybackEngineHandle {
    send: tokio::sync::mpsc::Sender<PlaybackMessage>,
}

impl PlaybackEngineHandle {
    /// Start a new playback engine
    pub async fn new(session: Weak<GameTts>) -> eyre::Result<PlaybackEngineHandle> {
        let (send, recv) = tokio::sync::mpsc::channel(10);
        let audio_manager = kira::AudioManager::<DefaultBackend>::new(AudioManagerSettings::default())?;

        let engine = PlaybackEngine {
            audio_manager,
            current_track: None,
            session_handle: session,
            recv,
            current_request: None,
            current_settings: None,
            current_queue: Default::default(),
            current_sound: None,
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
    pub playback: Option<PlaybackSettings>,
}

#[derive(Debug, Clone)]
pub enum PlaybackMessage {
    Stop,
    Start(VecDeque<PlaybackVoiceLine>),
}

pub struct PlaybackEngine {
    session_handle: Weak<GameTts>,

    recv: tokio::sync::mpsc::Receiver<PlaybackMessage>,

    audio_manager: AudioManager<DefaultBackend>,
    current_track: Option<TrackHandle>,
    current_sound: Option<StaticSoundHandle>,
    current_settings: Option<PlaybackSettings>,

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

        tracing::trace!("Stopping PlaybackEngine for unknown reason");

        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn handle_message(&mut self, message: PlaybackMessage) -> eyre::Result<()> {
        match message {
            PlaybackMessage::Stop => {
                self.current_request = None;
                self.current_track = None;
                self.current_sound = None;
                self.current_settings = None;
                self.current_queue.clear();
            }
            PlaybackMessage::Start(lines) => {
                // If we start a new line set we first clear out the old one
                self.current_request = None;
                self.current_track = None;
                self.current_sound = None;
                self.current_settings = None;
                self.current_queue = lines;
                let session = self.session()?;

                // Actually request our first voice line
                if let Some(request) = self.current_queue.pop_front() {
                    self.start_playback_request(request, session.clone()).await?;
                }
                // Add the items to a generation queue so that playbacks after the current one are quick
                if !self.current_queue.is_empty() {
                    session.add_all_to_queue(self.current_queue.iter().map(|l| l.line.clone()).collect()).await?;
                    // As we're preemptively sending these off we should ensure we don't request _another_ regeneration when actually playing this line.
                    self.current_queue
                        .iter_mut()
                        .for_each(|l| l.line.force_generate = false);
                }
            }
        }
        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn handle_tts_sample(&mut self, tts: Arc<TtsResponse>) -> eyre::Result<()> {
        let Ok(file) = StaticSoundData::from_file(&tts.file_path) else {
            // Can only happen if the cache was corrupted somehow (or the user's filesystem is broken)
            // We'll just request a regeneration of this line
            tracing::warn!(?tts.file_path, "Given file-path for TTS line was invalid, requesting new generation");
            let session = self.session()?;
            let mut new_line = tts.line.clone();
            new_line.force_generate = true;

            self.start_playback_request(PlaybackVoiceLine {
                line: new_line,
                playback: self.current_settings.clone()
            }, session).await?;
            return Ok(())
        };

        self.current_request = None;
        let mut track = self.current_track.as_mut().expect("Invariant violation");
        self.current_sound = Some(track.play(file)?);
        Ok(())
    }

    async fn handle_queue_tick(&mut self) -> eyre::Result<()> {
        let has_stopped = self.current_sound.as_ref().map(|s| s.state() == PlaybackState::Stopped).unwrap_or_default();
        if has_stopped && self.current_request.is_none() {
            if let Some(request) = self.current_queue.pop_front() {
                self.start_playback_request(request, self.session()?).await?;
            }
        }

        Ok(())
    }

    #[tracing::instrument(skip_all)]
    async fn start_playback_request(&mut self, request: PlaybackVoiceLine, session: Arc<GameTts>) -> eyre::Result<()> {
        let (snd, rcv) = tokio::sync::oneshot::channel();
        let playback_s = request.playback.unwrap_or_default();
        let mut track = self.audio_manager.add_sub_track(playback_s.construct_track())?;
        let volume = playback_s.volume.unwrap_or(1.0).max(0.0).min(1.0);
        let volume_db = Decibels(20.0 * volume.log10());

        track.set_volume(volume_db, Tween::default());

        self.current_sound = None;
        self.current_track = Some(track);
        self.current_settings = Some(playback_s);
        tokio::task::spawn(async move { session.request_tts_with_channel(request.line, snd).await });
        self.current_request = Some(rcv);

        Ok(())
    }

    fn session(&self) -> eyre::Result<Arc<GameTts>> {
        self.session_handle
            .upgrade()
            .context("Parent session is no longer available")
    }
}

#[derive(serde::Deserialize, serde::Serialize, Debug, schemars::JsonSchema, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum PlaybackEnvironment {
    Outdoors,
    Indoors,
    Cave
}
/// Large amount of reverb
/// Modicum of reverb
/// No applied reverb

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct PlaybackSettings {
    /// The environment of the listener.
    ///
    /// Affects the amount of reverb applied
    pub environment: Option<PlaybackEnvironment>,
    /// Playback volume, should be in the interval `[0.0, 1.0]`
    pub volume: Option<f32>
}

impl PlaybackSettings {
    /// Create a track based on these playback settings
    ///
    /// Applies:
    /// * Low-pass filter at `16_000` HZ
    /// * Optional Reverb based on environment
    fn construct_track(&self) -> TrackBuilder {
        let mut builder = TrackBuilder::new();
        builder.add_effect(FilterBuilder::new().cutoff(16_000.));
        if let Some(env) = self.environment {
            // Arbitrarily picked based on what sounded decent
            // Outdoors is equivalent to no reverb at all.
            let (mix, feedback) = match env {
                PlaybackEnvironment::Outdoors => (0.0, 0.0),
                // PlaybackEnvironment::IndoorsSmall => (0.01, 0.1),
                PlaybackEnvironment::Indoors => (0.04, 0.1),
                PlaybackEnvironment::Cave => (0.2, 0.6),
            };
            builder.add_effect(ReverbBuilder::new().mix(mix).feedback(feedback));
        }

        builder
    }
}

