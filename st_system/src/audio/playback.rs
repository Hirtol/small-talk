use crate::{
    audio::{
        scale_tempo::{RefInterlacedSamples, ScaleTempo},
        AudioData,
    }, data::TtsModel,
    session::{GameSessionHandle, GameTts},
    voice_manager::VoiceManager,
    TtsResponse,
    VoiceLine,
};
use eyre::{ContextCompat, OptionExt};
use futures::{future::BoxFuture, FutureExt};
use kira::{
    effect::{
        filter::{FilterBuilder, FilterMode},
        reverb::ReverbBuilder,
    }, sound::{
        static_sound::{StaticSoundData, StaticSoundHandle},
        PlaybackState,
    }, track::{TrackBuilder, TrackHandle}, AudioManager, AudioManagerSettings,
    Decibels,
    DefaultBackend,
    Tween,
};
use std::{
    collections::VecDeque,
    fs::File,
    io::BufReader,
    sync::{Arc, Weak},
    time::Duration,
};
use tokio::sync::broadcast;

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
            session_handle: session,
            recv,
            current_queue: Default::default(),
            scale_tempo: create_scale_tempo(44_100),
            state: PlaybackEngineState::Idle,
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

    /// Set the speed of playback for an ongoing playback.
    pub async fn set_speed(&self, new_speed: f64) -> eyre::Result<()> {
        Ok(self.send.send(PlaybackMessage::ChangeSpeed(new_speed)).await?)
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
    ChangeSpeed(f64),
}

pub struct PlaybackEngine {
    session_handle: Weak<GameTts>,

    recv: tokio::sync::mpsc::Receiver<PlaybackMessage>,

    scale_tempo: ScaleTempo,
    audio_manager: AudioManager<DefaultBackend>,
    state: PlaybackEngineState,

    current_queue: VecDeque<PlaybackVoiceLine>,
}

impl PlaybackEngine {
    #[tracing::instrument(name = "run_playback", skip(self))]
    pub async fn run(mut self) -> eyre::Result<()> {
        // There is no callback/future we can use to detect a finished line, so we'll just have to poll it.
        let mut check_interval = tokio::time::interval(Duration::from_millis(100));
        loop {
            let one_shot_future: futures::future::OptionFuture<_> = match &mut self.state {
                PlaybackEngineState::Waiting(state) => Some(&mut state.current_request),
                _ => None,
            }
            .into();

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
                self.state = PlaybackEngineState::Idle;
                self.current_queue.clear();
            }
            PlaybackMessage::Start(lines) => {
                // If we start a new line set we first clear out the old one
                self.state = PlaybackEngineState::Idle;
                self.current_queue = lines;
                let session = self.session()?;

                // Actually request our first voice line
                if let Some(request) = self.current_queue.pop_front() {
                    self.start_playback_request(request, session.clone()).await?;
                }
                // Add the items to a generation queue so that playbacks after the current one are quick
                if !self.current_queue.is_empty() {
                    session
                        .add_all_to_queue(self.current_queue.iter().map(|l| l.line.clone()).collect())
                        .await?;
                    // As we're preemptively sending these off we should ensure we don't request _another_ regeneration when actually playing this line.
                    self.current_queue
                        .iter_mut()
                        .for_each(|l| l.line.force_generate = false);
                }
            }
            PlaybackMessage::ChangeSpeed(new_speed) => {
                self.set_speed(new_speed)?;
            }
        }
        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn handle_tts_sample(&mut self, tts: Arc<TtsResponse>) -> eyre::Result<()> {
        let Ok(mut file) = StaticSoundData::from_file(&tts.file_path) else {
            // Can only happen if the cache was corrupted somehow (or the user's filesystem is broken)
            tracing::warn!(?tts.file_path, "Given file-path for TTS line was invalid, requesting new generation");
            self.state = PlaybackEngineState::Idle;
            return Ok(());
        };

        let current_state = self
            .state
            .take_waiting()
            .ok_or_eyre("Unexpected state machine state, bailing")?;
        let speed_sound_data = process_speed_change_data(
            &mut self.scale_tempo,
            file.clone(),
            current_state.current_settings.speed.unwrap_or(1.0),
        );

        let mut track = self
            .audio_manager
            .add_sub_track(current_state.current_settings.construct_track())?;
        let volume = current_state.current_settings.volume.unwrap_or(1.0).max(0.0).min(1.0);
        let volume_db = Decibels(20.0 * volume.log10());

        track.set_volume(volume_db, Tween::default());

        let new_state = PlaybackEnginePlayingState {
            current_sound: track.play(speed_sound_data)?,
            current_track: track,
            current_sound_data: file,
            current_settings: current_state.current_settings,
        };
        self.state = PlaybackEngineState::Playing(new_state);
        Ok(())
    }

    fn set_speed(&mut self, new_speed: f64) -> eyre::Result<()> {
        match &mut self.state {
            PlaybackEngineState::Idle => {}
            PlaybackEngineState::Waiting(state) => {
                state.current_settings.speed = Some(new_speed);
            }
            PlaybackEngineState::Playing(state) => {
                let current_sound_handle = &mut state.current_sound;
                current_sound_handle.stop(Tween::default());
                let current_position = current_sound_handle.position();

                let new_sound_data =
                    process_speed_change_data(&mut self.scale_tempo, state.current_sound_data.clone(), new_speed);
                let mut new_sound_handle = state.current_track.play(new_sound_data)?;

                let current_speed = state.current_settings.speed.unwrap_or(1.0);
                let position_divider = new_speed / current_speed;
                let new_position = current_position / position_divider;
                new_sound_handle.seek_to(new_position);

                state.current_sound = new_sound_handle;
                state.current_settings.speed = Some(new_speed);
            }
        }
        Ok(())
    }

    async fn handle_queue_tick(&mut self) -> eyre::Result<()> {
        let has_stopped = match &self.state {
            PlaybackEngineState::Playing(state) => state.current_sound.state() == PlaybackState::Stopped,
            _ => false,
        };

        if has_stopped && !matches!(self.state, PlaybackEngineState::Waiting(_)) {
            if let Some(request) = self.current_queue.pop_front() {
                self.start_playback_request(request, self.session()?).await?;
            }
        }

        Ok(())
    }

    #[tracing::instrument(skip_all)]
    async fn start_playback_request(&mut self, request: PlaybackVoiceLine, session: Arc<GameTts>) -> eyre::Result<()> {
        let (snd, rcv) = tokio::sync::oneshot::channel();

        tokio::task::spawn(async move {
            if let Err(e) = session.request_tts_with_channel(request.line, snd).await {
                tracing::error!(?e, "Failed to request TTS for playback");
            }
        });

        let new_state = PlaybackEngineWaitingState {
            current_settings: request.playback.unwrap_or_default(),
            current_request: rcv,
        };
        self.state = PlaybackEngineState::Waiting(new_state);

        Ok(())
    }

    fn session(&self) -> eyre::Result<Arc<GameTts>> {
        self.session_handle
            .upgrade()
            .context("Parent session is no longer available")
    }
}

/// Speed up the given `data` while preserving pitch.
fn process_speed_change_data(scale_tempo: &mut ScaleTempo, mut data: StaticSoundData, speed: f64) -> StaticSoundData {
    if scale_tempo.sample_rate != data.sample_rate {
        *scale_tempo = create_scale_tempo(data.sample_rate);
    }

    if speed != 1.0 {
        let new_samples = scale_tempo.process(RefInterlacedSamples(&data.frames).as_ref(), speed);

        data.frames = super::scale_tempo::to_kira_frames(new_samples);
        data.slice = None;
        data
    } else {
        data
    }
}

#[derive(Debug)]
enum PlaybackEngineState {
    Idle,
    Waiting(PlaybackEngineWaitingState),
    Playing(PlaybackEnginePlayingState),
}

impl PlaybackEngineState {
    pub fn take_waiting(&mut self) -> Option<PlaybackEngineWaitingState> {
        match self {
            PlaybackEngineState::Waiting(_) => {
                let old_state = std::mem::replace(self, PlaybackEngineState::Idle);

                match old_state {
                    PlaybackEngineState::Waiting(state) => Some(state),
                    _ => unreachable!(),
                }
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
struct PlaybackEngineWaitingState {
    current_settings: PlaybackSettings,

    current_request: tokio::sync::oneshot::Receiver<Arc<TtsResponse>>,
}

#[derive(Debug)]
struct PlaybackEnginePlayingState {
    current_track: TrackHandle,
    current_sound: StaticSoundHandle,
    current_sound_data: StaticSoundData,
    current_settings: PlaybackSettings,
}

/// The environment which we should simulate through reverb/filters
///
/// # Variants
/// * `Outdoors` - No applied reverb
/// * `Indoors` - Modicum of reverb
/// * `Cave` - Large amount of reverb
#[derive(
    serde::Deserialize, serde::Serialize, Debug, schemars::JsonSchema, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash,
)]
pub enum PlaybackEnvironment {
    Outdoors,
    Indoors,
    Cave,
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct PlaybackSettings {
    /// The environment of the listener.
    ///
    /// Affects the amount of reverb applied
    pub environment: Option<PlaybackEnvironment>,
    /// Playback volume, should be in the interval `[0.0, 1.0]`
    pub volume: Option<f32>,
    /// Playback speed, should be strictly positive
    pub speed: Option<f64>,
}

impl PlaybackSettings {
    /// Create a track based on these playback settings
    ///
    /// Applies:
    /// * Low-pass filter at `16_000` HZ
    /// * Optional Reverb based on environment
    /// * High-pass filter for outdoors environment simulation
    fn construct_track(&self) -> TrackBuilder {
        let mut builder = TrackBuilder::new();
        builder.add_effect(FilterBuilder::new().mode(FilterMode::LowPass).cutoff(16_000.));
        if let Some(env) = self.environment {
            // Arbitrarily picked based on what sounded decent
            // Outdoors is equivalent to no reverb at all.
            let (mix, feedback) = match env {
                PlaybackEnvironment::Outdoors => (0.003, 0.5),
                PlaybackEnvironment::Indoors => (0.04, 0.1),
                PlaybackEnvironment::Cave => (0.2, 0.6),
            };
            builder.add_effect(ReverbBuilder::new().mix(mix).feedback(feedback));

            if let PlaybackEnvironment::Outdoors = env {
                // High pass filter to somewhat simulate outdoors environments.
                builder.add_effect(FilterBuilder::new().mode(FilterMode::HighPass).cutoff(130.));
            }
        }

        builder
    }
}

fn create_scale_tempo(sample_rate: u32) -> ScaleTempo {
    ScaleTempo::new(sample_rate, 2, 30, 0.2, 14)
}
