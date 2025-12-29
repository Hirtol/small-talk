use std::path::PathBuf;

#[derive(uniffi::Record, Clone, Debug)]
pub struct VoiceReference {
    pub name: String,
    pub location: VoiceDestination,
}

impl From<st_system::voice_manager::VoiceReference> for VoiceReference {
    fn from(value: st_system::voice_manager::VoiceReference) -> Self {
        Self {
            name: value.name,
            location: value.location.into(),
        }
    }
}

impl From<VoiceReference> for st_system::voice_manager::VoiceReference {
    fn from(value: VoiceReference) -> Self {
        Self {
            name: value.name,
            location: value.location.into(),
        }
    }
}

#[derive(uniffi::Enum, Clone, Debug)]
pub enum VoiceDestination {
    Global,
    Game { name: String },
}

impl From<st_system::voice_manager::VoiceDestination> for VoiceDestination {
    fn from(value: st_system::voice_manager::VoiceDestination) -> Self {
        match value {
            st_system::voice_manager::VoiceDestination::Global => Self::Global,
            st_system::voice_manager::VoiceDestination::Game(game) => Self::Game { name: game },
        }
    }
}

impl From<VoiceDestination> for st_system::voice_manager::VoiceDestination {
    fn from(value: VoiceDestination) -> Self {
        match value {
            VoiceDestination::Global => st_system::voice_manager::VoiceDestination::Global,
            VoiceDestination::Game { name } => st_system::voice_manager::VoiceDestination::Game(name)
        }
    }
}

#[derive(uniffi::Record, Clone, Debug)]
pub struct VoiceLine {
    pub line: String,
    pub person: TtsVoice,
    pub model: TtsModel,
    pub force_generate: bool,
    pub post: Option<PostProcessing>,
}

impl From<st_system::data::VoiceLine> for VoiceLine {
    fn from(value: st_system::data::VoiceLine) -> Self {
        Self {
            line: value.line,
            person: value.person.into(),
            model: value.model.into(),
            force_generate: value.force_generate,
            post: value.post.map(|p| p.into()),
        }
    }
}

impl From<VoiceLine> for st_system::data::VoiceLine {
    fn from(value: VoiceLine) -> Self {
        Self {
            line: value.line,
            person: value.person.into(),
            model: value.model.into(),
            force_generate: value.force_generate,
            post: value.post.map(|p| p.into()),
        }
    }
}

#[derive(uniffi::Record, Clone, Debug)]
pub struct TtsResponse {
    pub file_path: String,
    pub line: String,
    pub voice_used: VoiceReference,
}

impl From<st_system::data::TtsResponse> for TtsResponse {
    fn from(value: st_system::data::TtsResponse) -> Self {
        Self {
            file_path: value.file_path.to_string_lossy().to_string(),
            line: value.line,
            voice_used: value.voice_used.into(),
        }
    }
}

impl From<TtsResponse> for st_system::data::TtsResponse {
    fn from(value: TtsResponse) -> Self {
        Self {
            file_path: PathBuf::from(value.file_path),
            line: value.line,
            voice_used: value.voice_used.into(),
        }
    }
}

#[derive(uniffi::Enum, Clone, Debug)]
pub enum TtsVoice {
    ForceVoice(VoiceReference),
    CharacterVoice(CharacterVoice),
}

impl From<st_system::data::TtsVoice> for TtsVoice {
    fn from(value: st_system::data::TtsVoice) -> Self {
        match value {
            st_system::data::TtsVoice::ForceVoice(voice_ref) => Self::ForceVoice(voice_ref.into()),
            st_system::data::TtsVoice::CharacterVoice(char) => Self::CharacterVoice(char.into()),
        }
    }
}

impl From<TtsVoice> for st_system::data::TtsVoice {
    fn from(value: TtsVoice) -> Self {
        match value {
            TtsVoice::ForceVoice(voice_ref) => Self::ForceVoice(voice_ref.into()),
            TtsVoice::CharacterVoice(char) => Self::CharacterVoice(char.into()),
        }
    }
}

#[derive(uniffi::Record, Clone, Debug)]
pub struct PostProcessing {
    pub verify_percentage: Option<u8>,
    pub trim_silence: bool,
    pub normalise: bool,
    pub rvc: Option<RvcOptions>,
}

impl From<st_system::data::PostProcessing> for PostProcessing {
    fn from(value: st_system::data::PostProcessing) -> Self {
        Self {
            verify_percentage: value.verify_percentage,
            trim_silence: value.trim_silence,
            normalise: value.normalise,
            rvc: value.rvc.map(|r| r.into()),
        }
    }
}

impl From<PostProcessing> for st_system::data::PostProcessing {
    fn from(value: PostProcessing) -> Self {
        Self {
            verify_percentage: value.verify_percentage,
            trim_silence: value.trim_silence,
            normalise: value.normalise,
            rvc: value.rvc.map(|r| r.into()),
        }
    }
}

#[derive(uniffi::Record, Clone, Debug)]
pub struct RvcOptions {
    pub model: RvcModel,
    pub high_quality: bool,
}

impl From<st_system::data::RvcOptions> for RvcOptions {
    fn from(value: st_system::data::RvcOptions) -> Self {
        Self {
            model: value.model.into(),
            high_quality: value.high_quality,
        }
    }
}

impl From<RvcOptions> for st_system::data::RvcOptions {
    fn from(value: RvcOptions) -> Self {
        Self {
            model: value.model.into(),
            high_quality: value.high_quality,
        }
    }
}

#[derive(uniffi::Enum, Clone, Debug)]
pub enum RvcModel {
    SeedVc,
}

impl From<st_system::data::RvcModel> for RvcModel {
    fn from(value: st_system::data::RvcModel) -> Self {
        match value {
            st_system::data::RvcModel::SeedVc => Self::SeedVc,
        }
    }
}

impl From<RvcModel> for st_system::data::RvcModel {
    fn from(value: RvcModel) -> Self {
        match value {
            RvcModel::SeedVc => Self::SeedVc,
        }
    }
}

#[derive(uniffi::Enum, Clone, Debug)]
pub enum TtsModel {
    Xtts,
    IndexTts,
    EchoTts,
}

impl From<st_system::data::TtsModel> for TtsModel {
    fn from(value: st_system::data::TtsModel) -> Self {
        match value {
            st_system::data::TtsModel::Xtts => Self::Xtts,
            st_system::data::TtsModel::IndexTts => Self::IndexTts,
            st_system::data::TtsModel::EchoTts => Self::EchoTts,
        }
    }
}

impl From<TtsModel> for st_system::data::TtsModel {
    fn from(value: TtsModel) -> Self {
        match value {
            TtsModel::Xtts => st_system::data::TtsModel::Xtts,
            TtsModel::IndexTts => Self::IndexTts,
            TtsModel::EchoTts => Self::EchoTts,
        }
    }
}

#[derive(uniffi::Enum, Clone, Debug, Copy)]
pub enum PlaybackEnvironment {
    Outdoors,
    Indoors,
    Cave,
}

impl From<st_system::audio::playback::PlaybackEnvironment> for PlaybackEnvironment {
    fn from(value: st_system::audio::playback::PlaybackEnvironment) -> Self {
        match value {
            st_system::audio::playback::PlaybackEnvironment::Outdoors => Self::Outdoors,
            st_system::audio::playback::PlaybackEnvironment::Indoors => Self::Indoors,
            st_system::audio::playback::PlaybackEnvironment::Cave => Self::Cave,
        }
    }
}

impl From<PlaybackEnvironment> for st_system::audio::playback::PlaybackEnvironment {
    fn from(value: PlaybackEnvironment) -> Self {
        match value {
            PlaybackEnvironment::Outdoors => st_system::audio::playback::PlaybackEnvironment::Outdoors,
            PlaybackEnvironment::Indoors => st_system::audio::playback::PlaybackEnvironment::Indoors,
            PlaybackEnvironment::Cave => st_system::audio::playback::PlaybackEnvironment::Cave,
        }
    }
}

#[derive(uniffi::Record, Clone, Debug)]
pub struct PlaybackSettings {
    pub environment: Option<PlaybackEnvironment>,
    pub volume: Option<f32>,
    pub speed: Option<f64>,
}

impl From<st_system::audio::playback::PlaybackSettings> for PlaybackSettings {
    fn from(value: st_system::audio::playback::PlaybackSettings) -> Self {
        Self {
            environment: value.environment.map(|e| e.into()),
            volume: value.volume,
            speed: value.speed,
        }
    }
}

impl From<PlaybackSettings> for st_system::audio::playback::PlaybackSettings {
    fn from(value: PlaybackSettings) -> Self {
        Self {
            environment: value.environment.map(|e| e.into()),
            volume: value.volume,
            speed: value.speed,
        }
    }
}

#[derive(uniffi::Record, Clone, Debug)]
pub struct PlaybackVoiceLine {
    pub request: VoiceLine,
    pub playback: Option<PlaybackSettings>,
}

impl From<st_system::audio::playback::PlaybackVoiceLine> for PlaybackVoiceLine {
    fn from(value: st_system::audio::playback::PlaybackVoiceLine) -> Self {
        Self {
            request: value.line.into(),
            playback: value.playback.map(|p| p.into()),
        }
    }
}

impl From<PlaybackVoiceLine> for st_system::audio::playback::PlaybackVoiceLine {
    fn from(value: PlaybackVoiceLine) -> Self {
        Self {
            line: value.request.into(),
            playback: value.playback.map(|p| p.into()),
        }
    }
}

#[derive(uniffi::Record, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CharacterVoice {
    pub name: String,
    pub gender: Option<Gender>,
}

impl From<st_system::CharacterVoice> for CharacterVoice {
    fn from(value: st_system::CharacterVoice) -> Self {
        Self {
            name: value.name,
            gender: value.gender.map(|i| i.into()),
        }
    }
}

impl From<CharacterVoice> for st_system::data::CharacterVoice {
    fn from(value: CharacterVoice) -> Self {
        Self {
            name: value.name,
            gender: value.gender.map(|i| i.into()),
        }
    }
}

#[derive(uniffi::Enum, Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Gender {
    Male,
    Female,
}

impl From<st_system::Gender> for Gender {
    fn from(value: st_system::Gender) -> Self {
        match value {
            st_system::Gender::Male => Gender::Male,
            st_system::Gender::Female => Gender::Female,
        }
    }
}

impl From<Gender> for st_system::Gender {
    fn from(value: Gender) -> Self {
        match value {
            Gender::Male => st_system::Gender::Male,
            Gender::Female => st_system::Gender::Female,
        }
    }
}
