use st_system::voice_manager::{FsVoice, FsVoiceData};

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

#[derive(uniffi::Enum, Clone, Debug)]
pub enum VoiceDestination {
    Global,
    Game(String),
}

impl From<st_system::voice_manager::VoiceDestination> for VoiceDestination {
    fn from(value: st_system::voice_manager::VoiceDestination) -> Self {
        match value {
            st_system::voice_manager::VoiceDestination::Global => Self::Global,
            st_system::voice_manager::VoiceDestination::Game(game) => Self::Game(game),
        }
    }
}
