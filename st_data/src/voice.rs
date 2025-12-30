use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Internal name for a particular voice.
pub type Voice = String;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct VoiceReference {
    pub name: Voice,
    pub location: VoiceLocation,
}

impl VoiceReference {
    pub fn from_strings(name: Voice, location: String) -> VoiceReference {
        Self {
            name,
            location: VoiceLocation::from(location),
        }
    }

    pub fn global(name: impl Into<Voice>) -> VoiceReference {
        VoiceReference {
            name: name.into(),
            location: VoiceLocation::Global,
        }
    }

    pub fn game(name: impl Into<Voice>, game_name: impl Into<String>) -> VoiceReference {
        VoiceReference {
            name: name.into(),
            location: VoiceLocation::Game(game_name.into()),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum VoiceLocation {
    Global,
    Game(String)
}

impl VoiceLocation {
    pub fn to_string_value(&self) -> String {
        match self {
            VoiceLocation::Global => "global".into(),
            VoiceLocation::Game(game_val) => game_val.clone()
        }
    }
}

impl From<String> for VoiceLocation {
    fn from(value: String) -> Self {
        if value == "global" || value == "Global" {
            Self::Global
        } else {
            Self::Game(value)
        }
    }
}