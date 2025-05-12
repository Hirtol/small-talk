use std::path::PathBuf;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use crate::session::db::DatabaseGender;
use crate::voice_manager::VoiceReference;

/// Internal name for a particular voice.
pub type Voice = String;

/// The name of a character, this will be associated with a set voice
pub type CharacterName = String;

#[derive(Debug, Clone)]
pub struct TtsResponse {
    /// Local file path to the generated line 
    pub file_path: PathBuf,
    pub line: VoiceLine,
    pub voice_used: Voice,
}

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Deserialize, Serialize)]
pub struct VoiceLine {
    pub line: String,
    /// The person who ought to voice the line
    pub person: TtsVoice,
    pub model: TtsModel,
    /// Force the generation of a new line, even if it already existed in the cache.
    pub force_generate: bool,
    /// Optional audio post-processing
    pub post: Option<PostProcessing>
}

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Deserialize, Serialize, JsonSchema)]
pub struct PostProcessing {
    /// Verify whether a voice line was generated correctly by running Whisper on it.
    ///
    /// The given percentage should be in the range `[0..100]`,
    /// where a higher percentage means a larger match with the original prompt.
    /// If the TTS is below this threshold it will be regenerated.
    pub verify_percentage: Option<u8>,
    /// Whether to remove leading and trailing silences from the generated file
    pub trim_silence: bool,
    /// Whether to normalise the audio that was generated.
    pub normalise: bool,
    /// Whether to use RVC (seed-vc)
    pub rvc: Option<RvcOptions>
}

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Deserialize, Serialize, JsonSchema)]
pub struct RvcOptions {
    pub model: RvcModel,
    /// Whether to prefer high-quality (`true`) or faster conversion (`false`)
    pub high_quality: bool,
}

#[derive(Deserialize, Serialize, Debug, JsonSchema, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum RvcModel {
    SeedVc,
}

#[derive(Deserialize, Serialize, Debug, JsonSchema, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum TtsVoice {
    /// Force the request to use the given [Voice]
    ForceVoice(VoiceReference),
    /// Let the backend handle voice assignment for this character.
    CharacterVoice(CharacterVoice),
}

#[derive(Deserialize, Serialize, Debug, JsonSchema, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct CharacterVoice {
    /// The name of the character speaking
    pub name: CharacterName,
    /// The gender of the given person.
    /// 
    /// If this [CharacterName] does not yet have a [Voice] assigned a random one with a fitting gender will be assigned.
    pub gender: Option<Gender>,
}

#[derive(Deserialize, Serialize, Debug, JsonSchema, Clone, Copy, Ord, PartialOrd, Eq, PartialEq, Hash, Default)]
pub enum Gender {
    #[default]
    Male,
    Female,
}

impl Gender {
    pub fn to_db(self) -> DatabaseGender {
        self.into()
    }
}

impl From<DatabaseGender> for Gender {
    fn from(value: DatabaseGender) -> Self {
        match value {
            DatabaseGender::Male => Gender::Male,
            DatabaseGender::Female => Gender::Female
        }
    }
}

impl From<Gender> for DatabaseGender {
    fn from(value: Gender) -> Self {
        match value {
            Gender::Male => {
                DatabaseGender::Male
            }
            Gender::Female => DatabaseGender::Female
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub enum TtsModel {
    Xtts,
}