use std::path::PathBuf;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use crate::system::TtsModel;
use crate::system::voice_manager::VoiceReference;

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

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
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

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct PostProcessing {
    /// Whether to remove leading and trailing silences from the generated file
    pub trim_silence: bool,
    /// Whether to use RVC (seed-vc)
    pub rvc: Option<RvcOptions>
}

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct RvcOptions {
    pub model: RvcModel,
    /// Whether to prefer high-quality (`true`) or faster conversion (`false`)
    pub high_quality: bool,
}

#[derive(Deserialize, Serialize, Debug, JsonSchema, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum RvcModel {
    /// Zero-shot RVC model
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

#[derive(Deserialize, Serialize, Debug, JsonSchema, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum Gender {
    Male,
    Female,
}