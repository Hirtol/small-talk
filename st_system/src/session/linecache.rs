use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use serde::de::Error;
use crate::voice_manager::{VoiceDestination, VoiceReference};

#[derive(Debug, Clone, Default)]
pub struct LineCache {
    /// Voice -> Line voiced -> file name
    pub voice_to_line: HashMap<VoiceReference, HashMap<String, String>>,
}

// Needed in order to properly handle VoiceReference
impl Serialize for LineCache {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Create a temporary HashMap<String, HashMap<String, String>>
        let transformed: HashMap<String, HashMap<String, String>> = self
            .voice_to_line
            .iter()
            .map(|(key, value)| {
                let key_str = match &key.location {
                    VoiceDestination::Global => format!("global_{}", key.name),
                    VoiceDestination::Game(game_name) => format!("game_{game_name}_{}", key.name),
                };
                (key_str, value.clone())
            })
            .collect();

        transformed.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for LineCache {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize into a temporary HashMap<String, HashMap<String, String>>
        let raw_map: HashMap<String, HashMap<String, String>> = HashMap::deserialize(deserializer)?;

        // Convert back to HashMap<VoiceReference, HashMap<String, String>>
        let voice_to_line = raw_map
            .into_iter()
            .map(|(key, value)| {
                let (location, name) = if let Some(rest) = key.strip_prefix("global_") {
                    (VoiceDestination::Global, rest.to_string())
                } else if let Some(rest) = key.strip_prefix("game_") {
                    let (game_name, character) = rest
                        .split_once("_")
                        .ok_or_else(|| D::Error::custom("No game identifier found"))?;
                    (VoiceDestination::Game(game_name.into()), character.to_string())
                } else {
                    return Err(serde::de::Error::custom(format!("Invalid key format: {}", key)));
                };

                Ok((VoiceReference { name, location }, value))
            })
            .collect::<Result<HashMap<_, _>, D::Error>>()?;

        Ok(LineCache { voice_to_line })
    }
}