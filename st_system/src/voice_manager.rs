use std::collections::HashMap;
use itertools::Itertools;
use st_ml::emotion_classifier::BasicEmotion;
use std::path::PathBuf;
use std::sync::Arc;
use eyre::ContextCompat;
use path_abs::{PathInfo, PathOps};
use rand::prelude::IteratorRandom;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use walkdir::DirEntry;
use crate::config::TtsSystemConfig;
use crate::error::VoiceManagerError;
use crate::session::db;
use crate::Voice;

#[derive(Debug, Clone)]
pub struct VoiceManager {
    conf: Arc<TtsSystemConfig>,
}

impl VoiceManager {
    pub fn new(conf: Arc<TtsSystemConfig>) -> Self {
        Self { conf }
    }

    pub fn get_voice(&self, voice: VoiceReference) -> Result<FsVoiceData, VoiceManagerError> {
        // Necessary guard, if we don't have this we'd refer to the root `voice` directory!
        if voice.name.is_empty() {
            return Err(VoiceManagerError::VoiceDoesNotExist {
                voice: voice.name,
            });
        }

        let path = voice.location.to_path(&self.conf).join(&voice.name);

        if path.exists() {
            Ok(FsVoiceData {
                dir: path,
                reference: voice,
            })    
        } else {
            Err(VoiceManagerError::VoiceDoesNotExist {
                voice: voice.name,
            })
        }
    }

    /// Return all applicable voices (including game specific and global) for the given game.
    pub fn get_voices(&self, game_name: &str) -> Vec<FsVoiceData> {
        let mut result = self.get_global_voices();
        result.extend(self.get_game_voices(game_name));
        result
    }

    pub fn get_game_voices(&self, game_name: &str) -> Vec<FsVoiceData> {
        walkdir::WalkDir::new(self.conf.game_voice(game_name))
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_entry(|d| d.file_type().is_dir())
            .flatten()
            .map(|d| FsVoiceData {
                reference: VoiceReference {
                    name: d.file_name().to_string_lossy().into_owned(),
                    location: VoiceDestination::Game(game_name.into()),
                },
                dir: d.into_path(),
            })
            .collect_vec()
    }

    pub fn get_global_voices(&self) -> Vec<FsVoiceData> {
        walkdir::WalkDir::new(self.conf.global_voice())
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_entry(|d| d.file_type().is_dir())
            .flatten()
            .map(|d| FsVoiceData {
                reference: VoiceReference {
                    name: d.file_name().to_string_lossy().into_owned(),
                    location: VoiceDestination::Global,
                },
                dir: d.into_path(),
            })
            .collect_vec()
    }
    
    /// Store all given voice samples in the appropriate place in `dest`.
    /// 
    /// Renames the sample to the expected name representing the emotion embedded in the sample.
    /// This is later used for sample collection.
    pub fn store_voice_samples(&mut self, dest: VoiceDestination, voice_name: &str, samples: Vec<VoiceSample>) -> eyre::Result<()> {
        let destination = dest.to_path(&self.conf).join(voice_name);
        std::fs::create_dir_all(&destination)?;
        
        let mut existing_samples = {
            let refs = VoiceReference {
                name: voice_name.into(),
                location: dest,
            };
            if let Ok(voice_data) = self.get_voice(refs) {
                voice_data.get_samples()?
            } else {
                HashMap::default()
            }
        };
        
        for sample in samples {
            let sample_collection = existing_samples.entry(sample.emotion).or_default();
            let name = format!("{:?}_{}.wav", sample.emotion, sample_collection.len());
            let mut sample_dest = destination.join(name);
            std::fs::write(&sample_dest, sample.data)?;
            if let Some(text) = sample.spoken_text {
                sample_dest.set_extension("txt");
                std::fs::write(sample_dest, text)?
            }
        }
        
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct VoiceReference {
    pub name: Voice,
    pub location: VoiceDestination,
}

impl VoiceReference {
    pub fn from_strings(name: Voice, location: String) -> VoiceReference {
        Self {
            name,
            location: VoiceDestination::from(location),
        }
    }

    pub fn global(name: impl Into<Voice>) -> VoiceReference {
        VoiceReference {
            name: name.into(),
            location: VoiceDestination::Global,
        }
    }
    
    pub fn game(name: impl Into<Voice>, game_name: impl Into<String>) -> VoiceReference {
        VoiceReference {
            name: name.into(),
            location: VoiceDestination::Game(game_name.into()),
        }
    }
}

impl From<db::voice_lines::Model> for VoiceReference {
    fn from(value: db::voice_lines::Model) -> Self {
       Self {
            name: value.voice_name,
            location: value.voice_location.into()
        }
    }
}

impl From<db::characters::Model> for VoiceReference {
    fn from(value: db::characters::Model) -> Self {
        Self {
            name: value.voice_name,
            location: value.voice_location.into()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum VoiceDestination {
    Global,
    Game(String)
}

impl VoiceDestination {
    pub fn to_string_value(&self) -> String {
        match self {
            VoiceDestination::Global => "global".into(),
            VoiceDestination::Game(game_val) => game_val.clone()
        }
    }

    pub fn to_path(&self, conf: &TtsSystemConfig) -> PathBuf {
        match self {
            VoiceDestination::Global => {
                conf.global_voice()
            },
            VoiceDestination::Game(game_name) => {
                conf.game_voice(game_name)
            }
        }
    }
}

impl From<String> for VoiceDestination {
    fn from(value: String) -> Self {
        if value == "global" || value == "Global" {
            Self::Global
        } else {
            Self::Game(value)
        }
    }
}

/// A voice for TTS usage which is found on disk
#[derive(Debug, Clone)]
pub struct FsVoice {
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct FsVoiceData {
    pub reference: VoiceReference,
    pub dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct VoiceSample {
    pub emotion: BasicEmotion,
    /// If there is spoken text present, list what it is
    pub spoken_text: Option<String>,
    pub data: Vec<u8>
}

#[derive(Debug, Clone)]
pub struct FsVoiceSample {
    /// The emotion voiced by the sample.
    pub emotion: BasicEmotion,
    /// Optional reference to the txt file containing the spoken words in the given sample.
    pub spoken_text: Option<PathBuf>,
    /// The path of the sample.
    pub sample: PathBuf,
}

impl FsVoiceSample {
    /// Hard link this voice sample to the given directory, and use the given `name`
    /// as the reference.
    /// 
    /// Both directories are expected to be on the same filesystem.
    pub fn link_to_name(&self, dir: PathBuf, name: &str) -> eyre::Result<LinkedFsVoiceSample> {
        let sample_ext = self.sample.extension();
        let target_sample = dir.join(name).with_extension(sample_ext.unwrap_or("wav".as_ref()));
        std::fs::hard_link(&self.sample, &target_sample)?;
        
        let target_spoken = if let Some(spoken) = &self.spoken_text {
            let target_text_name = format!("{name}.reference.txt");
            let target_spoken = dir.join(target_text_name);
            
            std::fs::hard_link(spoken, &target_spoken)?;
            Some(target_spoken)
        } else {
            None
        };
        
        Ok(LinkedFsVoiceSample(FsVoiceSample {
            emotion: self.emotion,
            spoken_text: target_spoken,
            sample: target_sample,
        }))
    }
    
    /// Read the sample's data
    pub async fn data(&self) -> eyre::Result<Vec<u8>> {
        Ok(tokio::fs::read(&self.sample).await?)
    }
    
    /// If the sample has spoken text, recall what it was.
    pub async fn spoken_text(&self) -> eyre::Result<Option<String>> {
        let Some(path) = self.spoken_text_path() else {
            return Ok(None);
        };
        Ok(Some(tokio::fs::read_to_string(&path).await?))
    }
    
    /// The saved contents for this text.
    pub fn spoken_text_path(&self) -> Option<PathBuf> {
        let sample_dir = self.sample.with_extension("txt");
        
        if sample_dir.exists() {
            Some(sample_dir)
        } else {
            None
        }
    }
}

/// A hard-linked [FsVoiceSample] which will be deleted when dropped.
#[derive(Debug)]
pub struct LinkedFsVoiceSample(FsVoiceSample);

impl Drop for LinkedFsVoiceSample {
    fn drop(&mut self) {
        // We ignore the errors here, not great, but it's a best-effort cleanup.
        let _ = std::fs::remove_file(&self.0.sample);
        if let Some(spoken) = &self.0.spoken_text {
            let _ = std::fs::remove_file(spoken);
        }
    }
}

impl std::ops::Deref for LinkedFsVoiceSample {
    type Target = FsVoiceSample;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl FsVoiceData {
    /// Return all samples of the given emotion on disk.
    pub fn get_emotion_samples(&self, emotion: BasicEmotion) -> eyre::Result<Vec<FsVoiceSample>> {
        Ok(walkdir::WalkDir::new(&self.dir)
            .min_depth(1)
            .max_depth(2)
            .into_iter()
            .filter_entry(is_wav)
            .flatten()
            .filter(|d| emotion.matches_file(&d.file_name().to_string_lossy()))
            .map(|d| {
                let text = d.path().with_extension("txt");
                FsVoiceSample {
                    emotion,
                    spoken_text: text.exists().then_some(text),
                    sample: d.into_path(),
                }
            })
            .collect())
    }
    
    fn all_samples(&self) -> impl Iterator<Item=FsVoiceSample> {
        walkdir::WalkDir::new(&self.dir)
            .min_depth(1)
            .max_depth(2)
            .into_iter()
            .filter_entry(is_wav)
            .flatten()
            .flat_map(|d| {
                let text = d.path().with_extension("txt");
                let emotion = BasicEmotion::from_file_name(&d.file_name().to_string_lossy())?;
                Some(FsVoiceSample {
                    emotion,
                    spoken_text: text.exists().then_some(text),
                    sample: d.into_path(),
                })
            })
    }
    
    /// Select any random sample in the dataset.
    pub fn random_sample(&self) -> eyre::Result<FsVoiceSample> {
        self.all_samples()
            .choose(&mut rand::rng())
            .context("No sample available")
    }
    
    /// Try to find a random voice sample which adheres to the given predicate
    pub fn try_random_sample(&self, predicate: impl FnMut(&FsVoiceSample) -> bool) -> eyre::Result<FsVoiceSample> {
        self.all_samples()
            .filter(predicate)
            .choose(&mut rand::rng())
            .context("No sample available matching the predicate")
    }
    
    pub fn get_samples(&self) -> eyre::Result<HashMap<BasicEmotion, Vec<FsVoiceSample>>> {
        let mut output = HashMap::new();

        for out in self.all_samples() {
            let coll: &mut Vec<_> = output.entry(out.emotion).or_default();
            coll.push(out)
        }
        
        Ok(output)
    }

    /// Try and find a set of voice samples which match the given `emotion`.
    ///
    /// # Returns
    ///
    /// An iterator in the order of most-to-least matching order for the given `emotion`.
    pub fn try_emotion_sample(&self, emotion: BasicEmotion) -> eyre::Result<impl Iterator<Item=Vec<FsVoiceSample>> + use<>> {
        let mut samples = self.get_samples()?;

        Ok(emotion.to_preference_order()
            .into_iter()
            .flat_map(move |emotion| samples.remove(&emotion)))
    }
}

fn is_wav(d: &DirEntry) -> bool {
    d.file_type().is_file() && d.path().extension().map(|e| e.to_string_lossy() == "wav").unwrap_or_default()
}