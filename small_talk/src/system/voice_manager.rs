use std::collections::HashMap;
use crate::config::{Config, SharedConfig};
use itertools::Itertools;
use small_talk_ml::emotion_classifier::BasicEmotion;
use std::path::PathBuf;
use path_abs::PathOps;
use walkdir::DirEntry;

#[derive(Debug, Clone)]
pub struct VoiceManager {
    conf: SharedConfig,
}

impl VoiceManager {
    pub fn new(conf: SharedConfig) -> Self {
        Self { conf }
    }

    pub fn get_voice(&self, dest: VoiceDestination, voice: &str) -> eyre::Result<FsVoiceData> {
        let path = dest.to_path(&self.conf).join(voice);
        if path.exists() {
            Ok(FsVoiceData {
                dir: path,
                name: voice.into(),
            })    
        } else {
            Err(eyre::eyre!("Voice does not exist"))
        }
    }

    /// Return all applicable voices (including game specific and global) for the given game.
    pub fn get_voices(&self, game_name: &str) -> Vec<FsVoiceData> {
        let mut result = self.get_global_voices();
        result.extend(self.get_game_voices(game_name));
        result
    }

    pub fn get_game_voices(&self, game_name: &str) -> Vec<FsVoiceData> {
        walkdir::WalkDir::new(super::dirs::game_voice(&self.conf, game_name))
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_entry(|d| d.file_type().is_dir())
            .flatten()
            .map(|d| FsVoiceData {
                name: d.file_name().to_string_lossy().into_owned(),
                dir: d.into_path(),
            })
            .collect_vec()
    }

    pub fn get_global_voices(&self) -> Vec<FsVoiceData> {
        walkdir::WalkDir::new(super::dirs::global_voice(&self.conf))
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_entry(|d| d.file_type().is_dir())
            .flatten()
            .map(|d| FsVoiceData {
                name: d.file_name().to_string_lossy().into_owned(),
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
            if let Ok(voice_data) = self.get_voice(dest.clone(), voice_name) {
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

#[derive(Clone, Debug)]
pub enum VoiceDestination<'a> {
    Global,
    Game(&'a str)
}

impl VoiceDestination<'_> {
    pub fn to_path(&self, conf: &Config) -> PathBuf {
        match self {
            VoiceDestination::Global => {
                super::dirs::global_voice(conf)
            },
            VoiceDestination::Game(game_name) => {
                super::dirs::game_voice(conf, game_name)
            }
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
    pub name: String,
    pub dir: PathBuf,
}

#[derive(Debug, Clone)]
pub struct FsVoiceSample {
    pub emotion: BasicEmotion,
    pub sample: PathBuf,
}

#[derive(Debug, Clone)]
pub struct VoiceSample {
    pub emotion: BasicEmotion,
    /// If there is spoken text present, list what it is
    pub spoken_text: Option<String>,
    pub data: Vec<u8>
}

impl FsVoiceSample {
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
            .map(|d| FsVoiceSample {
                emotion,
                sample: d.into_path(),
            })
            .collect())
    }
    
    pub fn get_samples(&self) -> eyre::Result<HashMap<BasicEmotion, Vec<FsVoiceSample>>> {
        let mut output = HashMap::new();
        let itr = walkdir::WalkDir::new(&self.dir)
            .min_depth(1)
            .max_depth(2)
            .into_iter()
            .filter_entry(is_wav)
            .flatten()
            .flat_map(|d| {
                let emotion = BasicEmotion::from_file_name(&d.file_name().to_string_lossy())?;
                Some((emotion, FsVoiceSample {
                    emotion,
                    sample: d.into_path(),
                }))
            });
        
        for (emotion, out) in itr {
            let coll: &mut Vec<_> = output.entry(emotion).or_default();
            coll.push(out)
        }
        
        Ok(output)
    }
}

fn is_wav(d: &DirEntry) -> bool {
    d.file_type().is_file() && d.path().extension().map(|e| e.to_string_lossy() == "wav").unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use crate::config::{Config, SharedConfig};
    use crate::system::voice_manager::{VoiceDestination, VoiceManager};

    #[tokio::test]
    pub async fn test_name() {
        let mut conf = crate::config::initialise_config().unwrap();
        let conf = SharedConfig::new(conf);
        
        let mut man = VoiceManager::new(conf);
        
        let t = man.get_voice(VoiceDestination::Global, "Baphomet").unwrap();
        println!("{:#?}", t.get_samples().unwrap());
        println!("T: {:#?}", man.get_global_voices())
    }
}