use std::sync::Arc;
use std::time::Duration;
use eyre::ContextCompat;
use itertools::Itertools;
use path_abs::PathInfo;
use rayon::prelude::*;
use tokio::time::sleep;
use st_http::config::SharedConfig;
use st_system::emotion::EmotionBackend;
use st_system::rvc_backends::RvcCoordinator;
use st_system::rvc_backends::seedvc::local::{LocalSeedHandle, LocalSeedVcConfig};
use st_system::tts_backends::alltalk::local::{LocalAllTalkConfig, LocalAllTalkHandle};
use st_system::tts_backends::TtsCoordinator;
use st_system::{PostProcessing, RvcModel, RvcOptions, TtsModel, TtsSystem, TtsVoice, VoiceLine};
use st_system::voice_manager::{VoiceDestination, VoiceManager, VoiceReference};

#[derive(clap::Args, Debug)]
pub struct ReassignCommand {
    /// The name of the game-session which contains the voice we want to change.
    game_name: String,
    /// The voice to change
    voice: String,
    /// The location, either 'global' or '{GAME_NAME}'
    voice_location: String,
    /// Name of the new voice
    #[clap(long)]
    target_voice: String,
    /// The location, either 'global' or '{GAME_NAME}'
    #[clap(long)]
    target_location: String,
}

impl ReassignCommand {
    #[tracing::instrument(skip_all, fields(self.sample_path))]
    pub async fn run(self, config: SharedConfig) -> eyre::Result<()> {
        let tts_sys = create_tts_system(config)?;
        let game_sess = tts_sys.get_or_start_session(&self.game_name).await?;

        let new_voice = VoiceReference {
            name: self.target_voice,
            location: self.target_location.into(),
        };
        let source_voice = VoiceReference {
            name: self.voice,
            location: self.voice_location.into(),
        };
        let assigned_voices = game_sess.character_voices().await?;
        let lines_to_redo = game_sess.voice_lines(&source_voice).await?;

        for (character, voice) in assigned_voices {
            if voice != source_voice {
                continue;
            }

            tracing::info!(?character, old_voice=?voice, ?new_voice, "Reassigned character voice");

            game_sess.force_character_voice(character, new_voice.clone()).await?;
        }

        tracing::info!(todo=lines_to_redo.len(), "Regenerating lines");

        let voice_lines = lines_to_redo.into_iter().map(|line| {
            VoiceLine {
                line,
                person: TtsVoice::ForceVoice(new_voice.clone()),
                model: TtsModel::Xtts,
                force_generate: false,
                post: Some(PostProcessing {
                    verify_percentage: None,
                    trim_silence: true,
                    normalise: true,
                    rvc: Some(RvcOptions {
                        model: RvcModel::SeedVc,
                        high_quality: true,
                    }),
                }),
            }
        });
        for line in voice_lines {
            // Ignore errors
            let _ = game_sess.request_tts(line).await?;
        }
        
        Ok(())
    }
}

fn create_tts_system(config: SharedConfig) -> eyre::Result<Arc<TtsSystem>> {
    let xtts = config
        .xtts
        .as_ref()
        .map(|xtts| {
            let all_talk_cfg = LocalAllTalkConfig {
                instance_path: xtts.local_all_talk.clone(),
                timeout: xtts.timeout,
                api: xtts.alltalk_cfg.clone(),
            };

            LocalAllTalkHandle::new(all_talk_cfg)
        })
        .transpose()?;

    let tts_backend = TtsCoordinator::new(xtts, config.dirs.whisper_model.clone());

    let mut seedvc_cfg = config.seed_vc.as_ref().map(|seed_vc| LocalSeedVcConfig {
        instance_path: seed_vc.local_path.clone(),
        timeout: seed_vc.timeout,
        api: seed_vc.config.clone(),
        high_quality: false,
    });
    let seedvc = seedvc_cfg
        .clone()
        .map(|seedvc_cfg| LocalSeedHandle::new(seedvc_cfg.clone()))
        .transpose()?;
    let seedvc_hq = seedvc_cfg
        .map(|mut seedvc_cfg| {
            seedvc_cfg.high_quality = true;
            LocalSeedHandle::new(seedvc_cfg)
        })
        .transpose()?;
    let rvc_backend = RvcCoordinator::new(seedvc, seedvc_hq);

    let emotion_backend = EmotionBackend::new(&config.dirs)?;

    let handle = Arc::new(TtsSystem::new(
        config.dirs.clone(),
        tts_backend,
        rvc_backend,
        emotion_backend,
    ));

    Ok(handle)
}
