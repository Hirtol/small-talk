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
use st_system::tts_backends::indextts::local::LocalIndexHandle;
use st_system::voice_manager::{VoiceDestination, VoiceManager, VoiceReference};
use crate::args::ClapTtsModel;

#[derive(clap::Args, Debug)]
pub struct ReassignCommand {
    /// The name of the game-session which contains the voice we want to change.
    pub game_name: String,
    /// The voice to change
    pub voice: String,
    /// The location, either 'global' or '{GAME_NAME}'
    pub voice_location: String,
    /// Name of the new voice
    #[clap(long)]
    pub target_voice: String,
    /// The location, either 'global' or '{GAME_NAME}'
    #[clap(long)]
    pub target_location: String,
    /// The TTS Model to use for the re-generation
    #[clap(long)]
    pub model: ClapTtsModel,
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

        let mut voice_lines = lines_to_redo.into_iter().map(|line| {
            VoiceLine {
                line,
                person: TtsVoice::ForceVoice(new_voice.clone()),
                model: self.model.into(),
                force_generate: true,
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
        }).collect_vec();

        while let Some(line) = voice_lines.pop() {
            if let Err(_) = game_sess.request_tts(line.clone()).await {
                // Retry failed ones
                tracing::debug!("Pushing {line:?} onto retry queue");
                voice_lines.push(line)
            }
        }

        Ok(())
    }
}

fn create_tts_system(config: SharedConfig) -> eyre::Result<Arc<TtsSystem>> {
    let xtts = config
        .xtts
        .if_enabled()
        .map(|xtts| {
            let all_talk_cfg = LocalAllTalkConfig {
                instance_path: xtts.local_all_talk.clone(),
                timeout: xtts.timeout,
                api: xtts.alltalk_cfg.clone(),
            };

            LocalAllTalkHandle::new(all_talk_cfg)
        })
        .transpose()?;
    let index = config
        .index_tts
        .if_enabled()
        .map(|cfg| LocalIndexHandle::new(cfg.clone()))
        .transpose()?;

    let tts_backend = TtsCoordinator::new(xtts, index, config.dirs.whisper_model.clone());

    let mut seedvc_cfg = config.seed_vc.if_enabled().map(|seed_vc| LocalSeedVcConfig {
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
