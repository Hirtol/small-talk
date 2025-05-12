use std::sync::Arc;
use rayon::prelude::*;
use st_http::config::SharedConfig;
use st_system::emotion::EmotionBackend;
use st_system::rvc_backends::RvcCoordinator;
use st_system::rvc_backends::seedvc::local::{LocalSeedHandle, LocalSeedVcConfig};
use st_system::tts_backends::alltalk::local::{LocalAllTalkConfig, LocalAllTalkHandle};
use st_system::tts_backends::TtsCoordinator;
use st_system::{PostProcessing, RvcModel, RvcOptions, TtsModel, TtsSystem, TtsVoice, VoiceLine};
use st_system::voice_manager::{VoiceDestination, VoiceManager, VoiceReference};

#[derive(clap::Args, Debug)]
pub struct MigrateCommand {
    /// The name of the game-session which contains the voice we want to change.
    game_name: String,
}

impl MigrateCommand {
    #[tracing::instrument(skip_all, fields(self.sample_path))]
    pub async fn run(self, config: SharedConfig) -> eyre::Result<()> {
        let tts_sys = create_tts_system(config)?;
        let game_sess = tts_sys.get_or_start_session(&self.game_name).await?;
        // game_sess.migrate_config_to_db().await?;
        Ok(())
    }
}

// pub fn migrate_config_to_db(&self) -> impl Future<Output = eyre::Result<()>> + Send {
//         // Migrate characters
//         use sea_orm::*;
//         tracing::info!("Migrating characters");
//         async {
//             for (character, voice) in self.game_tts.data.game_data.character_map.pin_owned().iter() {
//                 use st_db::entity::*;
//                 let write = self.game_tts.data.game_db.writer().begin().await?;
//                 let gender = if self.game_tts.data.game_data.male_voices.contains(voice) {
//                     Gender::Male
//                 } else if self.game_tts.data.game_data.female_voices.contains(voice) {
//                     Gender::Female
//                 } else {
//                     tracing::warn!("Assuming male gender for: {character}");
//                     Gender::Male
//                 };
//
//                 let to_add = characters::ActiveModel {
//                     id: Default::default(),
//                     character_name: character.clone().into_active_value(),
//                     character_gender: gender.to_db().to_value().into_active_value(),
//                     voice_name: voice.name.clone().into_active_value(),
//                     voice_location: match &voice.location {
//                         VoiceDestination::Global => "global".to_string().into_active_value(),
//                         VoiceDestination::Game(game) => game.clone().into_active_value(),
//                     },
//                 };
//                 let character_model = to_add.clone().insert(&write).await?;
//                 // tracing::debug!(?character_model, "Inserted character {character}");
//
//                 // Get lines
//                 let line_cache = self.game_tts.data.line_cache.lock().await;
//                 if let Some(lines) = line_cache.voice_to_line.get(voice) {
//                     for (line, location) in lines.iter() {
//                         // First add the dialogue
//                         let dialogue_am = dialogue::ActiveModel {
//                             id: Default::default(),
//                             character_id: character_model.id.into_active_value(),
//                             dialogue_text: line.clone().into_active_value(),
//                         };
//                         // tracing::debug!(?dialogue_am, "Inserting new dialogue line");
//                         let dialogue_model = dialogue_am.insert(&write).await?;
//
//                         // Now add the actual voice line
//                         let voice_am = db::voice_lines::ActiveModel {
//                             id: Default::default(),
//                             dialogue_text: line.clone().into_active_value(),
//                             voice_name: to_add.voice_name.clone(),
//                             voice_location: to_add.voice_location.clone(),
//                             file_name: location.clone().into_active_value(),
//                         };
//                         let model = voice_am.insert(&write).await?;
//                         // tracing::debug!(?model, "Inserted voice line {line} with ID");
//                     }
//                 }
//
//                 write.commit().await?;
//             }
//
//             Ok(())
//         }
//     }

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
