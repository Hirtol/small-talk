use std::sync::Arc;
use st_http::config::SharedConfig;
use st_system::emotion::EmotionBackend;
use st_system::rvc_backends::RvcCoordinator;
use st_system::rvc_backends::seedvc::local::{LocalSeedHandle, LocalSeedVcConfig};
use st_system::tts_backends::alltalk::local::{LocalAllTalkConfig, LocalAllTalkHandle};
use st_system::tts_backends::echo_tts::EchoTtsHandle;
use st_system::tts_backends::indextts::IndexTtsHandle;
use st_system::tts_backends::TtsCoordinator;
use st_system::TtsSystem;
use crate::error;
use crate::session::StGameSessionFfi;

#[derive(uniffi::Object, Clone)]
pub struct StSystemFfi {
    pub config: Arc<st_http::config::Config>,
    pub system: Arc<TtsSystem>,
}

#[uniffi::export]
impl StSystemFfi {
    pub async fn start_game_session(&self, game_name: String) -> crate::Result<StGameSessionFfi> {
        let game_sess = self.system.get_or_start_session(&game_name).await?;
        Ok(
            StGameSessionFfi {
                system: self.clone(),
                handle: game_sess,
            }
        )
    }
}

pub(crate) fn create_tts_system(config: SharedConfig) -> eyre::Result<Arc<TtsSystem>> {
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
        .map(|cfg| IndexTtsHandle::new(cfg.clone()))
        .transpose()?;
    let echo = config
        .echo_tts
        .if_enabled()
        .map(|cfg| EchoTtsHandle::new(cfg.clone()))
        .transpose()?;

    let tts_backend = TtsCoordinator::new(xtts, index, echo, config.dirs.whisper_model.clone());

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