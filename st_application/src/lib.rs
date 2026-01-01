//! Root crate for shared application-level code such as configs and system set up

use crate::config::SmallTalkConfig;
use st_ml::stt::WhisperTranscribe;
use st_system::{
    emotion::{EmotionBackend, EmotionCoordinator}, rvc_backends::{
        seedvc::local::{LocalSeedHandle, LocalSeedVcConfig},
        RvcCoordinator,
    },
    tts_backends::{echo_tts::EchoTtsHandle, indextts::IndexTtsHandle, TtsCoordinator},
    TtsSystem,
    TtsSystemHandle,
};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

pub mod config;

#[derive(Clone)]
pub struct SmallTalkApplication {
    pub tts_system: TtsSystemHandle,
    pub root_cancel: CancellationToken,
}

impl SmallTalkApplication {
    #[tracing::instrument(name = "Create SmallTalkApplication", skip(config))]
    pub async fn new(config: &SmallTalkConfig) -> eyre::Result<Self> {
        let root_cancel = CancellationToken::new();
        let index = config
            .index_tts
            .if_enabled()
            .map(|cfg| IndexTtsHandle::new(cfg.clone(), root_cancel.clone()))
            .transpose()?;

        let echo = config
            .echo_tts
            .if_enabled()
            .map(|cfg| EchoTtsHandle::new(cfg.clone(), root_cancel.clone()))
            .transpose()?;

        let whisper = if let Some(whisper_path) = config.dirs.whisper_model.clone() {
            let cpu_threads = std::thread::available_parallelism()?.get() / 2;
            Some(Arc::new(Mutex::new(WhisperTranscribe::new(
                whisper_path,
                cpu_threads as u16,
            )?)))
        } else {
            None
        };

        let tts_backend = TtsCoordinator::new(index, echo, whisper);

        let seedvc_cfg = config.seed_vc.if_enabled().map(|seed_vc| LocalSeedVcConfig {
            instance_path: seed_vc.local_path.clone(),
            timeout: seed_vc.timeout,
            api: seed_vc.config.clone(),
            high_quality: false,
        });
        let seedvc = seedvc_cfg
            .clone()
            .map(|seedvc_cfg| LocalSeedHandle::new(seedvc_cfg.clone(), root_cancel.clone()))
            .transpose()?;
        let seedvc_hq = seedvc_cfg
            .map(|mut seedvc_cfg| {
                seedvc_cfg.high_quality = true;
                LocalSeedHandle::new(seedvc_cfg, root_cancel.clone())
            })
            .transpose()?;
        let rvc_backend = RvcCoordinator::new(seedvc, seedvc_hq);

        let emotion_backend = match (
            &config.dirs.emotion_classifier_model,
            &config.dirs.bert_embeddings_model,
        ) {
            (Some(classifier), Some(bert)) => Some(EmotionBackend::new(classifier, bert)?),
            _ => None,
        };
        let emotion_backend = EmotionCoordinator::new(emotion_backend);

        let handle = Arc::new(TtsSystem::new(
            config.dirs.clone(),
            tts_backend,
            rvc_backend,
            emotion_backend,
            root_cancel.clone(),
        ));

        let result = SmallTalkApplication {
            tts_system: handle,
            root_cancel,
        };

        Ok(result)
    }

    pub async fn shutdown(&self) -> eyre::Result<()> {
        self.tts_system.shutdown().await?;
        self.root_cancel.cancel();
        Ok(())
    }
}
