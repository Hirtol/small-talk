use std::sync::Arc;
use st_application::config::SharedConfig;
use st_application::SmallTalkApplication;
use st_system::emotion::EmotionBackend;
use st_system::rvc_backends::RvcCoordinator;
use st_system::rvc_backends::seedvc::local::{LocalSeedHandle, LocalSeedVcConfig};
use st_system::tts_backends::echo_tts::EchoTtsHandle;
use st_system::tts_backends::indextts::IndexTtsHandle;
use st_system::tts_backends::TtsCoordinator;
use st_system::TtsSystem;
use crate::error;
use crate::session::StGameSessionFfi;

#[derive(uniffi::Object, Clone)]
pub struct StSystemFfi {
    pub config: SharedConfig,
    pub system: SmallTalkApplication,
}

#[uniffi::export(async_runtime="tokio")]
impl StSystemFfi {
    pub async fn start_game_session(&self, game_name: String) -> crate::Result<StGameSessionFfi> {
        tracing::info!("Starting start_game_session");
        let game_sess = self.system.tts_system.get_or_start_session(&game_name).await?;
        tracing::info!("Finishing start_game_session");
        Ok(
            StGameSessionFfi {
                system: self.clone(),
                handle: game_sess,
            }
        )
    }

    pub async fn shutdown(&self) -> crate::Result<()> {
        Ok(self.system.shutdown().await?)
    }
}