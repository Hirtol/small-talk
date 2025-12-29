use crate::{records::VoiceReference, system::StSystemFfi};
use st_system::session::GameSessionHandle;

#[derive(uniffi::Object, Clone)]
pub struct StGameSessionFfi {
    pub system: StSystemFfi,
    pub handle: GameSessionHandle,
}

#[uniffi::export]
impl StGameSessionFfi {
    pub async fn available_voices(&self) -> crate::Result<Vec<VoiceReference>> {
        Ok(self
            .handle
            .available_voices()
            .await?
            .into_iter()
            .map(|i| i.reference.into())
            .collect())
    }
}
