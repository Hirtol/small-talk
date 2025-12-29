use crate::{
    records::{CharacterVoice, PlaybackVoiceLine, TtsResponse, VoiceLine, VoiceReference},
    system::StSystemFfi,
};
use st_system::session::GameSessionHandle;
use std::collections::HashMap;

#[derive(uniffi::Object, Clone)]
pub struct StGameSessionFfi {
    pub system: StSystemFfi,
    pub handle: GameSessionHandle,
}

#[uniffi::export(async_runtime="tokio")]
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

    pub async fn session_characters(&self) -> crate::Result<HashMap<CharacterVoice, VoiceReference>> {
        let output = self.handle.character_voices().await?;

        Ok(output.into_iter().map(|(k, v)| (k.into(), v.into())).collect())
    }

    pub async fn set_session_characters(
        &self,
        characters: HashMap<CharacterVoice, VoiceReference>,
    ) -> crate::Result<()> {
        for (character, voice) in characters {
            self.handle
                .force_character_voice(character.into(), voice.into())
                .await?;
        }
        Ok(())
    }

    pub async fn tts_playback_start(&self, requests: Vec<PlaybackVoiceLine>) -> crate::Result<()> {
        self.handle
            .playback
            .start(requests.into_iter().map(|i| i.into()).collect())
            .await?;

        Ok(())
    }

    pub async fn tts_playback_speed(&self, speed: f64) -> crate::Result<()> {
        self.handle.playback.set_speed(speed).await?;

        Ok(())
    }

    pub async fn tts_playback_stop(&self) -> crate::Result<()> {
        self.handle.playback.stop().await?;
        Ok(())
    }

    pub async fn tts_request(&self, request: VoiceLine) -> crate::Result<TtsResponse> {
        let out = self.handle.request_tts(request.into()).await?;

        Ok(st_system::TtsResponse::clone(&out).into())
    }

    pub async fn tts_queue(&self, requests: Vec<VoiceLine>) -> crate::Result<()> {
        self.handle
            .add_all_to_queue(requests.into_iter().map(|i| i.into()).collect())
            .await?;

        Ok(())
    }
}
