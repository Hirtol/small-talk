use crate::audio::audio_data::AudioData;
use reqwest::{multipart, ClientBuilder};
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Formatter};
use url::Url;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoTtsApiConfig {
    pub address: Url,
}

#[derive(Debug, Clone)]
pub struct EchoTtsAPI {
    pub config: EchoTtsApiConfig,
    client: reqwest::Client,
}

impl EchoTtsAPI {
    pub fn new(config: EchoTtsApiConfig) -> eyre::Result<Self> {
        let client = ClientBuilder::default().build()?;

        Ok(Self { config, client })
    }

    #[tracing::instrument(skip(self))]
    pub async fn ready(&self) -> eyre::Result<bool> {
        if let Ok(body) = self.client.get(self.url("/api/ready")?).send().await {
            Ok(body.text().await?.contains("true"))
        } else {
            Ok(false)
        }
    }

    /// Send a request for a generation to the given API.
    ///
    /// Returns the output path.
    #[tracing::instrument(skip(self))]
    pub async fn tts(&self, request: EchoTtsRequest) -> eyre::Result<AudioData> {
        let form = multipart::Form::new()
            .part(
                "audio_file",
                multipart::Part::bytes(request.wav_file_bytes)
                    .file_name("sample.wav")
                    .mime_str("application/octet-stream")?,
            )
            .text("text", request.text)
            .text("num_steps", request.num_steps.unwrap_or(40).to_string())
            .text("sequence_length", request.sequence_length.unwrap_or(640).to_string());

        let response = self
            .client
            .post(self.url("/api/tts_wav")?)
            .multipart(form)
            .send()
            .await?;
        response.error_for_status_ref()?;

        let content = response.bytes().await?;
        let cursor = std::io::Cursor::new(content);
        let mut wav = wavers::Wav::new(Box::new(cursor))?;

        Ok(AudioData::new(&mut wav)?)
    }

    fn url(&self, path: &str) -> eyre::Result<Url> {
        Ok(self.config.address.join(path)?)
    }
}

pub struct EchoTtsRequest {
    pub text: String,
    pub num_steps: Option<usize>,
    pub sequence_length: Option<usize>,
    pub wav_file_bytes: Vec<u8>,
}

impl Debug for EchoTtsRequest {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EchoTtsRequest")
            .field("text", &self.text)
            .field("num_steps", &self.num_steps)
            .field("sequence_length", &self.sequence_length)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::tts_backends::echo_tts::{
        api::{EchoTtsApiConfig, EchoTtsRequest},
        EchoTts,
    };

    #[tokio::test]
    async fn test_index_api() -> eyre::Result<()> {
        let api = EchoTts::new(EchoTtsApiConfig {
            address: "http://localhost:11997".try_into()?,
        })
        .await?;

        let wav = std::fs::read(r"G:\TTS\small-talk-data\game_data\Pathfinder-WOTR\voices\Regill\Neutral_13.wav")?;
        let out = api.api.tts(EchoTtsRequest {
            text: "Hoe verloopt de solicitatie procedure? Ik ben een ‘normale’ baan gewend de afgelopen tijd kwa soliciteren, maar weet dus niet hoe dat verschilt ten opzichten van een traineeship.".into(),
            num_steps: None,
            sequence_length: None,
            wav_file_bytes: wav,
        }).await?;

        out.write_to_wav_file("regil.wav".as_ref())?;

        Ok(())
    }
}
