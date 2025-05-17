use reqwest::{multipart, ClientBuilder};
use serde::{Deserialize, Serialize};
use url::Url;
use crate::audio::postprocessing::AudioData;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexTtsApiConfig {
    pub address: Url
}

#[derive(Debug, Clone)]
pub struct IndexTtsAPI {
    pub config: IndexTtsApiConfig,
    client: reqwest::Client,
}

impl IndexTtsAPI {
    pub fn new(config: IndexTtsApiConfig) -> eyre::Result<Self> {
        let client = ClientBuilder::default().build()?;

        Ok(Self {
            config,
            client,
        })
    }

    /// Check whether this SeedVc instance is ready.
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
    pub async fn tts(&self, request: IndexTtsRequest) -> eyre::Result<AudioData> {
        let form = multipart::Form::new()
            .part(
                "audio_file",
                multipart::Part::bytes(request.wav_file_bytes)
                    .file_name("sample.wav")
                    .mime_str("application/octet-stream")?,
            )
            .text("text", request.text);

        let response = self.client
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

#[derive(Debug)]
pub struct IndexTtsRequest {
    pub text: String,
    pub wav_file_bytes: Vec<u8>
}

#[cfg(test)]
mod tests {
    use crate::tts_backends::indextts::api::{IndexTtsAPI, IndexTtsApiConfig, IndexTtsRequest};
    use crate::tts_backends::indextts::IndexTts;

    #[tokio::test]
    async fn test_index_api() -> eyre::Result<()> {
        let api = IndexTts::new(IndexTtsApiConfig {
            address: "http://localhost:11996".try_into()?,
        }).await?;

        let wav = std::fs::read(r"G:\TTS\small-talk-data\game_data\Pathfinder-WOTR\voices\Regill\Neutral_13.wav")?;
        let out = api.api.tts(IndexTtsRequest { text: "Hoe verloopt de solicitatie procedure? Ik ben een ‘normale’ baan gewend de afgelopen tijd kwa soliciteren, maar weet dus niet hoe dat verschilt ten opzichten van een traineeship.".into(), wav_file_bytes: wav }).await?;

        out.write_to_wav_file("regil.wav".as_ref())?;

        Ok(())
    }
}