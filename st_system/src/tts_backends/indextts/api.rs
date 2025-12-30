use reqwest::{multipart, ClientBuilder};
use serde::{Deserialize, Serialize};
use url::Url;
use st_audio::audio_data::AudioData;
use crate::timeout::DroppableState;
use crate::tts_backends::generic_backend::TtsApi;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexTtsApiConfig {
    pub address: Url
}

#[derive(Debug)]
pub struct IndexTtsRequest {
    pub text: String,
    pub wav_file_bytes: Vec<u8>
}

#[derive(Debug, Clone)]
pub struct IndexTtsAPI {
    pub config: IndexTtsApiConfig,
    client: reqwest::Client,
}

impl TtsApi for IndexTtsAPI {
    type Request = IndexTtsRequest;

    async fn ready(&self) -> eyre::Result<bool> {
        if let Ok(body) = self.client.get(self.url("/api/ready")?).send().await {
            Ok(body.text().await?.contains("true"))
        } else {
            Ok(false)
        }
    }

    #[tracing::instrument(skip(self))]
    async fn tts(&self, request: Self::Request) -> eyre::Result<AudioData> {
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
}

impl DroppableState for IndexTtsAPI {
    type Context = IndexTtsApiConfig;

    async fn initialise_state(context: &Self::Context) -> eyre::Result<Self> {
        Self::new(context.clone())
    }

    async fn on_kill(&mut self) -> eyre::Result<()> {
        Ok(())
    }
}

impl IndexTtsAPI {
    pub fn new(config: IndexTtsApiConfig) -> eyre::Result<Self> {
        let client = ClientBuilder::default().build()?;

        Ok(Self {
            config,
            client,
        })
    }

    fn url(&self, path: &str) -> eyre::Result<Url> {
        Ok(self.config.address.join(path)?)
    }
}