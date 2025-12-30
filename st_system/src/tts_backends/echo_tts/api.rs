use st_audio::audio_data::AudioData;
use reqwest::{multipart, ClientBuilder};
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Formatter};
use rand::Rng;
use url::Url;
use crate::timeout::DroppableState;
use crate::tts_backends::generic_backend::TtsApi;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EchoTtsApiConfig {
    pub address: Url,
}

#[derive(Debug, Clone)]
pub struct EchoTtsAPI {
    pub config: EchoTtsApiConfig,
    client: reqwest::Client,
}

impl TtsApi for EchoTtsAPI {
    type Request = EchoTtsRequest;

    #[tracing::instrument(skip(self))]
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
            .text("text", request.text)
            .text("num_steps", request.num_steps.unwrap_or(30).to_string())
            .text("sequence_length", request.sequence_length.unwrap_or(640).to_string())
            .text("rng_seed", rand::rng().random::<u32>().to_string());

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
}

impl DroppableState for EchoTtsAPI {
    type Context = EchoTtsApiConfig;

    async fn initialise_state(context: &Self::Context) -> eyre::Result<Self> {
        Self::new(context.clone())
    }

    async fn on_kill(&mut self) -> eyre::Result<()> {
        Ok(())
    }
}

impl EchoTtsAPI {
    pub fn new(config: EchoTtsApiConfig) -> eyre::Result<Self> {
        let client = ClientBuilder::default().build()?;

        Ok(Self { config, client })
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