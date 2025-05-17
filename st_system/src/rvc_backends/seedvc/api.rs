use reqwest::{multipart, ClientBuilder};
use serde::{Deserialize, Serialize};
use url::Url;
use crate::audio::postprocessing::AudioData;
use crate::rvc_backends::{BackendRvcRequest};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedVcApiConfig {
    pub address: Url
}

pub struct SeedVcApi {
    config: SeedVcApiConfig,
    client: reqwest::Client,
}

impl SeedVcApi {
    pub fn new(config: SeedVcApiConfig) -> eyre::Result<Self> {
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
    pub async fn rvc(&self, request: BackendRvcRequest) -> eyre::Result<AudioData> {
        let bytes_to_send = bytemuck::allocation::try_cast_vec(request.audio.samples)
            .unwrap_or_else(|(_, vec)| bytemuck::cast_slice(&vec).to_vec());
        let form = multipart::Form::new()
            .part(
                "sound_samples",
                multipart::Part::bytes(bytes_to_send)
                    .file_name("sound_file.raw")
                    .mime_str("application/octet-stream")?,
            )
            .text("sample_rate", request.audio.sample_rate.to_string())
            .text("channels", request.audio.n_channels.to_string())
            .text("target_voice", request.target_voice.to_string_lossy().into_owned());

        // Make the POST request
        let response = self.client
            .post(self.url("/api/rvc")?)
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