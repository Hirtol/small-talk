use reqwest::{multipart, ClientBuilder};
use serde::{Deserialize, Serialize};
use url::Url;
use crate::system::rvc_backends::{BackendRvcRequest};

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
    pub async fn rvc(&self, request: BackendRvcRequest) -> eyre::Result<wavers::Wav<f32>> {
        let form = multipart::Form::new()
            .part(
                "sound_samples",
                multipart::Part::bytes(bytemuck::allocation::cast_vec(request.samples))
                    .file_name("sound_file.raw")
                    .mime_str("application/octet-stream")?,
            )
            .text("sample_rate", request.sample_rate.to_string())
            .text("channels", request.channels.to_string())
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
        let wav = wavers::Wav::new(Box::new(cursor))?;

        Ok(wav)
    }

    fn url(&self, path: &str) -> eyre::Result<Url> {
        Ok(self.config.address.join(path)?)
    }
}