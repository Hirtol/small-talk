use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use serde::{Deserialize, Serialize};
use reqwest::{ClientBuilder, Url};
use serde::de::DeserializeOwned;
use crate::system::tts_backends::alltalk::AllTalkConfig;

pub struct AllTalkApi {
    config: AllTalkConfig,
    client: reqwest::Client,
}

impl AllTalkApi {
    pub fn new(config: AllTalkConfig) -> eyre::Result<Self> {
        let client = ClientBuilder::default().build()?;

        Ok(Self {
            config,
            client,
        })
    }

    /// Check whether this AllTalk instance is ready.
    pub async fn ready(&self) -> eyre::Result<bool> {
        let body = self.client.get(self.url("/api/ready")?).send().await?;
        Ok(body.text().await? == "Ready")
    }
    
    /// Force AllTalk to reload from disk, namely used when adding new voices.
    pub async fn reload_settings(&self) -> eyre::Result<()> {
        self.get("/api/reload_config").await
    }

    /// Retrieve the current settings from AllTalk
    pub async fn current_settings(&self) -> eyre::Result<AllTalkSettings> {
        self.get("/api/currentsettings").await
    }

    /// Retrieve the voices which AllTalk currently has available
    pub async fn voices(&self) -> eyre::Result<Voices> {
        self.get("/api/voices").await
    }

    /// Send a request for a generation to the given API.
    /// 
    /// Returns the output path.
    pub async fn tts_request(&self, request: TtsRequest) -> eyre::Result<TtsResponse> {
        let response = self.client
            .post(self.url("/api/tts-generate")?)
            .form(&request)
            .send()
            .await?;

        Ok(response.json().await?)
    }

    async fn get<T: DeserializeOwned>(&self, path: &str) -> eyre::Result<T> {
        Ok(self.client.get(self.url(path)?).send().await?.json().await?)
    }

    fn url(&self, path: &str) -> eyre::Result<Url> {
        Ok(self.config.address.join(path)?)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TtsRequest {
    pub text_input: String,
    pub text_filtering: Option<TextFiltering>,
    pub character_voice_gen: String,
    pub rvccharacter_voice_gen: Option<String>,
    pub rvccharacter_pitch: Option<i32>,
    pub narrator_enabled: Option<bool>,
    pub narrator_voice_gen: Option<String>,
    pub rvcnarrator_voice_gen: Option<String>,
    pub rvcnarrator_pitch: Option<i32>,
    pub text_not_inside: Option<String>,
    pub language: String,
    pub output_file_name: String,
    pub output_file_timestamp: Option<bool>,
    pub autoplay: Option<bool>,
    pub autoplay_volume: Option<f32>,
    pub speed: Option<f32>,
    pub pitch: Option<i32>,
    pub temperature: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

impl Default for TtsRequest {
    fn default() -> Self {
        Self {
            text_input: "Hello World".to_string(),
            text_filtering: None,
            character_voice_gen: "male_01.wav".to_string(),
            rvccharacter_voice_gen: None,
            rvccharacter_pitch: None,
            narrator_enabled: None,
            narrator_voice_gen: None,
            rvcnarrator_voice_gen: None,
            rvcnarrator_pitch: None,
            text_not_inside: None,
            language: "en".to_string(),
            output_file_name: "generic_output".to_string(),
            output_file_timestamp: None,
            autoplay: None,
            autoplay_volume: None,
            speed: None,
            pitch: None,
            temperature: None,
            repetition_penalty: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum TextFiltering {
    None,
    Standard,
    Html,
}

impl Display for TextFiltering {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TextFiltering::None => write!(f, "none"),
            TextFiltering::Standard => write!(f, "standard"),
            TextFiltering::Html => write!(f, "html"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TtsResponse {
    pub status: String,             // Consider using an enum for predefined statuses
    pub output_file_path: String,
    pub output_file_url: String,
    pub output_cache_url: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TTSModel {
    pub name: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AllTalkSettings {
    pub engines_available: Vec<String>,
    pub current_engine_loaded: String,
    pub models_available: Vec<TTSModel>,
    pub current_model_loaded: String,
    pub manufacturer_name: String,
    pub audio_format: String,
    pub deepspeed_capable: bool,
    pub deepspeed_available: bool,
    pub deepspeed_enabled: bool,
    pub generationspeed_capable: bool,
    pub generationspeed_set: f64,
    pub lowvram_capable: bool,
    pub lowvram_enabled: bool,
    pub pitch_capable: bool,
    pub pitch_set: f64,
    pub repetitionpenalty_capable: bool,
    pub repetitionpenalty_set: f64,
    pub streaming_capable: bool,
    pub temperature_capable: bool,
    pub temperature_set: f64,
    pub ttsengines_installed: bool,
    pub languages_capable: bool,
    pub multivoice_capable: bool,
    pub multimodel_capable: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Voices {
    pub status: String,
    pub voices: Vec<String>,
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use reqwest::Url;
    use crate::system::tts_backends::alltalk::AllTalkConfig;
    use crate::system::tts_backends::alltalk::api::{AllTalkApi, TtsRequest};

    #[tokio::test]
    pub async fn test_live() {
        let api = AllTalkApi::new(AllTalkConfig::new(Url::from_str("http://localhost:7851").unwrap())).unwrap();
        
        assert!(api.ready().await.unwrap());
        
        println!("Settings: {:#?}", api.current_settings().await.unwrap());
        println!("Voices: {:#?}", api.voices().await.unwrap());
        
        println!("Gen: {:#?}", api.tts_request(TtsRequest::default()).await.unwrap())
    }
}