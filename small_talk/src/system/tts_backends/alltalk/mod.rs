use std::net::SocketAddr;
use std::time::Duration;
use reqwest::{ClientBuilder, Url};
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use api::{AllTalkApi, AllTalkSettings};

pub mod api;

pub struct AllTalkConfig {
    address: Url,
}

impl AllTalkConfig {
    pub fn new(address: Url) -> Self {
        Self {
            address,
        }
    }
}

pub struct AllTalkTTS {
    api: AllTalkApi,
    all_talk: AllTalkSettings,
}

impl AllTalkTTS {
    pub async fn new(config: AllTalkConfig) -> eyre::Result<Self> {
        let api_client = AllTalkApi::new(config)?;
        
        // Wait for it to be ready
        tokio::time::timeout(Duration::from_secs(60), async {
            while !api_client.ready().await? {
                tracing::trace!("AllTalk not ready yet, waiting");
                tokio::time::sleep(Duration::from_secs(1)).await
            }
            
            Ok::<_, eyre::Report>(())
        }).await??;

        let settings = api_client.current_settings().await?;

        Ok(Self {
            api: api_client,
            all_talk: settings,
        })
    }
}

