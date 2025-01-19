use std::time::Duration;
use crate::system::rvc_backends::seedvc::api::{SeedVcApi, SeedVcApiConfig};

pub mod api;
pub mod local;

pub struct SeedRvc {
    api: SeedVcApi,
}

impl SeedRvc {
    pub async fn new(config: SeedVcApiConfig) -> eyre::Result<Self> {
        let api_client = SeedVcApi::new(config)?;

        // Wait for it to be ready
        tokio::time::timeout(Duration::from_secs(120), async {
            while !api_client.ready().await? {
                tracing::trace!("SeedVc not ready yet, waiting");
                tokio::time::sleep(Duration::from_secs(1)).await
            }

            Ok::<_, eyre::Report>(())
        }).await??;
        tracing::trace!("SeedVc ready!");

        Ok(Self {
            api: api_client,
        })
    }
}