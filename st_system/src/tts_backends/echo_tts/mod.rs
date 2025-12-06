use crate::tts_backends::{
    echo_tts::api::{EchoTtsAPI, EchoTtsApiConfig},
};
use std::time::Duration;

pub mod api;
pub mod local;

pub struct EchoTts {
    api: EchoTtsAPI,
}

impl EchoTts {
    pub async fn new(config: EchoTtsApiConfig) -> eyre::Result<Self> {
        let api_client = EchoTtsAPI::new(config)?;

        // Wait for it to be ready
        tokio::time::timeout(Duration::from_secs(120), async {
            while !api_client.ready().await? {
                tracing::trace!("EchoTTS not ready yet, waiting");
                tokio::time::sleep(Duration::from_secs(1)).await
            }

            Ok::<_, eyre::Report>(())
        })
        .await??;
        tracing::trace!("EchoTTS ready!");

        Ok(Self { api: api_client })
    }
}

mod text_processing {
    use papaya::HashMap;

    pub struct TextProcessor {
        replace_tokens: HashMap<String, String>,
    }

    impl TextProcessor {
        pub fn new(tokens: HashMap<String, String>) -> Self {
            Self { replace_tokens: tokens }
        }

        pub fn process(&self, text: impl Into<String>) -> String {
            let mut stack = text.into();

            // TODO: For now a _very_ inefficient replacement, but later on use [AhoCorasick::replace_all]
            for (token, replacement) in self.replace_tokens.pin().iter() {
                stack = stack.replace(token, replacement)
            }

            stack
        }
    }
}
