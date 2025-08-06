use std::time::Duration;
use crate::tts_backends::indextts::api::{IndexTtsAPI, IndexTtsApiConfig};

pub mod api;
pub mod local;

pub struct IndexTts {
    api: IndexTtsAPI,
}

impl IndexTts {
    pub async fn new(config: IndexTtsApiConfig) -> eyre::Result<Self> {
        let api_client = IndexTtsAPI::new(config)?;

        // Wait for it to be ready
        tokio::time::timeout(Duration::from_secs(120), async {
            while !api_client.ready().await? {
                tracing::trace!("IndexTTS not ready yet, waiting");
                tokio::time::sleep(Duration::from_secs(1)).await
            }

            Ok::<_, eyre::Report>(())
        }).await??;
        tracing::trace!("IndexTTS ready!");

        Ok(Self {
            api: api_client,
        })
    }
}


mod text_processing {
    //! Index-TTS has a few pronunciation peculiarities which we need to handle by preprocessing text:
    //! 1. Conjunctions with a dash (e.g., 'barely-there') should have the dash removed or the pronunciation will have a long pause.
    //! 2. Certain words need a literal writing (e.g., 'tieflings' -> 'teeflings') in order to have a correct pronunciation.

    use papaya::HashMap;

    pub struct TextProcessor {
        replace_tokens: HashMap<String, String>,
        dash_replace: regex::Regex,
        apostrophe_replace: regex::Regex,
    }

    impl TextProcessor {
        pub fn new(tokens: HashMap<String, String>) -> Self {
            Self {
                replace_tokens: tokens,
                dash_replace: regex::Regex::new(r"(\w+)-(\w+)").unwrap(),
                apostrophe_replace: regex::Regex::new(r"(\w+)'s").unwrap(),
            }
        }

        pub fn process(&self, text: impl AsRef<str>) -> String {
            let stack = text.as_ref();

            let dash_replaced = self.dash_replace.replace_all(stack, "$1 $2").into_owned();
            let mut dash_replaced = self.apostrophe_replace.replace_all(&dash_replaced, "$1 is").into_owned();

            // TODO: For now a _very_ inefficient replacement, but later on use [AhoCorasick::replace_all]
            for (token, replacement) in self.replace_tokens.pin().iter() {
                dash_replaced = dash_replaced.replace(token, replacement)
            }

            dash_replaced
        }
    }
}