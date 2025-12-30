use serde::{Deserialize, Serialize};
use st_application::config::SmallTalkConfig;
use std::{fmt::Debug};
use std::sync::Arc;
use tokio::net::ToSocketAddrs;

pub type SharedConfig = Arc<SmallTalkHttpConfig>;

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct SmallTalkHttpConfig {
    /// Bindings and host address
    #[serde(default)]
    pub app: ServerConfig,
    #[serde(default)]
    pub small_talk: SmallTalkConfig,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

impl ServerConfig {
    /// Turn the app config settings into a [ToSocketAddrs]
    pub fn bind_address(&self) -> impl ToSocketAddrs {
        (self.host.clone(), self.port)
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            host: "0.0.0.0".to_string(),
            port: 8100,
        }
    }
}
