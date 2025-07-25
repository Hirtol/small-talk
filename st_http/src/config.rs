use std::fmt::{Debug};
use std::io::Write;
use tokio::net::ToSocketAddrs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use st_system::config::TtsSystemConfig;
use st_system::rvc_backends::seedvc::api::SeedVcApiConfig;
use st_system::tts_backends::alltalk::AllTalkConfig;

pub type SharedConfig = Arc<Config>;

static CONFIG_FILE: &str = "st_config.toml";

/// Initialise the config file.
///
/// Creates a new config file if it doesn't yet exist, otherwise loads the existing one.
///
/// It follows a simple priority order (from most to least important, which will determine overrides):
/// 1. Environment variables prefixed by `smalltalk`
/// 2. Local config file (same execution directory as process)
/// 3. Standard config file, located in some `appdata` directory.
pub fn initialise_config() -> eyre::Result<Config> {
    let c_path = get_full_config_path();

    if !c_path.exists() {
        save_config(&Config::default())?;
    }

    let c = config::Config::builder()
        .add_source(config::File::with_name(&c_path.to_string_lossy()).required(true))
        .add_source(config::File::with_name(CONFIG_FILE).required(false))
        .add_source(config::Environment::with_prefix("smalltalk"))
        .build()?;
    
    Ok(c.try_deserialize()?)
}

/// Save the provided config to the known config directory.
pub fn save_config(app_settings: &Config) -> eyre::Result<()> {
    std::fs::create_dir_all(get_config_directory())?;

    let mut config_file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(get_full_config_path())?;

    let basic_output = toml::to_string_pretty(app_settings)?;

    config_file.write_all(basic_output.as_bytes())?;

    Ok(())
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Config {
    /// Bindings and host address
    #[serde(default)]
    pub app: ServerConfig,
    /// All directory related configs
    #[serde(default)]
    pub dirs: Arc<TtsSystemConfig>,
    #[serde(default)]
    pub xtts: SubsystemConfig<TtsConfig>,
    #[serde(default)]
    pub index_tts: SubsystemConfig<st_system::tts_backends::indextts::local::LocalIndexTtsConfig>,
    #[serde(default)]
    pub seed_vc: SubsystemConfig<RvcConfig>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct SubsystemConfig<T> {
    /// Whether this subsystem should be enabled or disabled.
    pub enabled: bool,
    #[serde(flatten)]
    pub inner: T,
}

impl<T> SubsystemConfig<T> {
    /// Turn this config into an [Option]. `Some` if `enabled` is true, `None` otherwise.
    pub fn if_enabled(&self) -> Option<&T> {
        if self.enabled {
            Some(&self.inner)
        } else {
            None
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TtsConfig {
    /// Directory containing an AllTalk instance.
    pub local_all_talk: PathBuf,
    /// How long until the resources allocated to the local ML should be freed after not being used.
    pub timeout: Duration,
    pub alltalk_cfg: AllTalkConfig,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RvcConfig {
    /// Directory containing a SeedVc instance.
    pub local_path: PathBuf,
    /// How long until the resources allocated to the local ML should be freed after not being used.
    pub timeout: Duration,
    pub config: SeedVcApiConfig,
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

impl Default for TtsConfig {
    fn default() -> Self {
        let app_dir = st_system::get_app_dirs().config_dir;
        Self {
            local_all_talk: app_dir.join("alltalk"),
            timeout: Duration::from_secs(30 * 60),
            alltalk_cfg: AllTalkConfig::new(url::Url::parse("http://localhost:7851/").unwrap()),
        }
    }
}

impl Default for RvcConfig {
    fn default() -> Self {
        let app_dir = st_system::get_app_dirs().config_dir;
        Self {
            local_path: app_dir.join("seedvc"),
            timeout: Duration::from_secs(30 * 60),
            config: SeedVcApiConfig {
                address: url::Url::parse("http://localhost:9999/").unwrap()
            },
        }
    }
}

/// Retrieve the *full* path to the config file.
///
/// This is just [get_config_directory] + [CONFIG_FILE]
pub fn get_full_config_path() -> PathBuf {
    get_config_directory().join(CONFIG_FILE)
}

/// Retrieve the directory which will be used to locate/save the config file.
pub fn get_config_directory() -> PathBuf {
    st_system::get_app_dirs().config_dir
}