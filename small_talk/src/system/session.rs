use crate::{
    config::SharedConfig,
    system::{Voice, VoiceLine},
};
use path_abs::PathOps;
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    path::Path,
};

const CONFIG_NAME: &str = "config.json";

pub type GameSessionActorHandle = tokio::sync::mpsc::UnboundedSender<GameSessionMessage>;

#[derive(Debug, Clone)]
pub enum GameSessionMessage {
    
}

pub struct GameSessionActor {
    pub game_data: GameData,
    pub config: SharedConfig,
    pub queue: VecDeque<VoiceLine>,
}

impl GameSessionActor {
    pub async fn create_or_load_from_file(game_name: &str, config: SharedConfig) -> eyre::Result<Self> {
        let dir = config.dirs.game_data_path().join(game_name);

        if tokio::fs::try_exists(&dir).await? {
            Self::load_from_dir(&dir, config).await
        } else {
            Self::create(game_name, &dir, config).await
        }
    }

    async fn create(game_name: &str, dir: &Path, config: SharedConfig) -> eyre::Result<Self> {
        let data = GameData {
            game_name: game_name.into(),
            character_map: Default::default(),
        };
        let out = serde_json::to_vec_pretty(&data)?;

        tokio::fs::create_dir_all(dir).await?;
        tokio::fs::write(dir.join(CONFIG_NAME), &out).await?;

        Ok(Self {
            game_data: data,
            config,
            queue: Default::default(),
        })
    }

    async fn load_from_dir(dir: &Path, config: SharedConfig) -> eyre::Result<Self> {
        let game_data = tokio::fs::read(dir.join(CONFIG_NAME)).await?;
        let data = serde_json::from_slice(&game_data)?;

        Ok(Self {
            game_data: data,
            config,
            queue: Default::default(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GameData {
    pub game_name: String,
    pub character_map: papaya::HashMap<String, Voice>,
}

pub struct SessionOrchestrator {
    pub game_sessions: papaya::HashMap<String, GameSessionActorHandle>,
    
}
