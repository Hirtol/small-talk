use std::path::PathBuf;
use aide::axum::IntoApiResponse;
use aide::axum::routing::{get_with, post, post_with};
use aide::transform::TransformOperation;
use axum::extract::{Path, State};
use axum::handler::Handler;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use crate::api::{ApiResult, ApiRouter, AppState};
use crate::api::extractor::Json;
use crate::api::session::Session;
use crate::system::{TtsResponse, Voice};
use crate::system::voice_manager::VoiceReference;

pub fn config() -> ApiRouter<AppState> {
    ApiRouter::new().nest("/session/:game_name",
                          ApiRouter::new()
                              .api_route("/start", post_with(session_start, session_start_docs))
                              .api_route("/voices", get_with(get_session_voices, get_session_voices_docs))
                              .api_route("/stop", post_with(session_stop, session_stop_docs))
                              .merge(super::tts::config()),
    )
}

#[derive(Debug, JsonSchema, Serialize, Deserialize)]
pub struct ApiSessionStart {
    /// The name of the game session, should equal the name of the game being played.
    pub game_name: String,
}

#[tracing::instrument(skip(state))]
pub async fn session_start(state: State<AppState>, Path(game_name): Path<String>) -> ApiResult<Json<Session>> {
    let _ = state.system.get_or_start_session(&game_name).await?;
    
    Ok(Session {
        id: game_name,
    }.into())
}

fn session_start_docs(op: TransformOperation) -> TransformOperation {
    op.description("Start a session, initialising a request queue and optional player")
        .response::<200, Json<Session>>()
}

#[tracing::instrument(skip(state))]
pub async fn get_session_voices(state: State<AppState>, Path(game_name): Path<String>) -> ApiResult<Json<Vec<VoiceReference>>> {
    let sess = state.system.get_or_start_session(&game_name).await?;
    
    let output = sess.available_voices().await?.into_iter().map(|v| v.reference).collect();
    
    Ok(Json(output))
}

fn get_session_voices_docs(op: TransformOperation) -> TransformOperation {
    op.description("Retrieve the available voices for characters within this game session.\nThis includes global voices.")
        .response::<200, Json<Vec<VoiceReference>>>()
}

#[tracing::instrument(skip(state))]
pub async fn session_stop(state: State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Session>> {
    state.system.stop_session(&id).await?;

    Ok(Json(Session {
        id,
    }))
}

fn session_stop_docs(op: TransformOperation) -> TransformOperation {
    op.description("Stop a session, dropping any TTS requests still in the queue")
        .response::<200, Json<Session>>()
}
