use std::collections::HashMap;
use aide::axum::routing::{get_with, post, post_with, put_with};
use aide::transform::TransformOperation;
use axum::extract::{Path, State};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use crate::api::{ApiResult, ApiRouter, AppState};
use crate::api::extractor::{Json};
use crate::api::session::Session;
use st_system::{CharacterName, Voice};
use st_system::voice_manager::VoiceReference;

pub fn config() -> ApiRouter<AppState> {
    ApiRouter::new().nest("/session/{id}",
                          ApiRouter::new()
                              .api_route("/start", post_with(session_start, session_start_docs))
                              .api_route("/stop", post_with(session_stop, session_stop_docs))
                              .api_route("/voices", get_with(get_session_voices, get_session_voices_docs))
                              .api_route("/characters", get_with(get_session_characters, get_session_characters_docs))
                              .api_route("/characters", put_with(put_session_character, put_session_characters_docs))
                              .merge(super::tts::config()),
    ).with_path_items(|t| t.tag("Game Session TTS").description("All routes related to TTS requests for a particular game"))
}

#[derive(Debug, JsonSchema, Serialize, Deserialize)]
pub struct ApiSessionStart {
    /// The name of the game session, should equal the name of the game being played.
    pub game_name: String,
}

#[tracing::instrument(skip(state))]
pub async fn session_start(state: State<AppState>, Path(game_name): Path<Session>) -> ApiResult<Json<Session>> {
    let _ = state.system.get_or_start_session(&game_name.id).await?;
    
    Ok(game_name.into())
}

fn session_start_docs(op: TransformOperation) -> TransformOperation {
    op.description("Start a session, initialising a request queue and optional player")
        .response::<200, Json<Session>>()
}

#[tracing::instrument(skip(state))]
pub async fn session_stop(state: State<AppState>, Path(game_name): Path<Session>) -> ApiResult<Json<Session>> {
    state.system.stop_session(&game_name.id).await?;

    Ok(game_name.into())
}

fn session_stop_docs(op: TransformOperation) -> TransformOperation {
    op.description("Stop a session, dropping any TTS requests still in the queue")
        .response::<200, Json<Session>>()
}

#[tracing::instrument(skip(state))]
pub async fn get_session_voices(state: State<AppState>, Path(game_name): Path<Session>) -> ApiResult<Json<Vec<VoiceReference>>> {
    let sess = state.system.get_or_start_session(&game_name.id).await?;
    
    let output = sess.available_voices().await?.into_iter().map(|v| v.reference).collect();
    
    Ok(Json(output))
}

fn get_session_voices_docs(op: TransformOperation) -> TransformOperation {
    op.description("Retrieve the available voices for characters within this game session.\nThis includes global voices.")
        .response::<200, Json<Vec<VoiceReference>>>()
}

#[tracing::instrument(skip(state))]
pub async fn get_session_characters(state: State<AppState>, Path(game_name): Path<Session>) -> ApiResult<Json<HashMap<CharacterName, VoiceReference>>> {
    let sess = state.system.get_or_start_session(&game_name.id).await?;

    let output = sess.character_voices().await?;

    Ok(Json(output))
}

fn get_session_characters_docs(op: TransformOperation) -> TransformOperation {
    op.description("Retrieve the assigned voices for each character in this game session.\nThis does not include characters which have not yet been seen.")
        .response::<200, Json<HashMap<CharacterName, VoiceReference>>>()
}

#[derive(Debug, Deserialize, JsonSchema)]
struct PutSessionCharacter {
    name: CharacterName,
    voice: VoiceReference,
}

#[tracing::instrument(skip(state))]
pub async fn put_session_character(state: State<AppState>, Path(game_name): Path<Session>, Json(put): Json<PutSessionCharacter>) -> ApiResult<()> {
    let sess = state.system.get_or_start_session(&game_name.id).await?;

    sess.force_character_voice(put.name, put.voice).await?;

    Ok(())
}

fn put_session_characters_docs(op: TransformOperation) -> TransformOperation {
    op.description("Force the given character to always use the given voice, potentially overriding any existing voice used.")
        .response::<200, ()>()
}
