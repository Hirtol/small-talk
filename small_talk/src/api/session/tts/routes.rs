use crate::{
    api::{
        extractor::Json,
        session::{
            tts::{ApiTtsRequest, ApiTtsResponse},
            Session,
        },
        ApiResult, ApiRouter, AppState,
    },
    system::playback::PlaybackVoiceLine,
};
use aide::{axum::routing::post_with, transform::TransformOperation};
use axum::extract::{Path, State};
use schemars::JsonSchema;
use serde::Serialize;
use std::collections::VecDeque;

pub fn config() -> ApiRouter<AppState> {
    ApiRouter::new().nest(
        "/tts",
        ApiRouter::new()
            .api_route("/request", post_with(tts_request, tts_request_docs))
            .api_route("/queue", post_with(tts_queue, tts_queue_docs))
            .nest(
                "/playback",
                ApiRouter::new()
                    .api_route("/start", post_with(tts_playback_start, tts_playback_start_request_docs))
                    .api_route("/stop", post_with(tts_playback_stop, tts_playback_stop_request_docs)),
            ),
    )
}

#[tracing::instrument(skip_all)]
pub async fn tts_request(
    state: State<AppState>,
    Path(game_name): Path<Session>,
    Json(request): Json<ApiTtsRequest>,
) -> ApiResult<Json<ApiTtsResponse>> {
    let session_handle = state.system.get_or_start_session(&game_name.id).await?;
    let result = session_handle.request_tts(request.into()).await?;

    let api_result = ApiTtsResponse {
        file_path: result.file_path.clone(),
    };

    Ok(api_result.into())
}

fn tts_request_docs(op: TransformOperation) -> TransformOperation {
    op.description("Start a TTS request. This will only return upon the completion of the TTS generation.")
        .response::<204, Json<ApiTtsResponse>>()
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct TtsQueueResponse {
    items: usize,
}

#[tracing::instrument(skip_all)]
pub async fn tts_queue(
    state: State<AppState>,
    Path(game_name): Path<Session>,
    Json(request): Json<Vec<ApiTtsRequest>>,
) -> ApiResult<Json<TtsQueueResponse>> {
    let session_handle = state.system.get_or_start_session(&game_name.id).await?;
    let items = request.len();
    session_handle
        .add_all_to_queue(request.into_iter().map(|v| v.into()).collect())
        .await?;

    let api_result = TtsQueueResponse { items };

    Ok(api_result.into())
}

fn tts_queue_docs(op: TransformOperation) -> TransformOperation {
    op.description("Add all lines to the async TTS queue. This request will not block and instead immediately return.")
        .response::<200, Json<TtsQueueResponse>>()
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct TtsPlaybackRequest {
    /// The line to request.
    request: ApiTtsRequest,
    /// Request a volume between [0.0, 1.0] for the playback of this line.
    volume: Option<f32>,
}

#[tracing::instrument(skip_all)]
pub async fn tts_playback_start(
    state: State<AppState>,
    Path(game_name): Path<Session>,
    Json(requests): Json<VecDeque<TtsPlaybackRequest>>,
) -> ApiResult<()> {
    let session_handle = state.system.get_or_start_session(&game_name.id).await?;
    session_handle
        .playback
        .start(
            requests
                .into_iter()
                .map(|api| PlaybackVoiceLine {
                    line: api.request.into(),
                    volume: api.volume,
                })
                .collect(),
        )
        .await?;

    Ok(())
}

fn tts_playback_start_request_docs(op: TransformOperation) -> TransformOperation {
    op.description("Start a local playback of the given voice-line. This will return immediately, even if the voiceline hasn't finished playing yet.")
        .response::<200, ()>()
}

#[tracing::instrument(skip_all)]
pub async fn tts_playback_stop(state: State<AppState>, Path(game_name): Path<Session>) -> ApiResult<()> {
    let session_handle = state.system.get_or_start_session(&game_name.id).await?;
    session_handle.playback.stop().await?;

    Ok(())
}

fn tts_playback_stop_request_docs(op: TransformOperation) -> TransformOperation {
    op.description("Stop a playback if one is currently ongoing")
        .response::<200, ()>()
}
