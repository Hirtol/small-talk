use aide::axum::routing::post_with;
use aide::transform::TransformOperation;
use axum::extract::{Path, State};
use schemars::JsonSchema;
use serde::Serialize;
use crate::api::{ApiResult, ApiRouter, AppState};
use crate::api::error::ApiError;
use crate::api::extractor::Json;
use crate::api::session::tts::{ApiTtsRequest, ApiTtsResponse};

pub fn config() -> ApiRouter<AppState> {
    ApiRouter::new().nest("/tts",
                          ApiRouter::new()
                              .api_route("/request", post_with(tts_request, tts_request_docs))
                              .api_route("/queue", post_with(tts_queue, tts_queue_docs)),
    )
}

#[tracing::instrument(skip_all)]
pub async fn tts_request(state: State<AppState>, Path(game_name): Path<String>, Json(request): Json<ApiTtsRequest>) -> ApiResult<Json<ApiTtsResponse>> {
    let session_handle = state.system.get_or_start_session(&game_name).await?;
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
pub async fn tts_queue(state: State<AppState>, Path(game_name): Path<String>, Json(request): Json<Vec<ApiTtsRequest>>) -> ApiResult<Json<TtsQueueResponse>> {
    let session_handle = state.system.get_or_start_session(&game_name).await?;
    let items = request.len();
    session_handle.add_all_to_queue(request.into_iter().map(|v| v.into()).collect()).await?;

    let api_result = TtsQueueResponse {
        items,
    };

    Ok(api_result.into())
}

fn tts_queue_docs(op: TransformOperation) -> TransformOperation {
    op.description("Add all lines to the async TTS queue. This request will not block and instead immediately return.")
        .response::<200, Json<TtsQueueResponse>>()
}