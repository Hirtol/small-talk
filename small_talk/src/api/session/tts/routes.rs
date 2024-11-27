use aide::axum::routing::post_with;
use aide::transform::TransformOperation;
use axum::extract::{Path, State};
use crate::api::{ApiResult, ApiRouter, AppState};
use crate::api::error::ApiError;
use crate::api::extractor::Json;
use crate::api::session::tts::{ApiTtsRequest, ApiTtsResponse};

pub fn config() -> ApiRouter<AppState> {
    ApiRouter::new().nest("/tts",
                          ApiRouter::new()
                              .api_route("/request", post_with(tts_request, tts_request_docs)),
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