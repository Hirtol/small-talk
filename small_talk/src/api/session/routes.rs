use std::path::PathBuf;
use aide::axum::IntoApiResponse;
use aide::axum::routing::{post, post_with};
use aide::transform::TransformOperation;
use axum::extract::State;
use axum::handler::Handler;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use crate::api::{ApiResult, ApiRouter, AppState};
use crate::api::error::ApiError;
use crate::api::extractor::Json;
use crate::system::Voice;

pub fn config() -> ApiRouter<AppState> {
    ApiRouter::new().nest("/tts", 
                          ApiRouter::new()
                              .api_route("/request", post_with(tts_request, tts_request_docs)),
    )
}

#[derive(Serialize, Debug, JsonSchema)]
pub struct TtsResponse {
    pub file_path: PathBuf,
    pub data_url: String,
    pub voice_used: Voice,
}

#[derive(Deserialize, Serialize, Debug, JsonSchema)]
pub struct TtsRequest {
    pub text: String,
    pub voice: TtsVoice
}

#[derive(Deserialize, Serialize, Debug, JsonSchema)]
pub enum TtsVoice {
    Set(String),
    Random(RandomVoice),
}

#[derive(Deserialize, Serialize, Debug, JsonSchema)]
pub struct RandomVoice {
    gender: Gender,
}

#[derive(Deserialize, Serialize, Debug, JsonSchema)]
pub enum Gender {
    Male,
    Female,
}

#[tracing::instrument(skip_all)]
pub async fn tts_request(state: State<AppState>, request: Json<TtsRequest>) -> ApiResult<Json<TtsResponse>> {
    // Ok(Json(TtsResponse {
    //     file_path: Default::default(),
    //     data_url: "".to_string(),
    // }))
    Err(ApiError::Other(eyre::eyre!("Fuck it")))
    // todo!()
}

fn tts_request_docs(op: TransformOperation) -> TransformOperation {
    op.description("Start a TTS request. This will only return upon the completion of the TTS generation.")
        .response::<204, Json<TtsResponse>>()
}