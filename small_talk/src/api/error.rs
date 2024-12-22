use aide::axum::IntoApiResponse;
use aide::{OperationIo, OperationOutput};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum_jsonschema::JsonSchemaRejection;
use error_set::error_set;
use schemars::JsonSchema;
use serde::{Serialize};
use crate::api::extractor::Json;

error_set! {
    #[derive(OperationIo)]
    ApiError = {
        #[display("Internal error, please submit a bug report: {0}")]
        Other(eyre::Error),
        #[display("JSON validation error {source:?}")]
        Json {
            source: JsonSchemaRejection
        }
    };
}

#[derive(serde::Serialize, serde::Deserialize, JsonSchema, Clone, Debug, PartialEq, Eq)]
pub struct ApiResponseError<T> {
    pub code: u16,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<T>,
}

impl<T: Serialize> IntoResponse for ApiResponseError<T> {
    fn into_response(self) -> Response {
        (StatusCode::from_u16(self.code).unwrap(), Json(self)).into_response()
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let response = ApiResponseError {
            code: 0,
            message: self.to_string(),
            details: None::<()>,
        };
        
        let status_error = match self {
            ApiError::Other(e) => {
                tracing::error!("Internal error occurred: {e:?}");
                StatusCode::INTERNAL_SERVER_ERROR
            }
            ApiError::Json { source } => {
                return source.into_response()
            }
        };

        (status_error, Json(response)).into_response()
    }
}

impl From<JsonSchemaRejection> for ApiError {
    fn from(value: JsonSchemaRejection) -> Self {
        ApiError::Json {
            source: value,
        }
    }
}
