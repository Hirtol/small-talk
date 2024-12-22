use aide::operation::OperationIo;
use axum::extract::{FromRequest, FromRequestParts};
use axum::response::IntoResponse;
use serde::Serialize;
use crate::api::error::ApiError;

#[derive(FromRequest, OperationIo)]
#[from_request(via(axum_jsonschema::Json), rejection(ApiError))]
#[aide(
    input_with = "axum_jsonschema::Json<T>",
    output_with = "axum_jsonschema::Json<T>",
    json_schema
)]
pub struct Json<T>(pub T);

impl<T> IntoResponse for Json<T>
where
    T: Serialize,
{
    fn into_response(self) -> axum::response::Response {
        axum::Json(self.0).into_response()
    }
}

impl<T> From<T> for Json<T> {
    fn from(value: T) -> Self {
        Json(value)
    }
}

#[derive(FromRequestParts, OperationIo)]
#[from_request(via(axum::extract::Query), rejection(ApiError))]
#[aide(
    input_with = "axum::extract::Query<T>",
    output_with = "axum_jsonschema::Json<T>",
    json_schema
)]
#[aide]
pub struct Query<T>(pub T);

#[derive(FromRequestParts, OperationIo)]
#[from_request(via(axum::extract::Path), rejection(ApiError))]
#[aide(
    input_with = "axum::extract::Path<T>",
    output_with = "axum_jsonschema::Json<T>",
    json_schema
)]
pub struct Path<T>(pub T);