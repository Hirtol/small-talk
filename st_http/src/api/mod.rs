use std::ops::Deref;
use std::sync::Arc;
use aide::axum::IntoApiResponse;
use aide::axum::routing::{get, get_with};
use aide::openapi::{OpenApi, Tag};
use aide::scalar::Scalar;
use aide::transform::TransformOpenApi;
use axum::{Extension, Router};
use axum::response::IntoResponse;
use crate::api::error::{ApiError, ApiResponseError};
use crate::api::extractor::Json;
use crate::config::SharedConfig;
use st_system::{TtsSystem, TtsSystemHandle};

mod extractor;
pub mod error;
pub mod session;

pub type ApiRouter<S = ()> = aide::axum::ApiRouter<S>;
pub type ApiResult<T, E = ApiError> = Result<T, E>;

#[derive(Clone)]
pub struct AppState {
    pub(crate) config: SharedConfig,
    pub(crate) system: TtsSystemHandle,
}

/// Root config for all GraphQL queries
pub fn config(app_state: AppState) -> Router<AppState> {
    aide::r#gen::on_error(|error| {
        tracing::error!(?error, "Aide Error");
    });

    aide::r#gen::extract_schemas(true);
    let mut api = OpenApi::default();
    
    let base_router = ApiRouter::new()
        .nest_api_service("/docs", docs_routes())
        .merge(session::routes::config());
    
    ApiRouter::new()
        .nest("/api", base_router)
        .finish_api_with(&mut api, api_docs)
        .layer(Extension(Arc::new(api)))
}

pub fn docs_routes() -> ApiRouter {
    aide::r#gen::infer_responses(true);

    let router: ApiRouter = ApiRouter::new()
        .api_route(
            "/",
            get_with(
                Scalar::new("/api/docs/api.json")
                    .with_title("Aide Axum")
                    .axum_handler(),
                |op| op.description("This documentation page."),
            ),
        )
        .route("/api.json", get(serve_docs));

    aide::r#gen::infer_responses(false);

    router
}

async fn serve_docs(Extension(api): Extension<Arc<OpenApi>>) -> impl IntoApiResponse {
    Json(api).into_response()
}

fn api_docs(api: TransformOpenApi) -> TransformOpenApi {
    api.title("Small Talk")
        .summary("A TTS application")
        .description(include_str!("../../../README.md"))
        .default_response_with::<Json<ApiResponseError<()>>, _>(|res| {
            res.example(ApiResponseError {
                code: 500,
                message: "An error occurred".to_string(),
                details: None,
            })
        })
}