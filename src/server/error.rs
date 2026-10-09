//! OpenAI-style error bodies: `{"error": {"message", "type", "param", "code"}}`.

use axum::{
    Json,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::Serialize;

use crate::domain::RunId;

use super::openai::Drew;

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub message: String,
    pub kind: &'static str,
    pub code: &'static str,
    pub run_id: Option<RunId>,
    /// Attached when a persisted run failed, so the caller sees its trace.
    pub drew: Option<Box<Drew>>,
}

impl ApiError {
    fn new(status: StatusCode, kind: &'static str, code: &'static str, message: String) -> Self {
        Self {
            status,
            message,
            kind,
            code,
            run_id: None,
            drew: None,
        }
    }

    pub fn unauthorized(message: &str) -> Self {
        Self::new(
            StatusCode::UNAUTHORIZED,
            "invalid_request_error",
            "invalid_api_key",
            message.into(),
        )
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_request",
            message.into(),
        )
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            "not_found",
            message.into(),
        )
    }

    pub fn internal(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            code,
            message.into(),
        )
    }

    pub fn with_run(self, run_id: RunId) -> Self {
        Self {
            run_id: Some(run_id),
            ..self
        }
    }

    pub fn with_drew(self, drew: Drew) -> Self {
        Self {
            drew: Some(Box::new(drew)),
            ..self
        }
    }
}

#[derive(Serialize)]
struct Body<'a> {
    error: ErrorObject<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    drew: Option<&'a Drew>,
}

#[derive(Serialize)]
struct ErrorObject<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    kind: &'a str,
    param: Option<&'a str>,
    code: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    run_id: Option<RunId>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = Body {
            error: ErrorObject {
                message: &self.message,
                kind: self.kind,
                param: None,
                code: self.code,
                run_id: self.run_id,
            },
            drew: self.drew.as_deref(),
        };
        let mut response = (self.status, Json(body)).into_response();
        if self.status == StatusCode::UNAUTHORIZED {
            response.headers_mut().insert(
                header::WWW_AUTHENTICATE,
                header::HeaderValue::from_static("Bearer"),
            );
        }
        response
    }
}
