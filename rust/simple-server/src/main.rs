use axum::{
    routing::{get, post},
    http::StatusCode,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::env;

#[tokio::main]
async fn main() {
    let hostname = env::var("WENDY_HOSTNAME").unwrap_or_else(|_| "0.0.0.0".to_string());

    let app = Router::new()
        .route("/", get(root))
        .route("/hello/:name", get(hello))
        .route("/users", post(create_user));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:4001").await.unwrap();
    println!("Server running on http://{}:4001", hostname);
    axum::serve(listener, app).await.unwrap();
}

async fn root() -> &'static str {
    println!("Received request: GET /");
    "Hello, World!"
}

async fn hello(axum::extract::Path(name): axum::extract::Path<String>) -> String {
    println!("Received request: GET /hello/{}", name);
    format!("Hello, {}!", name)
}

async fn create_user(Json(payload): Json<CreateUser>) -> (StatusCode, Json<User>) {
    println!("Received request: POST /users");
    let user = User {
        id: 1,
        username: payload.username,
    };
    (StatusCode::CREATED, Json(user))
}

#[derive(Deserialize)]
struct CreateUser {
    username: String,
}

#[derive(Serialize)]
struct User {
    id: u64,
    username: String,
}
