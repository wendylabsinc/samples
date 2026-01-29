use axum::{routing::get, Json, Router};
use chrono::Utc;
use rand::Rng;
use serde::Serialize;
use std::env;
use std::path::PathBuf;
use tower_http::services::{ServeDir, ServeFile};

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Car {
    name: String,
    make: String,
    year: u32,
    color: String,
    created_at: String,
}

const CAR_NAMES: &[&str] = &["Honda", "Toyota", "Ford", "Chevrolet", "BMW", "Mercedes", "Audi", "Tesla", "Nissan", "Mazda"];
const CAR_MAKES: &[&str] = &["Civic", "Camry", "Mustang", "Corvette", "M3", "C-Class", "A4", "Model 3", "Altima", "MX-5"];

fn random_car() -> Car {
    let mut rng = rand::thread_rng();
    let name = CAR_NAMES[rng.gen_range(0..CAR_NAMES.len())].to_string();
    let make = CAR_MAKES[rng.gen_range(0..CAR_MAKES.len())].to_string();
    let year = rng.gen_range(1990..2025);
    let color = format!(
        "#{:02X}{:02X}{:02X}",
        rng.gen_range(0..=255),
        rng.gen_range(0..=255),
        rng.gen_range(0..=255)
    );
    let created_at = Utc::now().to_rfc3339();

    Car {
        name,
        make,
        year,
        color,
        created_at,
    }
}

async fn get_random_car() -> Json<Car> {
    Json(random_car())
}

#[tokio::main]
async fn main() {
    let hostname = env::var("WENDY_HOSTNAME").unwrap_or_else(|_| "0.0.0.0".to_string());

    // Serve the frontend dist folder
    let frontend_dist = env::var("FRONTEND_DIST")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            let container_path = PathBuf::from("/app/frontend/dist");
            if container_path.exists() {
                return Some(container_path);
            }
            let cwd_path = PathBuf::from("frontend/dist");
            if cwd_path.exists() {
                return Some(cwd_path);
            }
            let dev_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .map(|p| p.join("frontend/dist"));
            dev_path.filter(|p| p.exists())
        })
        .unwrap_or_else(|| PathBuf::from("/app/frontend/dist"));

    println!("Server running on http://{}:4002", hostname);
    println!("Serving frontend from: {:?}", frontend_dist);

    let index_file = frontend_dist.join("index.html");

    let serve_dir = ServeDir::new(&frontend_dist)
        .not_found_service(ServeFile::new(&index_file));

    let app = Router::new()
        .route("/api/random-car", get(get_random_car))
        .fallback_service(serve_dir);

    // Bind to 0.0.0.0 to accept connections from all interfaces (required for container networking)
    let listener = tokio::net::TcpListener::bind("0.0.0.0:4002").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
