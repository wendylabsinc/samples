import express from "express";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const hostname = process.env.WENDY_HOSTNAME || "0.0.0.0";
const port = 5002;

const CAR_NAMES = ["Honda", "Toyota", "Ford", "Chevrolet", "BMW", "Mercedes", "Audi", "Tesla", "Nissan", "Mazda"];
const CAR_MAKES = ["Civic", "Camry", "Mustang", "Corvette", "M3", "C-Class", "A4", "Model 3", "Altima", "MX-5"];

function randomCar() {
  const name = CAR_NAMES[Math.floor(Math.random() * CAR_NAMES.length)];
  const make = CAR_MAKES[Math.floor(Math.random() * CAR_MAKES.length)];
  const year = Math.floor(Math.random() * (2025 - 1990)) + 1990;
  const color = `#${Math.floor(Math.random() * 16777215).toString(16).padStart(6, "0").toUpperCase()}`;
  const createdAt = new Date().toISOString();

  return { name, make, year, color, createdAt };
}

// Serve the frontend dist folder
const frontendDist =
  process.env.FRONTEND_DIST ||
  (require("fs").existsSync("/app/frontend/dist") ? "/app/frontend/dist" : path.join(__dirname, "../../frontend/dist"));

console.log(`Serving frontend from: ${frontendDist}`);

// API routes
app.get("/api/random-car", (_req, res) => {
  res.json(randomCar());
});

// Static files
app.use(express.static(frontendDist));

// Fallback to index.html for SPA routing
app.get("*", (_req, res) => {
  res.sendFile(path.join(frontendDist, "index.html"));
});

app.listen(port, () => {
  console.log(`Server running on http://${hostname}:${port}`);
});
