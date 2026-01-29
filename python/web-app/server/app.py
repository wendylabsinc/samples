import os
import random
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI()

CAR_NAMES = ["Honda", "Toyota", "Ford", "Chevrolet", "BMW", "Mercedes", "Audi", "Tesla", "Nissan", "Mazda"]
CAR_MAKES = ["Civic", "Camry", "Mustang", "Corvette", "M3", "C-Class", "A4", "Model 3", "Altima", "MX-5"]

def random_car():
    name = random.choice(CAR_NAMES)
    make = random.choice(CAR_MAKES)
    year = random.randint(1990, 2024)
    color = f"#{random.randint(0, 0xFFFFFF):06X}"
    created_at = datetime.now(timezone.utc).isoformat()
    return {
        "name": name,
        "make": make,
        "year": year,
        "color": color,
        "createdAt": created_at,
    }

# Determine frontend dist path
container_path = Path("/app/frontend/dist")
local_path = Path(__file__).parent.parent / "frontend" / "dist"

if os.environ.get("FRONTEND_DIST"):
    frontend_dist = os.environ["FRONTEND_DIST"]
elif container_path.exists():
    frontend_dist = str(container_path)
else:
    frontend_dist = str(local_path)

hostname = os.environ.get("WENDY_HOSTNAME", "0.0.0.0")

print(f"Serving frontend from: {frontend_dist}")

# API routes
@app.get("/api/random-car")
async def get_random_car():
    return JSONResponse(random_car())

# Mount static files for assets
assets_path = Path(frontend_dist) / "assets"
if assets_path.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Serve the SPA - return index.html for all routes"""
    file_path = Path(frontend_dist) / full_path
    if file_path.is_file():
        return FileResponse(file_path)
    return FileResponse(f"{frontend_dist}/index.html")

if __name__ == "__main__":
    import uvicorn
    print(f"Server running on http://{hostname}:3002")
    uvicorn.run(app, host="0.0.0.0", port=3002)
