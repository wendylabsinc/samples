#!/usr/bin/env python3
import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Car(BaseModel):
    make: str
    year: int

hostname = (
    os.environ.get("WENDY_DEVICE_HOSTNAME")
    or os.environ.get("WENDY_HOSTNAME")
    or "localhost"
)

@app.on_event("startup")
async def startup_event():
    print(f"Server running on {hostname}:3001", flush=True)

@app.get("/")
async def root():
    print("Received request: GET /", flush=True)
    return "hello-world"

@app.get("/json")
async def get_car():
    print("Received request: GET /json", flush=True)
    return Car(make="Tesla", year=2024)
