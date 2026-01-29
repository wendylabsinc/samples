#!/usr/bin/env python3
import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Car(BaseModel):
    make: str
    year: int

hostname = os.environ.get("WENDY_HOSTNAME", "0.0.0.0")

@app.on_event("startup")
async def startup_event():
    print(f"Server running on http://{hostname}:3001")

@app.get("/")
async def root():
    print("Received request: GET /")
    return "hello-world"

@app.get("/json")
async def get_car():
    print("Received request: GET /json")
    return Car(make="Tesla", year=2024)
