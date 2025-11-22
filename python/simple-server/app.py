#!/usr/bin/env python3
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Car(BaseModel):
    make: str
    year: int

@app.get("/")
async def root():
    return "hello-world"

@app.get("/json")
async def get_car():
    return Car(make="Tesla", year=2024)
