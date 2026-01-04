from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router

app = FastAPI(title="Email App Simulation API")

import os

origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://0.0.0.0:8000",
]

# Add production origins from environment variable
if os.getenv("ALLOWED_ORIGINS"):
    origins.extend(os.getenv("ALLOWED_ORIGINS").split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
@app.head("/")
async def root():
    return {"message": "Email App Simulation API is running"}
