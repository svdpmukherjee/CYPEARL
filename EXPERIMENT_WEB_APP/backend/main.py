from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router
from routes.dark_patterns import router as dark_patterns_router
from routes.fake_news import router as fake_news_router
import os

app = FastAPI(title="CYPEARL Experiment API")

# Local development origins (always allowed)
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://0.0.0.0:8000",
]

# Production origins come exclusively from env (set in Render dashboard).
# Comma-separated list, e.g. "https://your-app.vercel.app,https://custom.domain"
if os.getenv("ALLOWED_ORIGINS"):
    origins.extend(
        o.strip() for o in os.getenv("ALLOWED_ORIGINS").split(",") if o.strip()
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(router, prefix="/api")
app.include_router(dark_patterns_router, prefix="/api")
app.include_router(fake_news_router, prefix="/api")

@app.get("/")
@app.head("/")
async def root():
    return {"message": "CYPEARL Experiment API is running", "scenarios": ["phishing", "dark-patterns", "fake-news"]}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
