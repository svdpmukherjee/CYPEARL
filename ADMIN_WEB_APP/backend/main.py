"""
CYPEARL Admin Portal API - Combined Phase 1 & Phase 2

Phase 1: Persona Discovery - Clustering and validation
Phase 2: AI Persona Simulation - LLM-based agent testing
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import routers
from api.phase1 import router as phase1_router
from api.phase2 import router as phase2_router

# =============================================================================
# LIFESPAN HANDLER
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    print(" Starting CYPEARL Admin Portal...")
    print("   Phase 1: Persona Discovery - /api/phase1")
    print("   Phase 2: AI Simulation - /api/phase2")
    
    # Initialize Phase 2 state (providers, etc.)
    from api.phase2 import state as phase2_state
    print(f"   Initialized Phase 2 state")
    
    yield
    
    # Shutdown
    print(" Shutting down CYPEARL Admin Portal...")

# =============================================================================
# APP CONFIGURATION
# =============================================================================

app = FastAPI(
    title="CYPEARL Admin Portal API",
    description="AI-Driven Digital Personas for Ethical Phishing Simulation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
origins = [
    "http://localhost:5173",    # Vite default
    "http://localhost:3000",    # Create React App
    "http://127.0.0.1:5173",
    "http://0.0.0.0:5173",
]

# Add any additional origins from environment
if os.getenv("ALLOWED_ORIGINS"):
    origins.extend(os.getenv("ALLOWED_ORIGINS").split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# INCLUDE ROUTERS
# =============================================================================

app.include_router(
    phase1_router, 
    prefix="/api/phase1", 
    tags=["Phase 1: Persona Discovery"]
)

app.include_router(
    phase2_router, 
    prefix="/api/phase2", 
    tags=["Phase 2: AI Simulation"]
)

# =============================================================================
# ROOT ENDPOINTS
# =============================================================================

@app.get("/")
def root():
    """Root endpoint with service info."""
    return {
        "status": "healthy",
        "service": "CYPEARL Admin Portal",
        "version": "1.0.0",
        "phases": {
            "phase1": {
                "name": "Persona Discovery",
                "endpoint": "/api/phase1",
                "status": "active"
            },
            "phase2": {
                "name": "AI Simulation",
                "endpoint": "/api/phase2",
                "status": "active"
            }
        }
    }

@app.get("/health")
def health_check():
    """Lightweight health check endpoint for uptime monitoring (e.g., UptimeRobot)."""
    return {"status": "ok"}

@app.get("/api")
def api_info():
    """API information endpoint."""
    return {
        "name": "CYPEARL API",
        "description": "Cybersecurity Persona Early Assessment & Research Lab",
        "endpoints": {
            "phase1": {
                "base": "/api/phase1",
                "summary": "GET /summary",
                "features": "GET /features",
                "run": "POST /run",
                "optimize": "POST /optimize",
                "industry": "GET /industry-analysis",
                "export": "GET /export/ai-personas"
            },
            "phase2": {
                "base": "/api/phase2",
                "providers": "GET /providers",
                "models": "GET /models",
                "personas": "GET /personas",
                "experiments": "GET /experiments",
                "results": "GET /results/{id}",
                "fidelity": "GET /analysis/{id}/fidelity"
            }
        },
        "docs": "/docs"
    }

# =============================================================================
# RUN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )