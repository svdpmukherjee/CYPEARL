# CYPEARL — Experiment Web App

A simulated email inbox where participants judge whether workplace emails are phishing or legitimate. This is the main data-collection platform for the CYPEARL phishing-susceptibility study.

**Live demo:** https://cypearl.vercel.app/

## What it is

The second stage of the CYPEARL pipeline. Participants recruited on Prolific are placed in a fictional workplace ("LuxConsultancy") and shown a realistic inbox. They evaluate each email and choose an action (mark safe, report phishing, delete, ignore) while the app silently records how they behave.

Each participant sees **16 emails** created after SCREENING_FORM survey: 8 generic (shared by everyone) and 8 matched to their specific job cluster (out of 10 selected). Together these cover a 2×2×2×2 factorial design (phishing status × sender familiarity × urgency × framing).

## What it measures

- **Behavioral:** action choice, response latency, dwell time, link clicks, link hovers, sender inspection
- **Self-report (per email):** confidence, suspicion, perceived work relevance, free-text reason
- **Pre-survey (individual differences):** cognitive reflection, phishing self-efficacy, knowledge quiz, general trust, impulsivity, personality (BFI-10), prior training and victimization

## What it produces

A rich per-decision dataset (roughly 1000 participants × 16 emails) stored in MongoDB, combining behavioral traces and survey measures. This collected data drives the CDPS behavioral model and, downstream, the AI-persona simulation in the Admin Web App.

> Phishing is the primary scenario, but the same platform also serves two companion deception scenarios (dark patterns, fake news) under the `/api` namespace.

## Tech stack

- **Backend:** Python, FastAPI, MongoDB (async Motor), Docker
- **Frontend:** React, Vite, Tailwind CSS
- **Hosting:** Vercel (frontend), Render (backend), MongoDB Atlas (data)

## Project layout

```
EXPERIMENT_WEB_APP/
├── backend/
│   ├── main.py            # FastAPI app, wires up routers under /api
│   ├── database.py        # async MongoDB connection
│   ├── routes/            # phishing, dark_patterns, fake_news
│   ├── models/            # Pydantic data models per scenario
│   ├── services/
│   ├── scripts/           # stimulus seeding + NIST Phish Scale validation
│   ├── Dockerfile
│   └── requirements.txt
└── frontend/              # React + Vite + Tailwind inbox UI
```

## Running locally

After `git clone`:

**Backend**

```bash
cd EXPERIMENT_WEB_APP/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# create a .env with:
#   MONGO_URL=<your MongoDB connection string>
#   PHISHING_DB_NAME=<db name>            # required (primary scenario)
#   DARKPATTERNS_DB_NAME=<db name>        # required
#   FAKENEWS_DB_NAME=<db name>            # required
#   ALLOWED_ORIGINS=https://your-frontend.vercel.app   # optional, prod CORS
uvicorn main:app --reload          # serves on http://localhost:8000
```

Or with Docker:

```bash
docker build -t cypearl-experiment ./backend
docker run -p 8000:8000 --env-file ./backend/.env cypearl-experiment
```

**Frontend**

```bash
cd EXPERIMENT_WEB_APP/frontend
npm install
npm run dev                         # serves on http://localhost:5173
```

The backend allows `localhost:5173` and the deployed Vercel origin via CORS.

## Where this fits

Stage 2 of 3 in the CYPEARL pipeline: Screening Form → **Experiment Web App** → Admin Web App.
