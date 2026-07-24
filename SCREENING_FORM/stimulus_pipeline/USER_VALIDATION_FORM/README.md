# CYPEARL User Validation Form

A React + MongoDB study app that presents recruited participants with 16
workplace emails for their chosen job cluster and records their realism
judgements and edits. It replaces the single-file
`07_factorial/email_validation_form.html` prototype with a multi-page, per-email
flow that writes responses straight to MongoDB.

## What is here

```
USER_VALIDATION_FORM/
  emails_master.json      the 160 emails (10 clusters x 16), single source of truth
  emails_master.md        human-readable listing of all 160 emails + conditions
  scripts/
    build_master.js       (re)generates emails_master.json/.md from the 07_factorial sources
    seed_emails.py        loads emails_master.json into MongoDB (db user_validation_emails, collection all_emails)
    requirements.txt
  backend/                Express API (serves emails, saves responses)
  frontend/               React (Vite) participant app
```

## Data model (MongoDB, database `user_validation_emails`)

- **all_emails** - the 160 stimuli. Each doc: `cluster`, `recipient_role`, `n`,
  `set`, `src`, `conditions` (dir / urg / frame / action / asset), `sender`
  (name, title, contact, 14-digit mobile), `subject`, `body[5]`, `link`.
- **participants** - one per Prolific ID: `prolificId`, `cluster`,
  `recipientRole`, `personalizationName`, `consent`, `startedAt`, `completedAt`,
  `status`.
- **responses** - one per (`prolificId`, `src`): `realism` (1..5), `offItems[]`,
  `sectionEdits{}`, `comment`, `conditions`.

## Setup

### 0. MongoDB

Use a local server, Atlas, or Docker:

```bash
docker run -d --name cypearl-mongo -p 27017:27017 mongo:7
```

### 1. Seed the emails

```bash
cd scripts
pip install -r requirements.txt
export MONGODB_URI="mongodb://localhost:27017"
python seed_emails.py
```

To regenerate `emails_master.json` from the `07_factorial` sources first:

```bash
node scripts/build_master.js
```

### 2. Backend

```bash
cd backend
cp .env.example .env        # adjust MONGODB_URI / PORT / CORS_ORIGIN if needed
npm install
npm run dev                 # http://localhost:4000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev                 # http://localhost:5173 (proxies /api to :4000)
```

Open http://localhost:5173.

## Participant flow

1. Instructions and consent (must consent to continue).
2. Choose the closest job cluster.
3. Enter Prolific ID.
4. Take on the assigned job role (role + description).
5. Task recap and choose a first name for personalisation (must confirm it is
   not their own real name).
6. 16 email pages, one email each:
   - the email's conditions are described first;
   - the email is shown as a single card in one black ink with a full sender
     signature (name, title, extension, landline, 14-digit fictional mobile);
   - Q1: how realistic is this email for your role? (1 to 5);
   - Q2: if anything felt off, what was it? Selecting any option turns on
     per-section **Edit** buttons in the email so the participant rewrites just
     the parts they would change;
   - an optional free-text comment before moving on.

Responses save to MongoDB as the participant advances, and locally in the
browser so a refresh resumes where they left off.

## Architecture (how it connects to MongoDB)

The browser never talks to MongoDB directly (it cannot safely hold Atlas
credentials, and Atlas is not reachable from browser JS). The backend is the
only component with the Atlas URI:

```
React frontend  --HTTP-->  Express backend  --Atlas URI-->  MongoDB Atlas
```

- The **Atlas connection string goes in `backend/.env`** (`MONGODB_URI=...`).
  The backend reads it via dotenv.
- The **seed script** reads the same URI: it picks up `backend/.env`
  automatically (precedence: shell env, then `scripts/.env`, then
  `backend/.env`).
- The **frontend** only needs the backend running; in dev, Vite proxies `/api`
  to the backend, so there is nothing MongoDB-specific to configure there.

No Docker and no local MongoDB are required. Point everything at your Atlas URI.

## Troubleshooting

- `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate` (Python /
  seed script on macOS): Python does not use the macOS keychain. Install the
  requirements (they include `certifi`); the seed script passes the certifi CA
  bundle for `mongodb+srv://` / TLS URIs, which resolves it.
- `ServerSelectionTimeoutError` against Atlas: allowlist your current IP in
  Atlas under Network Access, and double-check the username/password in the URI.
- `localhost:27017 Connection refused`: the URI was not found. Confirm
  `backend/.env` has `MONGODB_URI=...` with no spaces around `=`.

## Notes

- The 14-digit mobile numbers use the Ofcom fictional 07700 900xxx range,
  extended to 14 digits so they are never dialable. They exist only to make the
  signature look realistic.
- `emails_master.json` is the single source of truth shared by the seed script
  and (via MongoDB) the backend, so the app and the data stay in sync.
