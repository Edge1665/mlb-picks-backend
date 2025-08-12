# MLB Picks Backend (FastAPI)

Deploy-friendly FastAPI backend that serves MLB predictions and pushes live updates every 5 minutes.

## Features
- `GET /health` healthcheck
- `GET /predictions?date=YYYY-MM-DD` returns current predictions (stubbed + structure; replace with model)
- `GET /me` returns the authenticated user (if Supabase JWT is passed; optional verification)
- `WebSocket /ws` broadcasts updated predictions every 5 minutes

## Environment
Copy `.env.example` to `.env` and set values.

- `PORT` (default 8000)
- `ALLOWED_ORIGINS` comma-separated (e.g., https://your-frontend.vercel.app,http://localhost:3000)
- `SUPABASE_JWT_AUDIENCE` (optional; typically 'authenticated')
- `SUPABASE_JWKS_URL` (optional; like https://<your-project>.supabase.co/auth/v1/keys)

If you skip JWKS setup, the backend won't strictly verify JWTs (okay for MVP if behind CORS and not sensitive).

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Deploy on Render (free tier)
1. Create a new **Web Service** from this `backend` folder (Python). 
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables (see `.env.example`).
