# GA4 Observable Incident Triage Agent

## Deploy
1. Upload all four files to the root of a public GitHub repository.
2. On Render create a Web Service from that repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   - `LLM_API_KEY`: provider API key (do not commit this)
   - `LLM_BASE_URL`: `https://generativelanguage.googleapis.com/v1beta/openai`
   - `LLM_MODEL`: a model supported by your API key, e.g. `gemini-2.5-flash` if enabled
6. Submit the service base URL only, e.g. `https://your-service.onrender.com`.

## Persistence
SQLite state is stored in `incident_agent.db`. On Render, use a persistent disk and set `DB_PATH` to a file on that disk if supported by your plan.

## Important
This is a starter implementation, not a verified perfect grader solution. Test carefully before spending assignment checks. It uses synthetic incident data only; never submit real production logs or credentials.
