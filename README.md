# Game Recommender Backend (Flask)

This README explains how to set up and run the Flask backend locally, and documents the main API endpoints and data location.

**Requirements**

- **Python:** 3.8+ installed and on your PATH
- **Virtual environment:** recommended (venv or virtualenv)

**Quick Setup (Windows PowerShell)**

1. Create and activate a virtual environment:

```powershell
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
```

If PowerShell blocks scripts, use the CMD-style activate instead:

```powershell
.\\venv\\Scripts\\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Update dependencies:

```powershell
python.exe -m pip install --upgrade pip
```

4. Run the backend:

```powershell
python app.py
```

The app starts in debug mode and listens on http://127.0.0.1:5000 by default.

**Alternative (Windows CMD)**

```cmd
python -m venv venv
venv\\Scripts\\activate.bat
pip install -r requirements.txt
python app.py
```

**API Endpoints**

- **POST /recommend**: Returns recommendations for a selected game.
  - Request JSON: { "selectedGame": "Game Name" }
  - Response: JSON object with `recommendations` (array of games with `name`, `release_date`, `price`, `tags`, `genres`, `popularity`, `reviews`, `score`).
- **GET /games**: Returns an alphabetically sorted list of all game names. Useful for autocomplete.
- **GET /games/search?q=term**: Returns up to 50 game names that contain `term` (case-insensitive).

Example curl (quick test):

```bash
curl -X GET "http://127.0.0.1:5000/games"

curl -X POST "http://127.0.0.1:5000/recommend" -H "Content-Type: application/json" -d '{"selectedGame":"Your Game Name"}'
```

**Data**

- The dataset is located at [flask-backend/data/recommendation_data.csv](flask-backend/data/recommendation_data.csv).
- If you update the CSV, restart the server to reload the data.

**Notes & Implementation details**

- CORS is enabled so a frontend can call these endpoints from another origin.
- The server computes TF-IDF features and cosine similarity at startup (may take a moment on first run depending on dataset size).
- The app uses `app.run(debug=True)` in `app.py` for local development. For production, serve with a WSGI server (Gunicorn, Waitress, etc.) and disable debug mode.

**Troubleshooting**

- If imports fail, ensure the virtual environment is activated and `pip install -r requirements.txt` completed without errors.
- If PowerShell won't run `Activate.ps1`, run PowerShell as Administrator and set `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` or use the `.\\venv\\Scripts\\activate` CMD-style activate.
- If the server errors reading the CSV, confirm the file exists at [flask-backend/data/recommendation_data.csv](flask-backend/data/recommendation_data.csv) and is a valid CSV.

**Next steps (optional)**

- Add a simple health-check endpoint (e.g., `/health`) for readiness checks.
- Add unit tests for endpoints.

---

Generated: concise developer README for local development.
