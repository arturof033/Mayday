
<br />
<div align="center">
  <h3 align="center">Mayday</h3>
  <p align="center">
    Drone mission feasibility and path recommendation console
    <br />
    <br />
    <a href="https://github.com/arturof033/Mayday/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/arturof033/Mayday/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

## Overview

Mayday is a web-based console that helps operators evaluate **drone delivery missions** and visualize routes.
It focuses on mission feasibility under realistic conditions instead of assuming perfect weather and battery.

The system:
- Calculates paths between locations while **avoiding obstacles** (using OpenStreetMap / Overpass).
- Fetches **weather data** (wind speed and direction) from OpenWeatherMap.
- Evaluates whether a mission is **feasible or too risky** using either:
  - A **rule-based model**, or
  - A trained **neural network feasibility model** (via a saved scikit-learn pipeline loaded with `joblib`).
- Persists users, settings, and mission history in **SQLite**.
- Provides a single-page **dashboard UI** with maps, mission history, and basic analytics.

> Note: This is a prototype / academic project. All drone states in the "Drones" page are simulated; no real drones are controlled.

## Tech Stack

- **Backend**
  - Python, Flask
  - Flask-CORS
  - SQLite (via `sqlite3`)
  - JWT auth (`PyJWT`) + password hashing (`bcrypt`)
  - External APIs:
    - OpenStreetMap Overpass (obstacles)
    - OpenWeatherMap (weather)
  - `joblib` to load a pre-trained feasibility model (`feasibility_nn_model.pkl`)

- **Frontend**
  - Static `index.html` (no framework)
  - Vanilla JavaScript
  - Leaflet.js (maps)
  - Chart.js (simple mission analytics)

## Features

- **Authentication**
  - Sign up / login with email + password
  - Passwords stored as bcrypt hashes in SQLite
  - JWT-based stateless authentication on the API

- **Mission simulation**
  - Choose preset locations or geocode free-text addresses
  - Visualize routes on a Leaflet map
  - Automatic or manual wind inputs
  - Mission feasibility evaluation using:
    - Rule-based model **or**
    - Neural network model (configurable in Settings)
  - Additional safety settings:
    - Treat low battery as riskier (aggressive model)
    - Block missions with strong headwind

- **Persistence & history**
  - Missions are saved per user in SQLite
  - History and basic success/fail analytics in the UI

- **Resilience & logging**
  - Retry + error handling for external APIs (Overpass, OpenWeatherMap, Nominatim)
  - Clear frontend messages if APIs fail and the app falls back to a direct path
  - Console logging of path deviation, API timing, and a simple test overview

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/arturof033/Mayday.git
cd Mayday
```

### 2. Create and activate a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux

# on Windows:
# python -m venv venv
# venv\Scripts\activate
```

### 3. Install backend dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `flask`, `flask-cors`
- `numpy`, `requests`
- `bcrypt`, `PyJWT`
- `joblib` (for loading the pre-trained feasibility model)

> You do **not** need scikit-learn to run the app.
> It is only needed if you want to retrain the model using `trainmodel.py`.

### 4. (Optional) Train or retrain the feasibility model

If you want to regenerate `feasibility_nn_model.pkl`:

```bash
python3 trainmodel.py
```

This uses scikit-learn (you must have it installed separately) to train a small neural network
on synthetic data and save it as `feasibility_nn_model.pkl`.  
If the model file is missing or fails to load, the app automatically falls back to the rule-based feasibility model.

### 5. Run the Flask API

```bash
python3 app.py
```

By default this starts Flask in debug mode on:

- Backend API: `http://127.0.0.1:5000/api`

### 6. Open the frontend

The frontend is a single `index.html` file. The simplest options are:

1. **Open directly in your browser**  
   - Open `index.html` in your browser and ensure the backend is running at `http://127.0.0.1:5000`.

2. **Serve via a simple static server** (recommended to avoid CORS quirks)
   - For example, using Python:
     ```bash
     python3 -m http.server 5500
     ```
   - Then open `http://127.0.0.1:5500/index.html` in your browser.

Log in with:
- The default demo user created on first run:
  - Email: `demo@mayday.com`
  - Password: `demo123`
- Or create a new account via the signup form.

## Folder Structure

High-level structure:

```text
Mayday/
├── app.py                 # Flask backend, APIs, pathfinding, feasibility logic
├── index.html             # Single-page frontend (login + console UI)
├── MayDay.css             # (If present) legacy styling (modern UI is inline in index.html)
├── MayDay.js              # Legacy JS (current SPA logic lives in index.html)
├── assets/
│   └── mayday_drone.html  # Additional static asset
├── requirements.txt       # Python dependencies
├── mayday.db              # SQLite database (created at runtime)
├── feasibility_nn_model.pkl  # Saved feasibility model (created by trainmodel.py)
├── trainmodel.py          # Script to train & save the neural network feasibility model
├── test_overview.txt      # Generated test checklist (written on app startup)
└── README.md
```

## Authors

Arturo Flores, <br>
&ensp;&ensp;&ensp;&ensp; 
    Department of Computer Science <br>
&ensp;&ensp;&ensp;&ensp; 
    California State University, Fullerton<br>
&ensp;&ensp;&ensp;&ensp;  
    arturof033@csu.fullerton.edu<br>
    
&ensp;&ensp;&ensp;&ensp; 
Johnny Nguyen, <br>
&ensp;&ensp;&ensp;&ensp; 
    Department of Computer Science <br>
&ensp;&ensp;&ensp;&ensp; 
    California State University, Fullerton<br>
&ensp;&ensp;&ensp;&ensp;  
    johnnynguyenha@csu.fullerton.edu<br>

&ensp;&ensp;&ensp;&ensp; 
Emily Vu, <br>
&ensp;&ensp;&ensp;&ensp; 
    Department of Computer Science <br>
&ensp;&ensp;&ensp;&ensp; 
    California State University, Fullerton<br>
&ensp;&ensp;&ensp;&ensp;  
    emilyvu@csu.fullerton.edu<br>




## Top Contributors

<a href="https://github.com/arturof033/Mayday/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=arturof033/Mayday" />
</a>

