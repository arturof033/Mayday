from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow front-end calls from file:// or localhost


user_db = {
    "demo@mayday.com": {
        "id": 1,
        "email": "demo@mayday.com",
        "name": "Demo Operator",
        "role": "Dispatcher",
        "theme": "dark"
    }
}

locations = [
    {"id": "warehouse_1",        "name": "Warehouse 1",          "lat": 34.0407, "lng": -118.2468},
    {"id": "warehouse_2",        "name": "Warehouse 2",          "lat": 34.0635, "lng": -118.4455},
    {"id": "distribution_1",     "name": "Distribution Site 1",  "lat": 33.8820, "lng": -117.8854},
    {"id": "distribution_2",     "name": "Distribution Site 2",  "lat": 33.6846, "lng": -117.8265},
    {"id": "regional_center_1",  "name": "Regional Center 1",    "lat": 33.7874, "lng": -117.8743},
    {"id": "regional_center_2",  "name": "Regional Center 2",    "lat": 33.9425, "lng": -118.4081},
    {"id": "drop_zone_1",        "name": "Drop Zone 1",          "lat": 33.7676, "lng": -118.1957},
    {"id": "drop_zone_2",        "name": "Drop Zone 2",          "lat": 32.7157, "lng": -117.1611},
]

# Drone fleet with statuses
drones = [
    {
        "id": 101,
        "name": "MX-4 Neptune",
        "model": "Quadcopter",
        "status": "in_flight",
        "battery": 78,
        "current_mission": "LA-OC-3421"
    },
    {
        "id": 102,
        "name": "VX-2 Aurora",
        "model": "VTOL",
        "status": "pending_launch",
        "battery": 92,
        "current_mission": "OC-SD-9012"
    },
    {
        "id": 103,
        "name": "RX-7 Horizon",
        "model": "Hexacopter",
        "status": "idle",
        "battery": 64,
        "current_mission": None
    },
    {
        "id": 104,
        "name": "LX-3 Falcon",
        "model": "Quadcopter",
        "status": "charging",
        "battery": 38,
        "current_mission": None
    },
]

missions = []  # will fill from front-end

# --- Routes ---

@app.route("/api/login", methods=["POST"])
def login():
    """Simple login: accepts any email/password, but if email isn’t in db, creates a user."""
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    user = user_db.get(email)
    if not user:
        user = {
            "id": len(user_db) + 1,
            "email": email,
            "name": email.split("@")[0].title(),
            "role": "Operator",
            "theme": "dark"
        }
        user_db[email] = user

    return jsonify({"token": email, "user": user})


@app.route("/api/me", methods=["GET"])
def me():
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if not token or token not in user_db:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(user_db[token])


@app.route("/api/profile", methods=["POST"])
def update_profile():
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if not token or token not in user_db:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    name = data.get("name")
    theme = data.get("theme")

    user = user_db[token]
    if name:
        user["name"] = name
    if theme in ("dark", "light"):
        user["theme"] = theme

    return jsonify(user)


@app.route("/api/locations", methods=["GET"])
def get_locations():
    return jsonify(locations)


@app.route("/api/drones", methods=["GET"])
def get_drones():
    return jsonify(drones)


@app.route("/api/drones/<int:drone_id>/status", methods=["POST"])
def update_drone_status(drone_id):
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if not token or token not in user_db:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    new_status = data.get("status")
    mission_code = data.get("mission_code")

    valid_statuses = ["idle", "in_flight", "pending_launch", "charging", "maintenance"]

    if new_status not in valid_statuses:
        return jsonify({"error": "Invalid status"}), 400

    drone = next((d for d in drones if d["id"] == drone_id), None)
    if not drone:
        return jsonify({"error": "Drone not found"}), 404

    drone["status"] = new_status

    # simple rules:
    if new_status in ["in_flight", "pending_launch"]:
        drone["current_mission"] = mission_code or drone["current_mission"] or "MANUAL-" + str(drone_id)
    else:
        drone["current_mission"] = None

    if new_status == "in_flight":
        drone["battery"] = max(10, drone["battery"] - 10)
    elif new_status == "charging":
        drone["battery"] = min(100, drone["battery"] + 15)

    return jsonify(drone)


@app.route("/api/missions", methods=["GET"])
def get_missions():
    return jsonify(missions)


@app.route("/api/missions", methods=["POST"])
def add_mission():
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if not token or token not in user_db:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    mission_id = len(missions) + 1
    mission = {
        "id": mission_id,
        "start": data.get("start"),
        "end": data.get("end"),
        "distance_km": data.get("distance_km"),
        "payload_kg": data.get("payload_kg"),
        "battery": data.get("battery"),
        "wind_speed": data.get("wind_speed"),
        "wind_direction": data.get("wind_direction"),
        "result": data.get("result", "pending")
    }
    missions.insert(0, mission)
    return jsonify(mission), 201


if __name__ == "__main__":
    app.run(debug=True)
