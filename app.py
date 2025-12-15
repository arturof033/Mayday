from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import requests
import os
from math import radians, cos, sin, sqrt, atan2
import bcrypt
import jwt
import secrets
from datetime import datetime, timedelta
from functools import wraps
import sqlite3
from contextlib import closing
import time
import joblib

app = Flask(__name__)

# Configure CORS to allow all origins and handle preflight requests
CORS(app, 
     resources={
         r"/api/*": {
             "origins": "*",
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization"],
             "supports_credentials": False,
             "max_age": 3600
         }
     },
     supports_credentials=False,
     automatic_options=True)

# database configuration
DATABASE = 'mayday.db'

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

def init_db():
    """Initialize the database with schema"""
    with closing(get_db()) as conn:
        # Users table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'Operator',
                theme TEXT NOT NULL DEFAULT 'dark',
                password_hash TEXT NOT NULL,
                aggressive_battery INTEGER DEFAULT 0,
                strict_wind INTEGER DEFAULT 0,
                feasibility_mode TEXT NOT NULL DEFAULT 'model',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # add settings columns if they don't exist (for existing databases)
        try:
            conn.execute('ALTER TABLE users ADD COLUMN aggressive_battery INTEGER DEFAULT 0')
        except sqlite3.OperationalError:
            pass  # column already exists
        try:
            conn.execute('ALTER TABLE users ADD COLUMN strict_wind INTEGER DEFAULT 0')
        except sqlite3.OperationalError:
            pass  # column already exists
        try:
            conn.execute("ALTER TABLE users ADD COLUMN feasibility_mode TEXT DEFAULT 'model'")
        except sqlite3.OperationalError:
            pass  # column already exists
        
        # missions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS missions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                start TEXT NOT NULL,
                end TEXT NOT NULL,
                distance_km REAL NOT NULL,
                payload_kg REAL NOT NULL,
                battery INTEGER NOT NULL,
                wind_speed REAL NOT NULL,
                wind_direction TEXT NOT NULL,
                result TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        
        # check if demo user exists, if not create it
        cursor = conn.execute('SELECT id FROM users WHERE email = ?', ('demo@mayday.com',))
        if cursor.fetchone() is None:
            demo_password_hash = bcrypt.hashpw("demo123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            conn.execute('''
                INSERT INTO users (email, name, role, theme, password_hash)
                VALUES (?, ?, ?, ?, ?)
            ''', ('demo@mayday.com', 'Demo Operator', 'Dispatcher', 'dark', demo_password_hash))
            conn.commit()

def get_user_by_email(email):
    """Get user by email from database"""
    with closing(get_db()) as conn:
        cursor = conn.execute(
            'SELECT id, email, name, role, theme, password_hash, aggressive_battery, strict_wind, feasibility_mode FROM users WHERE email = ?',
            (email,)
        )
        row = cursor.fetchone()
        if row:
            return {
                'id': row['id'],
                'email': row['email'],
                'name': row['name'],
                'role': row['role'],
                'theme': row['theme'],
                'password_hash': row['password_hash'],
                'aggressive_battery': bool(row['aggressive_battery']),
                'strict_wind': bool(row['strict_wind']),
                'feasibility_mode': row['feasibility_mode'] if 'feasibility_mode' in row.keys() else 'model'
            }
        return None

def create_user(email, name, role, theme, password_hash):
    """Create a new user in the database"""
    with closing(get_db()) as conn:
        cursor = conn.execute('''
            INSERT INTO users (email, name, role, theme, password_hash, aggressive_battery, strict_wind, feasibility_mode)
            VALUES (?, ?, ?, ?, ?, 0, 0, 'model')
        ''', (email, name, role, theme, password_hash))
        conn.commit()
        user_id = cursor.lastrowid
        return {
            'id': user_id,
            'email': email,
            'name': name,
            'role': role,
            'theme': theme,
            'aggressive_battery': False,
            'strict_wind': False,
            'feasibility_mode': 'model'
        }

def update_user(email, name=None, theme=None, aggressive_battery=None, strict_wind=None, feasibility_mode=None):
    """Update user information in the database"""
    with closing(get_db()) as conn:
        if name is not None:
            conn.execute('UPDATE users SET name = ? WHERE email = ?', (name, email))
        if theme is not None:
            conn.execute('UPDATE users SET theme = ? WHERE email = ?', (theme, email))
        if aggressive_battery is not None:
            conn.execute('UPDATE users SET aggressive_battery = ? WHERE email = ?', (1 if aggressive_battery else 0, email))
        if strict_wind is not None:
            conn.execute('UPDATE users SET strict_wind = ? WHERE email = ?', (1 if strict_wind else 0, email))
        if feasibility_mode is not None:
            conn.execute(
                'UPDATE users SET feasibility_mode = ? WHERE email = ?',
                (feasibility_mode if feasibility_mode in ("rule", "model") else "model", email)
            )
        conn.commit()
        return get_user_by_email(email)

def create_mission(user_id, start, end, distance_km, payload_kg, battery, wind_speed, wind_direction, result):
    """Create a new mission in the database"""
    with closing(get_db()) as conn:
        cursor = conn.execute('''
            INSERT INTO missions (user_id, start, end, distance_km, payload_kg, battery, wind_speed, wind_direction, result)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, start, end, distance_km, payload_kg, battery, wind_speed, wind_direction, result))
        conn.commit()
        mission_id = cursor.lastrowid
        return {
            'id': mission_id,
            'user_id': user_id,
            'start': start,
            'end': end,
            'distance_km': distance_km,
            'payload_kg': payload_kg,
            'battery': battery,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'result': result
        }

def get_missions_by_user(user_id):
    """Get all missions for a specific user, ordered by most recent first"""
    with closing(get_db()) as conn:
        cursor = conn.execute('''
            SELECT id, user_id, start, end, distance_km, payload_kg, battery, wind_speed, wind_direction, result, created_at
            FROM missions
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        missions = []
        for row in cursor.fetchall():
            missions.append({
                'id': row['id'],
                'user_id': row['user_id'],
                'start': row['start'],
                'end': row['end'],
                'distance_km': row['distance_km'],
                'payload_kg': row['payload_kg'],
                'battery': row['battery'],
                'wind_speed': row['wind_speed'],
                'wind_direction': row['wind_direction'],
                'result': row['result'],
                'created_at': row['created_at']
            })
        return missions

def get_all_missions():
    """Get all missions from all users, ordered by most recent first"""
    with closing(get_db()) as conn:
        cursor = conn.execute('''
            SELECT id, user_id, start, end, distance_km, payload_kg, battery, wind_speed, wind_direction, result, created_at
            FROM missions
            ORDER BY created_at DESC
        ''')
        missions = []
        for row in cursor.fetchall():
            missions.append({
                'id': row['id'],
                'user_id': row['user_id'],
                'start': row['start'],
                'end': row['end'],
                'distance_km': row['distance_km'],
                'payload_kg': row['payload_kg'],
                'battery': row['battery'],
                'wind_speed': row['wind_speed'],
                'wind_direction': row['wind_direction'],
                'result': row['result'],
                'created_at': row['created_at']
            })
        return missions

# initialize database and ml model on startup
init_db()

# --- Feasibility model loading ---

FEASIBILITY_MODEL = None
FEASIBILITY_MODEL_LOADED = False


def load_feasibility_model():
    """Load the scikit-learn feasibility model if available."""
    global FEASIBILITY_MODEL, FEASIBILITY_MODEL_LOADED
    model_path = os.path.join(os.path.dirname(__file__), "feasibility_nn_model.pkl")
    try:
        if os.path.exists(model_path):
            start_time = time.perf_counter()
            FEASIBILITY_MODEL = joblib.load(model_path)
            FEASIBILITY_MODEL_LOADED = True
            elapsed = time.perf_counter() - start_time
            print(f"loaded feasibility model from {model_path} in {elapsed:.2f}s")
        else:
            print(f"feasibility model file not found at {model_path}; using rule-based feasibility.")
            FEASIBILITY_MODEL = None
            FEASIBILITY_MODEL_LOADED = False
    except Exception as e:
        print(f"could not load feasibility model: {e}")
        import traceback
        print(traceback.format_exc())
        FEASIBILITY_MODEL = None
        FEASIBILITY_MODEL_LOADED = False


load_feasibility_model()

# JWT Configuration
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', secrets.token_urlsafe(32))
JWT_ALGORITHM = 'HS256'
JWT_EXPIRATION_HOURS = 24

def haversine_distance(coord1, coord2):
    R = 6371
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def get_weather_for_path(path):
    """fetch weather data for the path and calculate average wind speed and direction relative to path"""
    if not path or len(path) < 2:
        return {
            "wind_speed": 12,
            "wind_direction": "calm",
            "weather_api_failed": False,
            "weather_error": None
        }
    
    # get weather for midpoint of path
    mid_idx = len(path) // 2
    lat = path[mid_idx][0]
    lon = path[mid_idx][1]
    
    # calculate path direction (bearing from start to end)
    start = path[0]
    end = path[-1]
    
    # convert lat/lon to radians
    lat1_rad = radians(start[0])
    lat2_rad = radians(end[0])
    dlon = radians(end[1] - start[1])
    
    # calculate bearing (direction of travel)
    y = sin(dlon) * cos(lat2_rad)
    x = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dlon)
    path_bearing = (atan2(y, x) * 180 / 3.14159265359 + 360) % 360
    
    try:
        api_key = "333b93885e9d224a90c1e0e7e1630e75"
        if api_key:
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            start_time = time.perf_counter()
            response = requests.get(weather_url, timeout=5)
            elapsed = time.perf_counter() - start_time
            print(f"weather api call took {elapsed:.2f}s (status {response.status_code})")
            
            if response.status_code == 200:
                data = response.json()
                wind_data = data.get('wind', {})
                wind_speed_ms = wind_data.get('speed', 0)  # m/s
                wind_deg = wind_data.get('deg', None)  # degrees
                
                # convert m/s to km/h
                wind_speed_kmh = wind_speed_ms * 3.6
                
                # calculate relative wind direction if we have wind direction data
                if wind_deg is not None:
                    wind_relative = (wind_deg - path_bearing + 360) % 360
                    
                    if wind_speed_kmh < 5:
                        direction = "calm"
                    elif 315 <= wind_relative or wind_relative < 45:
                        direction = "headwind"
                    elif 135 <= wind_relative < 225:
                        direction = "tailwind"
                    else:
                        direction = "crosswind"
                else:
                    # no wind direction data
                    if wind_speed_kmh < 5:
                        direction = "calm"
                    else:
                        direction = "crosswind"
                
                print(f"weather fetched: speed={wind_speed_kmh:.1f} km/h, direction={direction}, deg={wind_deg}")
                return {
                    "wind_speed": round(wind_speed_kmh),
                    "wind_direction": direction,
                    "weather_api_failed": False,
                    "weather_error": None
                }
            else:
                print(f"weather api returned status {response.status_code}: {response.text[:200]}")
                return {
                    "wind_speed": 12,
                    "wind_direction": "calm",
                    "weather_api_failed": True,
                    "weather_error": f"weather api returned status {response.status_code}"
                }
    except Exception as e:
        print(f"error fetching weather: {e}")
        import traceback
        print(traceback.format_exc())
    
    # fallback: return default values
    return {
        "wind_speed": 12,
        "wind_direction": "calm",
        "weather_api_failed": True,
        "weather_error": "weather api failed, using default wind values"
    }

def get_osm_obstacles(start, end, buffer=0.05):
    min_lat = min(start[0], end[0]) - buffer
    max_lat = max(start[0], end[0]) + buffer
    min_lon = min(start[1], end[1]) - buffer
    max_lon = max(start[1], end[1]) + buffer
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    query = f"""
    [out:json][timeout:30];
    (
      way["building"="yes"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["building"="commercial"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["building"="industrial"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["aeroway"="aerodrome"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["aeroway"="runway"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["landuse"="military"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["man_made"="tower"]({min_lat},{min_lon},{max_lat},{max_lon});
      node["man_made"="tower"]({min_lat},{min_lon},{max_lat},{max_lon});
      node["aeroway"="aerodrome"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out geom;
    """
    
    # retry logic for api calls
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()
            response = requests.post(overpass_url, data=query, timeout=25)
            elapsed = time.perf_counter() - start_time
            print(f"overpass api attempt {attempt + 1} took {elapsed:.2f}s (status {response.status_code})")
            if response.status_code == 200:
                data = response.json()
                obstacles = []
                
                for element in data.get('elements', []):
                    if element['type'] == 'way' and 'geometry' in element:
                        coords = [[node['lat'], node['lon']] for node in element['geometry']]
                        if coords:
                            center_lat = sum(c[0] for c in coords) / len(coords)
                            center_lon = sum(c[1] for c in coords) / len(coords)
                            size = 0.008
                            if 'tags' in element:
                                tags = element['tags']
                                if tags.get('aeroway') in ['aerodrome', 'runway']:
                                    size = 0.03
                                elif tags.get('landuse') == 'military':
                                    size = 0.02
                                elif tags.get('building') in ['commercial', 'retail', 'mall']:
                                    size = 0.01
                                elif tags.get('building') == 'industrial':
                                    size = 0.008
                            
                            obstacle_type = tags.get('aeroway') or tags.get('landuse') or tags.get('building', 'obstacle')
                            obstacles.append({
                                'center': [center_lat, center_lon],
                                'type': obstacle_type,
                                'coords': coords,
                                'radius': size
                            })
                    elif element['type'] == 'node':
                        tags = element.get('tags', {})
                        size = 0.03 if tags.get('aeroway') == 'aerodrome' else 0.008
                        obstacle_type = tags.get('aeroway') or tags.get('man_made') or 'tower'
                        obstacles.append({
                            'center': [element['lat'], element['lon']],
                            'type': obstacle_type,
                            'coords': [[element['lat'], element['lon']]],
                            'radius': size
                        })
                
                return obstacles, None
            elif response.status_code == 504:
                last_error = f"API timeout (504) - server is overloaded"
                if attempt < max_retries - 1:
                    print(f"Overpass API timeout (504), retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2 * (attempt + 1))  # exponential backoff
                    continue
            elif response.status_code == 429:
                last_error = f"API rate limit exceeded (429)"
                if attempt < max_retries - 1:
                    print(f"Overpass API rate limited (429), retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(5 * (attempt + 1))  # longer wait for rate limits
                    continue
            else:
                last_error = f"API returned status {response.status_code}"
                if attempt < max_retries - 1:
                    print(f"Overpass API returned status {response.status_code}, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(2)
                    continue
        except requests.exceptions.Timeout:
            last_error = "Request timeout - API took too long to respond"
            if attempt < max_retries - 1:
                print(f"Request timeout, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(2 * (attempt + 1))
                continue
        except requests.exceptions.ConnectionError:
            last_error = "Connection error - could not reach API server"
            if attempt < max_retries - 1:
                print(f"Connection error, retrying... (attempt {attempt + 1}/{max_retries})")
                time.sleep(2 * (attempt + 1))
                continue
        except Exception as e:
            last_error = f"Error: {str(e)}"
            if attempt < max_retries - 1:
                print(f"Error fetching obstacles (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                time.sleep(2 * (attempt + 1))
                continue
    
    # all retries failed
    error_msg = f"Failed to fetch obstacle data after {max_retries} attempts. {last_error}"
    print(error_msg)
    return [], error_msg

def find_optimal_drone_path(start, end, obstacles):
    # filter major obstacles first
    major_obstacles = [
        obs for obs in obstacles 
        if obs['type'] in ['aerodrome', 'runway', 'military', 'tower']
    ]
    
    # if too many obstacles, prioritize ones closest to the flight path
    if len(major_obstacles) > 100:
        start_np = np.array(start)
        end_np = np.array(end)
        
        # calculate distance from each obstacle to the direct line between start/end
        def distance_to_path(obs):
            obs_pos = np.array(obs['center'])
            path_vec = end_np - start_np
            path_length = np.linalg.norm(path_vec)
            
            if path_length < 0.0001:
                return np.linalg.norm(obs_pos - start_np)
            
            path_unit = path_vec / path_length
            to_obs = obs_pos - start_np
            projection = np.dot(to_obs, path_unit)
            projection = max(0, min(path_length, projection))
            closest_point = start_np + path_unit * projection
            
            return np.linalg.norm(obs_pos - closest_point)
        
        # sort by distance to path, keep closest 100
        major_obstacles.sort(key=distance_to_path)
        major_obstacles = major_obstacles[:100]
    
    # check if direct path is blocked
    start_vec = np.array(start)
    end_vec = np.array(end)
    route_vec = end_vec - start_vec
    route_length = np.linalg.norm(route_vec)
    
    if route_length == 0:
        return [start, end], []
    
    route_unit = route_vec / route_length
    
    # check for obstacles blocking the direct path
    blocking_obstacles = []
    for obs in major_obstacles:
        obs_center = np.array(obs['center'])
        obs_radius = min(obs['radius'], 0.01)
        to_obs = obs_center - start_vec
        projection = np.dot(to_obs, route_vec) / (route_length * route_length)
        projection = max(0, min(1, projection))
        closest_point = start_vec + route_vec * projection
        dist_to_route_km = np.linalg.norm(obs_center - closest_point) * 111
        
        # use larger detection threshold for airports to avoid them earlier
        threshold_multiplier = 1.8 if obs['type'] in ['aerodrome', 'runway'] else 1.2
        if dist_to_route_km < obs_radius * 111 * threshold_multiplier:
            blocking_obstacles.append({
                'obstacle': obs,
                'projection': projection,
                'distance': dist_to_route_km
            })
    
    if not blocking_obstacles:
        return [start, end], []
    
    blocking_obstacles.sort(key=lambda x: x['projection'])
    
    # helper function to check if a path segment intersects an obstacle
    def segment_intersects_obstacle(seg_start, seg_end, obstacle):
        seg_start_vec = np.array(seg_start)
        seg_end_vec = np.array(seg_end)
        seg_vec = seg_end_vec - seg_start_vec
        seg_length = np.linalg.norm(seg_vec)
        if seg_length == 0:
            return False
        
        seg_unit = seg_vec / seg_length
        obs_center = np.array(obstacle['center'])
        obs_radius = min(obstacle['radius'], 0.01)
        
        to_obs = obs_center - seg_start_vec
        projection = np.dot(to_obs, seg_unit)
        projection = max(0, min(seg_length, projection))
        closest_point = seg_start_vec + seg_unit * projection
        dist_to_obs_km = np.linalg.norm(obs_center - closest_point) * 111
        
        threshold_multiplier = 1.8 if obstacle['type'] in ['aerodrome', 'runway'] else (1.5 if obstacle['type'] == 'military' else 1.2)
        return dist_to_obs_km < obs_radius * 111 * threshold_multiplier
    
    path = [start]
    avoided_obstacles = []
    current_pos = np.array(start)
    
    for block in blocking_obstacles:
        obs = block['obstacle']
        obs_center = np.array(obs['center'])
        obs_radius = min(obs['radius'], 0.01)
        
        # check if we already avoid this obstacle
        if obs in avoided_obstacles:
            continue
        
        # check if path from current position to end would intersect this obstacle
        # OR if the obstacle is between current position and end (using projection)
        to_obs_from_current = obs_center - current_pos
        to_end_from_current = end_vec - current_pos
        to_end_norm = np.linalg.norm(to_end_from_current)
        
        if to_end_norm > 0:
            projection_on_path = np.dot(to_obs_from_current, to_end_from_current) / (to_end_norm * to_end_norm)
            # obstacle is between current and end if projection is between 0 and 1
            obstacle_in_path = 0 <= projection_on_path <= 1
        else:
            obstacle_in_path = False
        
        if not segment_intersects_obstacle(current_pos.tolist(), end, obs) and not obstacle_in_path:
            continue
        
        # find the best waypoint to avoid this obstacle
        projection = block['projection']
        closest_point = start_vec + route_vec * projection
        perp_vec = obs_center - closest_point
        perp_norm = np.linalg.norm(perp_vec)
        
        if perp_norm > 0:
            perp = perp_vec / perp_norm
        else:
            perp = np.array([-route_unit[1], route_unit[0]])
        
        # choose side that's closer to goal
        goal_vec = end_vec - obs_center
        perp_dot = np.dot(goal_vec, perp)
        if perp_dot < 0:
            perp = -perp
        
        # place waypoint at safe distance
        if obs['type'] in ['aerodrome', 'runway']:
            safe_distance = obs_radius * 4.0
        elif obs['type'] == 'military':
            safe_distance = obs_radius * 3.5
        else:
            safe_distance = obs_radius * 3.0
        
        waypoint = obs_center + perp * safe_distance
        
        # verify waypoint avoids this obstacle
        if not segment_intersects_obstacle(current_pos.tolist(), waypoint.tolist(), obs) and \
           not segment_intersects_obstacle(waypoint.tolist(), end, obs):
            path.append(waypoint.tolist())
            current_pos = waypoint
            avoided_obstacles.append(obs)
    
    path.append(end)
    
    # final check: verify the path doesn't go through any obstacles
    final_path = [path[0]]
    for i in range(1, len(path)):
        segment_start = final_path[-1]
        segment_end = path[i]
        
        # check if this segment intersects any major obstacle
        segment_has_obstacle = False
        for obs in major_obstacles:
            if segment_intersects_obstacle(segment_start, segment_end, obs):
                segment_has_obstacle = True
                break
        
        if not segment_has_obstacle:
            final_path.append(segment_end)
        else:
            # need to add intermediate waypoint
            mid_point = [(segment_start[0] + segment_end[0]) / 2, (segment_start[1] + segment_end[1]) / 2]
            final_path.append(mid_point)
            final_path.append(segment_end)
    
    return final_path, avoided_obstacles

def simulate_drone_on_path(path, steps=50):
    if len(path) < 2:
        return path
    
    distances = [0]
    for i in range(1, len(path)):
        dist = np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
        distances.append(distances[-1] + dist)
    
    total_distance = distances[-1]
    
    positions = []
    for i in range(steps + 1):
        target_dist = (i / steps) * total_distance
        
        for j in range(len(distances) - 1):
            if distances[j] <= target_dist <= distances[j + 1]:
                segment_progress = (target_dist - distances[j]) / (distances[j + 1] - distances[j])
                pos = np.array(path[j]) + segment_progress * (np.array(path[j + 1]) - np.array(path[j]))
                positions.append(pos.tolist())
                break
    
    return positions


# --- Feasibility evaluation (rule + model) ---

WIND_DIR_MAP = {
    "tailwind": -1.0,
    "calm": 0.0,
    "crosswind": 1.0,
    "headwind": 2.0,
}


def evaluate_feasibility_rule(distance_km, payload_kg, battery_percent, wind_speed, wind_direction,
                              aggressive_battery=False, strict_wind=False):
    """
    Mirror the frontend rule-based feasibility check so we can fall back to it or use it directly.
    """
    cost = distance_km * 0.5 + payload_kg * 0.8 + wind_speed * 0.25

    if aggressive_battery:
        cost *= 1.15

    if strict_wind and wind_direction == "headwind" and wind_speed > 20:
        return {
            "success": False,
            "reason": "strict wind rule: headwind above 20 km/h blocks the mission.",
        }

    margin = battery_percent - cost
    success = margin > 12 and battery_percent > 35

    if success:
        reason = "battery margin is above the safety threshold. mission is likely to succeed."
    else:
        reason = "battery margin or health is too low. the drone might not make a safe return."

    return {"success": success, "reason": reason}


def evaluate_feasibility(distance_km, payload_kg, battery_percent, wind_speed, wind_direction,
                         aggressive_battery=False, strict_wind=False, feasibility_mode="model"):
    """
    Top-level feasibility evaluation that chooses between the neural network model
    and the rule-based system, with automatic fallback to the rule if the model
    is unavailable or raises an error.
    """
    mode = (feasibility_mode or "model").lower()
    use_model = mode == "model" and FEASIBILITY_MODEL is not None and FEASIBILITY_MODEL_LOADED

    if use_model:
        # apply strict_wind rule as a hard override before model evaluation
        if strict_wind and wind_direction == "headwind" and wind_speed > 20:
            return {
                "success": False,
                "reason": "strict wind rule: headwind above 20 km/h blocks the mission (overrides model).",
                "mode_used": "model",
            }
        
        try:
            # apply aggressive_battery setting by reducing effective battery for model input
            effective_battery = battery_percent
            if aggressive_battery:
                # reduce battery by 15% to make model more conservative
                effective_battery = battery_percent * 0.85
            
            # encode inputs for the scikit-learn pipeline
            wind_dir_encoded = WIND_DIR_MAP.get(wind_direction or "calm", 0.0)
            X = np.array([[float(distance_km),
                           float(payload_kg),
                           float(wind_speed),
                           float(wind_dir_encoded),
                           float(effective_battery)]])

            proba = None
            if hasattr(FEASIBILITY_MODEL, "predict_proba"):
                proba = FEASIBILITY_MODEL.predict_proba(X)[0][1]
                success = proba >= 0.5
            else:
                pred = FEASIBILITY_MODEL.predict(X)[0]
                success = bool(pred)

            if success:
                reason = "neural network feasibility model predicts this mission is likely to succeed."
            else:
                reason = "neural network feasibility model predicts high risk for this mission."

            if proba is not None:
                reason += f" model confidence (success class): {proba:.2f}."
            
            # note if settings affected the evaluation
            if aggressive_battery:
                reason += " aggressive battery setting applied (battery reduced by 15% for model input)."

            return {
                "success": success,
                "reason": reason,
                "mode_used": "model",
            }
        except Exception as e:
            print(f"error during feasibility model evaluation, falling back to rule-based logic: {e}")
            import traceback
            print(traceback.format_exc())

    # default / fallback: rule-based evaluation
    rule_result = evaluate_feasibility_rule(
        distance_km,
        payload_kg,
        battery_percent,
        wind_speed,
        wind_direction,
        aggressive_battery=aggressive_battery,
        strict_wind=strict_wind,
    )
    rule_result["mode_used"] = "rule"
    return rule_result


# user database is now stored in sqlite (see get_user_by_email, create_user, update_user functions above)

# Helper functions for authentication
def hash_password(password):
    """Hash a password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, password_hash):
    """Verify a password against its hash"""
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))

def generate_token(user):
    """Generate JWT token for user"""
    payload = {
        'user_id': user['id'],
        'email': user['email'],
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_token(token):
    """Verify JWT token and return user email"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload.get('email')
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
        if not token:
            return jsonify({"error": "No token provided"}), 401
        
        email = verify_token(token)
        if not email:
            return jsonify({"error": "Invalid or expired token"}), 401
        
        user = get_user_by_email(email)
        if not user:
            return jsonify({"error": "Invalid or expired token"}), 401
        
        request.current_user = user
        return f(*args, **kwargs)
    return decorated_function

locations = [
    {"id": "warehouse_1",        "name": "Warehouse 1",          "lat": 34.0407, "lng": -118.2468},
    {"id": "warehouse_2",        "name": "Warehouse 2",          "lat": 34.0635, "lng": -118.4455},
    {"id": "distribution_1",     "name": "Distribution Site 1",  "lat": 33.8820, "lng": -117.8854},
    {"id": "distribution_2",     "name": "Distribution Site 2",  "lat": 33.6846, "lng": -117.8265},
    {"id": "regional_center_1",  "name": "Regional Center 1",    "lat": 33.7874, "lng": -117.8743},
    {"id": "regional_center_2",  "name": "Regional Center 2",    "lat": 33.9425, "lng": -118.4081},
    {"id": "drop_zone_1",        "name": "Drop Zone 1",          "lat": 33.7676, "lng": -118.1957},
    {"id": "drop_zone_2",        "name": "Drop Zone 2",          "lat": 32.732157237315846, "lng":  -117.1436561915149},
    {"id": "santa_monica",       "name": "Santa Monica",         "lat": 34.007472980651265,  "lng":  -118.47696175080151},
    {"id": "downtown_la",        "name": "Downtown LA",         "lat": 34.052,  "lng": -118.243},
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

# missions are now stored in sqlite database (see create_mission, get_missions_by_user, get_all_missions functions)

# --- Routes ---

@app.route("/api/signup", methods=["POST"])
def signup():
    """Create a new user account with email and password"""
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    name = (data.get("name") or "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    # validate email format
    if "@" not in email or "." not in email.split("@")[1]:
        return jsonify({"error": "Invalid email format."}), 400

    # validate password strength
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters long."}), 400

    # check if user already exists
    existing_user = get_user_by_email(email)
    if existing_user:
        return jsonify({"error": "Email already registered. Please login instead."}), 400

    # create new user
    password_hash = hash_password(password)
    user = create_user(
        email=email,
        name=name or email.split("@")[0].title(),
        role="Operator",
        theme="dark",
        password_hash=password_hash
    )

    # generate jwt token
    token = generate_token(user)

    # return user data (without password hash)
    user_response = {k: v for k, v in user.items() if k != "password_hash"}
    return jsonify({"token": token, "user": user_response}), 201


@app.route("/api/login", methods=["POST"])
def login():
    """Authenticate user with email and password"""
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    user = get_user_by_email(email)
    if not user:
        return jsonify({"error": "Invalid email or password."}), 401

    # Verify password
    if not verify_password(password, user.get("password_hash", "")):
        return jsonify({"error": "Invalid email or password."}), 401

    # Generate JWT token
    token = generate_token(user)

    # Return user data (without password hash)
    user_response = {k: v for k, v in user.items() if k != "password_hash"}
    return jsonify({"token": token, "user": user_response})


@app.route("/api/me", methods=["GET"])
@require_auth
def me():
    user = request.current_user
    user_response = {k: v for k, v in user.items() if k != "password_hash"}
    return jsonify(user_response)


@app.route("/api/profile", methods=["POST"])
@require_auth
def update_profile():
    """Update user profile (name and theme)"""
    data = request.get_json() or {}
    name = data.get("name")
    theme = data.get("theme")

    user = request.current_user
    email = user["email"]
    
    # validate name is provided and not empty
    if name is not None and len(name.strip()) == 0:
        return jsonify({"error": "Name cannot be empty"}), 400
    
    # update user in database
    updated_user = update_user(email, name=name.strip() if name else None, theme=theme if theme in ("dark", "light") else None)
    
    user_response = {k: v for k, v in updated_user.items() if k != "password_hash"}
    return jsonify(user_response)


@app.route("/api/settings", methods=["GET"])
@require_auth
def get_settings():
    """Get user settings"""
    user = request.current_user
    return jsonify({
        "aggressive_battery": user.get("aggressive_battery", False),
        "strict_wind": user.get("strict_wind", False),
        "feasibility_mode": user.get("feasibility_mode", "model")
    })


@app.route("/api/settings", methods=["POST"])
@require_auth
def update_settings():
    """Update user settings"""
    data = request.get_json() or {}
    aggressive_battery = data.get("aggressive_battery")
    strict_wind = data.get("strict_wind")
    feasibility_mode = data.get("feasibility_mode")

    user = request.current_user
    email = user["email"]
    
    # Update settings in database
    updated_user = update_user(
        email,
        aggressive_battery=aggressive_battery if aggressive_battery is not None else None,
        strict_wind=strict_wind if strict_wind is not None else None,
        feasibility_mode=feasibility_mode if feasibility_mode in ("rule", "model") else None
    )
    
    return jsonify({
        "aggressive_battery": updated_user.get("aggressive_battery", False),
        "strict_wind": updated_user.get("strict_wind", False),
        "feasibility_mode": updated_user.get("feasibility_mode", "model")
    })


@app.route("/api/locations", methods=["GET"])
def get_locations():
    return jsonify(locations)


@app.route("/api/drones", methods=["GET"])
def get_drones():
    return jsonify(drones)


@app.route("/api/drones/<int:drone_id>/status", methods=["POST"])
@require_auth
def update_drone_status(drone_id):

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
@require_auth
def get_missions():
    """Get all missions for the authenticated user"""
    user = request.current_user
    user_missions = get_missions_by_user(user['id'])
    return jsonify(user_missions)


@app.route("/api/missions", methods=["POST"])
@require_auth
def add_mission():
    """Create a new mission for the authenticated user"""
    user = request.current_user
    data = request.get_json() or {}
    
    # validate required fields
    required_fields = ['start', 'end', 'distance_km', 'payload_kg', 'battery', 'wind_speed', 'wind_direction']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    # create mission in database
    mission = create_mission(
        user_id=user['id'],
        start=data.get("start"),
        end=data.get("end"),
        distance_km=float(data.get("distance_km")),
        payload_kg=float(data.get("payload_kg")),
        battery=int(data.get("battery")),
        wind_speed=float(data.get("wind_speed")),
        wind_direction=data.get("wind_direction"),
        result=data.get("result", "pending")
    )
    
    return jsonify(mission), 201


@app.route("/api/feasibility", methods=["POST"])
@require_auth
def api_feasibility():
    """Evaluate mission feasibility using either the ML model or the rule-based logic."""
    data = request.get_json() or {}
    try:
        distance_km = float(data.get("distance_km"))
        payload_kg = float(data.get("payload_kg"))
        battery_percent = float(data.get("battery_percent"))
        wind_speed = float(data.get("wind_speed"))
        wind_direction = (data.get("wind_direction") or "calm").lower()
    except (TypeError, ValueError):
        return jsonify({"error": "invalid or missing numeric fields for feasibility evaluation"}), 400

    user = request.current_user
    aggressive_battery = user.get("aggressive_battery", False)
    strict_wind = user.get("strict_wind", False)
    feasibility_mode = user.get("feasibility_mode", "model")

    result = evaluate_feasibility(
        distance_km,
        payload_kg,
        battery_percent,
        wind_speed,
        wind_direction,
        aggressive_battery=aggressive_battery,
        strict_wind=strict_wind,
        feasibility_mode=feasibility_mode,
    )
    return jsonify(result)


@app.route("/api/path", methods=["GET"])
def get_path():
    try:
        start_lat = request.args.get("start_lat", type=float)
        start_lng = request.args.get("start_lng", type=float)
        end_lat = request.args.get("end_lat", type=float)
        end_lng = request.args.get("end_lng", type=float)
        
        if not all([start_lat, start_lng, end_lat, end_lng]):
            return jsonify({"error": "start and end coordinates are required"}), 400
        
        start = [start_lat, start_lng]
        end = [end_lat, end_lng]
        
        # check if weather api should be used
        use_weather = request.args.get("use_weather", "true").lower() == "true"
        
        obstacles, api_error = get_osm_obstacles(start, end)
        
        if api_error:
            return jsonify({
                "error": api_error,
                "path": [start, end],
                "distance_km": round(haversine_distance(start, end), 2),
                "api_failed": True
            }), 503
        
        print(f"found {len(obstacles)} obstacles")
        
        path_calc_start = time.perf_counter()
        if obstacles:
            route, avoided_obstacles = find_optimal_drone_path(start, end, obstacles)
            print(f"path calculated with {len(route)} waypoints, avoided {len(avoided_obstacles)} obstacles")
        else:
            route, avoided_obstacles = [start, end], []
            print("no obstacles found, using direct path")
        
        route = simulate_drone_on_path(route, steps=50)
        path_calc_elapsed = time.perf_counter() - path_calc_start
        print(f"interpolated path has {len(route)} points")
        print(f"path calculation (waypoints + interpolation) took {path_calc_elapsed:.2f}s")
        
        total_distance = sum(haversine_distance(route[i-1], route[i]) for i in range(1, len(route)))
        direct_distance = haversine_distance(start, end)
        deviation_pct = ((total_distance - direct_distance) / direct_distance * 100) if direct_distance > 0 else 0.0
        
        print(
            f"distance check -> direct: {direct_distance:.2f} km, "
            f"path: {total_distance:.2f} km, "
            f"deviation: {deviation_pct:.1f}%"
        )
        
        if deviation_pct > 30:
            print(f"warning: path deviation is {deviation_pct:.1f}% longer than direct route")
        
        # calculate average wind along the path only if use_weather is true
        response_data = {
            "path": route,
            "distance_km": round(total_distance, 2),
            "direct_distance_km": round(direct_distance, 2),
            "deviation_pct": round(deviation_pct, 2),
            "avoided_obstacles": len(avoided_obstacles),
            "obstacles_found": len(obstacles),
            "waypoints": len(route),
            "api_failed": False
        }
        
        if use_weather:
            wind_data = get_weather_for_path(route)
            response_data["wind_speed"] = wind_data.get("wind_speed", 12)
            response_data["wind_direction"] = wind_data.get("wind_direction", "calm")
            response_data["weather_api_failed"] = wind_data.get("weather_api_failed", False)
            response_data["weather_error"] = wind_data.get("weather_error")
        else:
            # dont include wind data when manual mode is enabled
            response_data["wind_speed"] = None
            response_data["wind_direction"] = None
            response_data["weather_api_failed"] = False
            response_data["weather_error"] = None
        
        return jsonify(response_data)
    except Exception as e:
        import traceback
        print(f"pathfinding error: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


def print_test_overview():
    """print a simple test checklist to the console and write it to a file"""
    tests_text = """
tests

unit tests - validate functions, algorithms, and calculations.
  - calculated optimal path distance is within 30% of the true direct route.
  - the pathfinding algorithm clearly avoids known obstacles.
  - weather api response is clearly displayed onto the frontend.

integration tests - validate system component interaction.
  - frontend-backend api communication.
  - a path is able to be calculated from end to end.
  - mission submission workflow is able to be completed.

system tests - validate user workflow.
  - full user journey (login -> mission -> results).
  - error handling (api failures, invalid inputs).
  - cors and authentication.

performance tests - validate performance of the system.
  - path calculation time (< 30s).
  - api response times.
  - map rendering performance.
""".strip("\n")
    print(tests_text)
    try:
        with open("test_overview.txt", "w", encoding="utf-8") as f:
            f.write(tests_text + "\n")
    except Exception as e:
        print(f"could not write test_overview.txt: {e}")


if __name__ == "__main__":
    print_test_overview()
    app.run(debug=True)
