import tkinter as tk
import folium
import numpy as np
import time
import threading
import os
import webbrowser
import requests
from math import radians, cos, sin, sqrt, atan2

# function to calculate distance between two coordinates (Haversine formula)
def haversine_distance(coord1, coord2):
    R = 6371  # earth's radius in km
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def total_path_distance(path):
    """calculate total path distance in kilometers"""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(path)):
        total += haversine_distance(path[i-1], path[i])
    return total

# get buildings and obstacles from OpenStreetMap
def get_osm_obstacles(start, end, buffer=0.05):
    """
    Fetch buildings, restricted areas, and obstacles from OSM
    buffer: degree buffer around the route (roughly 5km at this latitude)
    """
    min_lat = min(start[0], end[0]) - buffer
    max_lat = max(start[0], end[0]) + buffer
    min_lon = min(start[1], end[1]) - buffer
    max_lon = max(start[1], end[1]) + buffer
    
    # overpass API query for obstacles
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # query for ALL buildings, airports, military zones, and tall structures
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
    
    try:
        print("Fetching obstacles from OpenStreetMap...")
        response = requests.post(overpass_url, data=query, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            obstacles = []
            
            for element in data.get('elements', []):
                if element['type'] == 'way' and 'geometry' in element:
                    # get center point of obstacle
                    coords = [[node['lat'], node['lon']] for node in element['geometry']]
                    if coords:
                        center_lat = sum(c[0] for c in coords) / len(coords)
                        center_lon = sum(c[1] for c in coords) / len(coords)
                        
                        # calculate size of obstacle
                        size = 0.008  # default radius (~800m)
                        if 'tags' in element:
                            tags = element['tags']
                            # larger radius for airports and major obstacles
                            if tags.get('aeroway') in ['aerodrome', 'runway']:
                                size = 0.03  # 3km radius for airports
                            elif tags.get('landuse') == 'military':
                                size = 0.02  # 2km for military
                            elif tags.get('building') in ['commercial', 'retail', 'mall']:
                                size = 0.01  # 1km for malls and commercial
                            elif tags.get('building') == 'industrial':
                                size = 0.008
                        
                        obstacles.append({
                            'center': [center_lat, center_lon],
                            'type': element.get('tags', {}).get('aeroway') or element.get('tags', {}).get('building', 'obstacle'),
                            'coords': coords,
                            'radius': size
                        })
                elif element['type'] == 'node':
                    tags = element.get('tags', {})
                    size = 0.03 if tags.get('aeroway') == 'aerodrome' else 0.008
                    obstacles.append({
                        'center': [element['lat'], element['lon']],
                        'type': tags.get('aeroway') or tags.get('man_made') or 'tower',
                        'coords': [[element['lat'], element['lon']]],
                        'radius': size
                    })
            
            print(f"Found {len(obstacles)} obstacles")
            return obstacles
        else:
            print(f"Failed to fetch OSM data: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching obstacles: {e}")
        return []

## alternate algorithm with more leniency on obstacles
def find_optimal_drone_path2(start, end, obstacles):
    print("Calculating optimal drone path...")
    
    # filter only high-risk obstacles
    major_obstacles = [
        obs for obs in obstacles 
        if obs['type'] in ['aerodrome', 'runway', 'military', 'tower']
    ]
    print(f"Focusing on {len(major_obstacles)} high-priority obstacles")
    
    # urban density failsafe
    if len(major_obstacles) > 200:
        print("Urban density too high — flying direct.")
        return [start, end], []
    
    path = [start]
    avoided_obstacles = []
    current = np.array(start)
    goal = np.array(end)
    max_iterations = 30
    iteration = 0
    
    while haversine_distance(current.tolist(), goal.tolist()) > 0.01 and iteration < max_iterations:
        iteration += 1
        direction = goal - current
        step_size = min(0.02, np.linalg.norm(direction))
        next_step = current + (direction / np.linalg.norm(direction)) * step_size if np.linalg.norm(direction) > 0 else current
        
        collision = False
        closest_obstacle = None
        min_clearance = float('inf')
        
        for obs in major_obstacles:
            obs_center = np.array(obs['center'])
            obs_radius = min(obs['radius'], 0.01)  # ✅ Clamp radius
            dist_to_obs_km = np.linalg.norm(next_step - obs_center) * 111
            
            if dist_to_obs_km < obs_radius * 111 * 0.8:
                collision = True
                if dist_to_obs_km < min_clearance:
                    min_clearance = dist_to_obs_km
                    closest_obstacle = obs
        
        if not collision:
            current = next_step
            path.append(current.tolist())
        else:
            obs_center = np.array(closest_obstacle['center'])
            to_obs = obs_center - current
            perp = np.array([-to_obs[1], to_obs[0]]) / np.linalg.norm(to_obs)
            tangent_direction = perp if np.dot(goal - current, perp) > 0 else -perp
            waypoint = obs_center + tangent_direction * (closest_obstacle['radius'] * 1.1)
            path.append(waypoint.tolist())
            current = waypoint
        
        # hard stop condition
        if len(path) > 200:
            print("Path too complex — simplifying.")
            return [start, end], avoided_obstacles
    
    path.append(end)
    return path, avoided_obstacles


def is_obstacle_blocking(start, end, obstacle_center, obstacle_radius):
    """Check if an obstacle blocks the direct path from start to end"""
    start = np.array(start)
    end = np.array(end)
    obstacle_center = np.array(obstacle_center)
    
    # vector from start to end
    path_vec = end - start
    path_length = np.linalg.norm(path_vec)
    
    if path_length < 0.0001:
        return False
    
    path_unit = path_vec / path_length
    
    # vector from start to obstacle
    to_obstacle = obstacle_center - start
    
    # project obstacle onto path
    projection_length = np.dot(to_obstacle, path_unit)
    
    # check if obstacle is along the path (not behind or way past)
    if projection_length < 0 or projection_length > path_length:
        return False
    
    # find closest point on path to obstacle
    closest_point = start + path_unit * projection_length
    
    # distance from obstacle center to path
    distance_to_path = np.linalg.norm(obstacle_center - closest_point)
    
    # convert radius from degrees to km for comparison
    radius_km = obstacle_radius * 111  # rough conversion
    distance_km = distance_to_path * 111
    
    return distance_km < radius_km

def can_fly_direct(point1, point2, obstacles):
    """Check if drone can fly directly between two points without hitting obstacles"""
    # only check obstacles with meaningful radius
    relevant_obstacles = [obs for obs in obstacles if obs.get('radius', 0) > 0.005]
    
    for obs in relevant_obstacles:
        if is_obstacle_blocking(point1, point2, np.array(obs['center']), obs['radius']):
            return False
    return True

# generate drone positions along a path
def simulate_drone_on_path(path, steps=50):
    """Interpolate positions along the given path to create smooth animation"""
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

# fallback: straight line
def simulate_drone_straight(start, end, steps=50):
    start = np.array(start)
    end = np.array(end)
    return [((start + (end - start) * i / steps)).tolist() for i in range(steps + 1)]

# create a folium map with route and obstacles
def create_map_with_animation(start, end, route, obstacles, avoided_obstacles, file_path):
    import folium
    from folium.plugins import TimestampedGeoJson

    m = folium.Map(location=start, zoom_start=12)

    # add start and end markers
    folium.Marker(start, popup="Start: Santa Monica", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(end, popup="End: Downtown LA", icon=folium.Icon(color='red')).add_to(m)

    # draw route
    folium.PolyLine(route, color="blue", weight=3, opacity=0.8).add_to(m)

    # add obstacles (optional for visual context)
    for obs in obstacles[:30]:
        folium.CircleMarker(
            obs['center'],
            radius=4,
            color='red',
            fill=True,
            fillOpacity=0.3,
            popup=f"Obstacle: {obs['type']}"
        ).add_to(m)

    # ANIMATION
    # create Timestamped GeoJSON for drone movement
    features = []
    for i, pos in enumerate(route):
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [pos[1], pos[0]]},
            "properties": {
                "time": f"2025-01-01T00:{i:02d}:00",
                "style": {"color": "orange"},
                "icon": "circle",
                "iconstyle": {
                    "fillColor": "orange",
                    "fillOpacity": 0.9,
                    "stroke": "true",
                    "radius": 7
                }
            }
        })

    TimestampedGeoJson({
        "type": "FeatureCollection",
        "features": features
    }, period="PT1S", add_last_point=True, auto_play=True, loop=False).add_to(m)

    m.save(file_path)


    # alternate animate drone flight along the optimal route
def animate_drone2(start, end, steps=50):
    os.makedirs("assets", exist_ok=True)
    file_path = os.path.abspath("assets/mayday_drone.html")

    obstacles = get_osm_obstacles(start, end)
    route, avoided_obstacles = find_optimal_drone_path2(start, end, obstacles) if obstacles else ([start, end], [])

    # interpolate route points for smoother animation
    route = simulate_drone_on_path(route, steps)

    print(f"Animating drone flight over {len(route)} steps...")
    total_km = total_path_distance(route)   
    print(f"Total flight distance: {total_km:.2f} km")

    create_map_with_animation(start, end, route, obstacles, avoided_obstacles, file_path)
    webbrowser.open(f"file://{file_path}", new=2)


# GUI setup
def main():
    root = tk.Tk()
    root.title("Mayday Drone Simulation")
    root.geometry("350x280")
    
    #start = [33.8823, -117.8851]  # CSUF
    #end = [34.10, -118.15]  # South Pasadena
    end = [34.016, -118.491]  # Santa Monica (near beach)
    start = [34.052, -118.243]    # Downtown LA
    #start = [33.920, -118.450]  # West of LAX (Pacific coast)
    #end = [33.965, -118.370]    # East of LAX (Inglewood area)


    def start_sim2():
        status_label.config(text="Status: Finding optimal path...")
        threading.Thread(target=lambda: run_sim2(start, end), daemon=True).start()

    def run_sim2(s, e):
        animate_drone2(s, e)
        status_label.config(text="Status: Simulation complete!")
    
    # UI Elements
    tk.Label(root, text="Mayday Drone Simulation", font=("Arial", 16, "bold")).pack(pady=15)
    tk.Label(root, text="Optimized Drone Pathfinding", font=("Arial", 9, "italic"), fg="blue").pack(pady=2)
    
    status_label = tk.Label(root, text="Status: Ready", font=("Arial", 9), fg="green")
    status_label.pack(pady=8)

    tk.Button(
        root, 
        text="Start Simulation Alternate Algorithm", 
        command=start_sim2, 
        width=20,
        bg="#4CAF50",
        font=("Arial", 10, "bold")
    ).pack(pady=10)
    
    tk.Button(
        root, 
        text="Exit", 
        command=root.destroy, 
        width=20,
        font=("Arial", 10)
    ).pack(pady=5)
    
    info_frame = tk.Frame(root)
    info_frame.pack(pady=5)
    
    tk.Label(info_frame, text="• Avoids tall buildings & obstacles", font=("Arial", 8), fg="gray").pack()
    tk.Label(info_frame, text="• Uses OpenStreetMap data", font=("Arial", 8), fg="gray").pack()
    
    root.mainloop()

if __name__ == "__main__":
    main()