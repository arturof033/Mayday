import tkinter as tk
import folium
import numpy as np
import time
import threading
import os
import webbrowser

# generate drone positions between two coordinates
def simulate_drone(start, end, steps=20):
    start = np.array(start)
    end = np.array(end)
    return [((start + (end - start) * i / steps)).tolist() for i in range(steps + 1)]

# create a folium map with the drone's current position
def create_map(start, end, drone_pos, file_path):
    m = folium.Map(location=drone_pos, zoom_start=13)
    folium.Marker(start, popup="Start", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(end, popup="End", icon=folium.Icon(color='red')).add_to(m)
    folium.PolyLine([start, end], color="blue", weight=2.5, opacity=1).add_to(m)
    folium.Marker(drone_pos, popup="Drone", icon=folium.Icon(color="orange")).add_to(m)
    m.save(file_path)

# animate drone flight and open in browser
def animate_drone(start, end, steps=30):
    os.makedirs("assets", exist_ok=True)
    file_path = os.path.abspath("assets/mayday_drone.html")

    positions = simulate_drone(start, end, steps)
    print("Simulating drone flight...")

    for i, pos in enumerate(positions):
        create_map(start, end, pos, file_path)
        print(f"Step {i+1}/{steps} -> Drone at {pos}")
        time.sleep(0.3)  # simulate movement

    print("Simulation complete. Opening map...")
    webbrowser.open(f"file://{file_path}", new=2)

# gui setup
def main():
    root = tk.Tk()
    root.title("Mayday Drone Simulation")
    root.geometry("300x200")

    start = [33.8823, -117.8851]
    end = [34.10, -118.15]

    def start_sim():
        threading.Thread(target=animate_drone, args=(start, end), daemon=True).start()

    tk.Label(root, text="Mayday Drone Simulation", font=("Arial", 14)).pack(pady=20)
    tk.Button(root, text="Start Simulation", command=start_sim, width=20).pack(pady=10)
    tk.Button(root, text="Exit", command=root.destroy, width=20).pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
