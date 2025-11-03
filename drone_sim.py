import folium
import numpy as np

def simulate_drone(start, end, steps=20, output_file="assets/mayday_drone_simulation.html"):
    """Simulate drone movement between two coordinates."""
    start = np.array(start)
    end = np.array(end)
    m = folium.Map(location=start.tolist(), zoom_start=13)

    folium.Marker(start.tolist(), popup="Start", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(end.tolist(), popup="End", icon=folium.Icon(color='red')).add_to(m)
    folium.PolyLine([start.tolist(), end.tolist()], color="blue", weight=2.5, opacity=1).add_to(m)

    positions = [((start + (end - start) * i / steps)).tolist() for i in range(steps + 1)]

    for pos in positions:
        folium.CircleMarker(pos, radius=6, color="orange", fill=True, fill_color="orange").add_to(m)

    m.save(output_file)
    print(f"Drone simulation saved to {output_file}")

if __name__ == "__main__":
    simulate_drone([34.05, -118.25], [34.10, -118.15])
