import folium

def create_route_map(start_coords, end_coords, output_file="assets/mayday_route.html"):
    """Create a Folium map with start and end markers and a line connecting them."""
    m = folium.Map(location=start_coords, zoom_start=13, tiles="OpenStreetMap")

    folium.Marker(start_coords, popup="Start", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(end_coords, popup="End", icon=folium.Icon(color='red')).add_to(m)
    folium.PolyLine([start_coords, end_coords], color="blue", weight=2.5, opacity=1).add_to(m)

    m.save(output_file)
    print(f"Route map saved to {output_file}")

if __name__ == "__main__":
    start = [34.05, -118.25]
    end = [34.10, -118.15]
    create_route_map(start, end)
