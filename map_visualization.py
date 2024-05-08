import folium

map = folium.Map(location=[12.9716, 77.5946], zoom_start=10)


folium.Marker([12.9716, 77.5946], popup='Bangalore').add_to(map)


map.save('bangalore_map.html')
