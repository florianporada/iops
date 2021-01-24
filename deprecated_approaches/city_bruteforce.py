import numpy as np
import math
import json


def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6378.137
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * \
        math.cos(lat2 * math.pi / 180) * \
        math.sin(dLon/2) * math.sin(dLon/2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c

    return d * 1000


def get_cities(filename='../cities_pop_10000000.geojson'):
    with open(filename) as f:
        gj = json.load(f)
        cities = [{'name': feat['properties']['name'], 'coords': [float(feat['geometry']['coordinates'][0]), float(
            feat['geometry']['coordinates'][1])]} for feat in gj['features']]

    print("Loaded " + str(len(cities)) + " city coordinates")

    return cities


# cities
cities = [{'name': "dunno", 'coords': [22, 14]}]

# cities = get_cities()

debug = False
range_lon = [-180, 180]
range_lat = [-90, 90]
current_lon_lat = [range_lon[0], range_lat[0]]
step = 0.0089  # 1 kilometer ~ 0.0089 degreees
# step = step / 1000  # meter
result = []


while current_lon_lat[0] < range_lon[1]:
    if current_lon_lat[0] > range_lon[1]:
        current_lon_lat[0] = range_lon[1]
    else:
        current_lon_lat[0] = current_lon_lat[0] + step

    while current_lon_lat[1] < range_lat[1]:
        if current_lon_lat[1] > range_lat[1]:
            current_lon_lat[1] = range_lat[1]
        else:
            current_lon_lat[1] = current_lon_lat[1] + step

        city_distances = []

        for city in cities:
            distance = haversine_distance(
                current_lon_lat[0], current_lon_lat[1], city['coords'][0], city['coords'][1])

            city_distances.append({
                'gridCoords': current_lon_lat,
                'cityCoords': city['coords'],
                'cityDistance': distance
            })

            if (debug):
                coords = city['coords']
                name = city['name']
                print(
                    f'Distance from {name} (lon/lat {current_lon_lat[0]:.7f}/{current_lon_lat[1]:.7f}) to lon/lat {coords[0]:.7f}/{coords[1]:.7f}: {distance}')

        sorted_city_distances = sorted(
            city_distances, key=lambda x: x['cityDistance'], reverse=False)

        closest_city = sorted_city_distances[0]

        result.append(closest_city)

# highest distance to closest city
overall = sorted(result, key=lambda x: x['cityDistance'], reverse=True)

print(f'closest: {overall[0]}, farthest: {overall[len(overall) - 1]}')
