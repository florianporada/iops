
import math
from pathlib import Path
from shapely.geometry import Point, MultiPolygon, shape, Polygon
import fiona
import json
from osgeo import gdal
import numpy as np
import elevation
from datetime import datetime


def get_haversine_distance(lon1, lat1, lon2, lat2):
    R = 6378.137
    R = 6371
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * \
        math.cos(lat2 * math.pi / 180) * math.sin(dLon/2) * math.sin(dLon/2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c

    # Kilometers
    return d


def get_cities(filename='geodata/cities_pop_10000000.geojson'):
    with open(filename) as f:
        gj = json.load(f)
        cities = [{'name': feat['properties']['name'], 'coords': [float(feat['geometry']['coordinates'][0]), float(
            feat['geometry']['coordinates'][1])]} for feat in gj['features']]

    print("Loaded " + str(len(cities)) + " city coordinates")

    return cities


def get_elevation_tile_from_web(bounds=(0, 0, 1, 1), output='0_0-DEM.tif'):
    # bounds -122.6 41.15 -121.9 41.6 (lon, lat lon lat) (bottom,left to top,right)
    filename = str(Path().absolute()) + '/' + output
    elevation.clip(bounds=bounds, output=filename)

    return filename


def create_grid_elements(shape):
    longitude_size = 0.99999999
    latitude_size = 0.99999999
    grid_dict = {}

    print("Start grid creation (this may take a while)")

    for latitude in range(-90, 90):
        for longitude in range(-180, 180):
            grid_element = Polygon([
                (longitude, latitude),
                (longitude + longitude_size, latitude),
                (longitude + longitude_size, latitude + latitude_size),
                (longitude, latitude + latitude_size)
            ])

            splitted = grid_element.intersection(shape)

            grid_dict[str(longitude) + ',' + str(latitude)] = splitted

        print(str(datetime.now()) +
              " Calculated intersections for lat row " + str(latitude) + "/90")

    return grid_dict


def get_map_shape(filename='geodata/ne_10m_ocean/ne_10m_ocean.shp', simplify=False, tolerance=0.5):
    multipolygon = MultiPolygon([shape(pol['geometry'])
                                 for pol in fiona.open(filename)])

    print("Loaded " + str(filename) + " multipolygon")

    if not multipolygon.is_valid:
        print("Shape not valid. Applying fix")
        multipolygon = multipolygon.buffer(0)

    if simplify:
        simplified = multipolygon.simplify(tolerance)
        simplified = simplified.buffer(0)

        return simplified

    return multipolygon


def is_in_shape(point, shape):
    p1 = Point(point)

    if p1.within(shape):
        return 0
    else:
        return 1
