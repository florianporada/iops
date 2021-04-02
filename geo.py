
import os
import math
from functools import partial
from pathlib import Path
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from shapely.geometry import Point, MultiPolygon, shape, Polygon
import shapely.ops as ops
import pyproj
import fiona
import json
from osgeo import gdal
import numpy as np
from datetime import datetime, date
from dotenv import load_dotenv
import pickle

load_dotenv()


def get_haversine_distance(lon1, lat1, lon2, lat2):
    R = 6378.137
    R = 6371
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(lat1 * math.pi / 180) * \
        math.cos(lat2 * math.pi / 180) * \
        math.sin(dLon / 2) * math.sin(dLon / 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c

    # Kilometers
    return d


def get_projected_area(geometry):
    geom_area = ops.transform(
        partial(
            pyproj.transform,
            pyproj.Proj(init='EPSG:4326'),
            pyproj.Proj(
                proj='aea',
                lat_1=geometry.bounds[1],
                lat_2=geometry.bounds[3])),
        geometry)

    # area in m^2
    return geom_area.area


def get_cities(filename='geodata/cities_pop_10000000.geojson'):
    with open(filename) as f:
        gj = json.load(f)
        cities = [{'name': feat['properties']['name'], 'coords': [float(feat['geometry']['coordinates'][0]), float(
            feat['geometry']['coordinates'][1])]} for feat in gj['features']]

    print("Loaded " + str(len(cities)) + " city coordinates")

    return cities


def create_grid_elements(shape):
    # 1°            0   Grad            1°              1°              111 km = 60 nm
    # 0,1°          1   Dezigrad        0°06'           0°06'           11,1 km = 6 nm
    # 0,01°	        2   Zentigrad       0°00,6'         0°00'36"        1,11 km = 6 kbl
    # 0,001°        3   Milligrad       0°00,06'        0°00'03,6"      111 m = 0,6 kbl
    # 0,0001°	    4   100 Mikrograd   0°00,006'       0°00'00,36"     11,1 m
    # 0,00001°      5   10 Mikrograd    0°00,0006'      0°00'00,04"     1,11 m
    # 0,000001°     6   Mikrograd       0°00,00006'     0°00'00,004"    11,1 cm
    # 0,0000001°    7   100 Nanograd    0°00,000006'    0°00'00,0004"   1,11 cm
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


def save_sentinel_tiles(date=(date(2020, 7, 1), date(2020, 8, 31)), lat_offset=0, lon_offset=0):
    row_dict = {}
    start = date[0].strftime("%Y_%m_%d")
    end = date[1].strftime("%Y_%m_%d")
    producttypes = ['S2MSI1C', 'S2MSI2A', 'S2MSI2Ap']
    producttype = producttypes[1]
    folder = f'./geodata/sentinel_2_{start}_to_{end}_{producttype}'

    if not Path(folder).is_dir():
        os.mkdir(folder)
    else:
        print(f"Folder '{folder}' already exists")

    for latitude in range(-90 + lat_offset, 90):
        for longitude in range(-180 + lon_offset, 180):
            # get the center of the tile by simple lon/lat shifting (size is 1x1 => center is 0.5,0.5)
            tile_center = Point(longitude + 0.5, latitude + 0.5)

            tile_dataframe = get_sentinel_tile_meta(
                tile_center, date=date, producttype=producttype, debug=True)

            row_dict[str(longitude) + ',' + str(latitude)] = tile_dataframe

        print(str(datetime.now()) +
              " Save sentinel tile data for lat row " + str(latitude) + "/90")

        filename = folder + '/row_' + str(latitude) + '.pkl'

        if not Path(filename).is_file():
            with open(folder + '/row_' + str(latitude) + '.pkl', 'wb') as fp:
                pickle.dump(row_dict, fp, pickle.HIGHEST_PROTOCOL)
        else:
            print(f'File {filename} already exists. Skip.')

        row_dict = {}


def get_sentinel_tile_meta(point, date=(date(2020, 7, 1), date(2020, 8, 31)), producttype='S2MSI2A', raw=False, debug=False):
    user = os.environ.get('SENTINEL_USER')
    password = os.environ.get('SENTINEL_PASSWORD')
    api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')

    products = api.query(point.wkt,
                         date=date,
                         platformname='Sentinel-2',
                         #  processinglevel='Level-2A',
                         producttype=producttype,
                         cloudcoverpercentage=(0, 10)
                         )

    if debug:
        print(str(datetime.now()) + ' Getting tile for: ' +
              point.wkt + '. Size: ' + str(len(products)))
    if raw:
        return products
    else:
        return api.to_geodataframe(products)


def get_optimized_tile_meta(point, tiles_dataframe):
    if tiles_dataframe.empty:
        return tiles_dataframe

    unsorted_rows = []

    for index, row in tiles_dataframe.iterrows():
        polygon = row['geometry']
        date = row['ingestiondate']
        cloudcoverpercentage = row['cloudcoverpercentage']
        centroid = polygon.centroid
        poly_area = get_projected_area(polygon)
        distance_to_point = get_haversine_distance(
            point[0], point[1], centroid.x, centroid.y)

        unsorted_rows.append({
            'df_index': index,
            'date': date,
            'cloudcoverpercentage': cloudcoverpercentage,
            'area': poly_area,
            'distance': distance_to_point,
        })

    sorted_rows_by_area = sorted(
        unsorted_rows, key=lambda k: k['area'], reverse=True)
    # sorted_rows_by_distance = sorted(
    #     unsorted_rows, key=lambda k: k['distance'])

    df_index = sorted_rows_by_area[0]['df_index']
    tile_meta = tiles_dataframe.loc[df_index]

    return tile_meta


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
