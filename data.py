import os

from pathlib import Path
import urllib3
import pickle
import _pickle as cPickle
import bz2
import elevation

from geo import get_map_shape, create_grid_elements


def get_elevation_tile_from_web(bounds=(0, 0, 1, 1), output='0_0-DEM.tif'):
    # bounds -122.6 41.15 -121.9 41.6 (lon, lat lon lat) (bottom,left to top,right)
    filename = str(Path().absolute()) + '/' + output
    elevation.clip(bounds=bounds, output=filename)

    return filename


def get_height_image_data():
    # Tile reference https://visibleearth.nasa.gov/grid
    # elevationio bounds: left bottom right top
    # tile bounds: Upper left	Lower right
    # {'name': 'heightmap.png', 'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_21600x10800.png'},
    # {'name': 'blue_marble.png', 'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/76000/76487/world.200406.3x21600x10800.png'},

    http = urllib3.PoolManager()
    to_download = [
        {
            'name': 'heightmap_A1.tif',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_A1_grey_geo.tif',
            'bbox': []
        },
        {
            'name': 'heightmap_A2.tif',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_A2_grey_geo.tif'
        },
        {
            'name': 'heightmap_B1.tif',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_B1_grey_geo.tif'
        },
        {
            'name': 'heightmap_B2.tif',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_B2_grey_geo.tif'
        },
        {
            'name': 'heightmap_C1.tif',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_C1_grey_geo.tif'
        },
        {
            'name': 'heightmap_C2.tif',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_C2_grey_geo.tif'
        },
        {
            'name': 'heightmap_D1.tif',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_D1_grey_geo.tif'
        },
        {
            'name': 'heightmap_D2.tif',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_D2_grey_geo.tif'
        },
    ]

    for el in to_download:
        r = http.request('GET', el['url'], preload_content=False)
        length = int(r.getheader('content-length'))
        size_mb = length / 1024 / 1024

        print('Start downloading ' + el['name'] +
              ' (' + str(round(size_mb, 2)) + 'MB)')

        with open('./image_data/' + el['name'], 'wb') as out:
            while True:
                data = r.read()
                if not data:
                    break
                out.write(data)

        r.release_conn()


def save_intersection_tiles(filename='geodata/ne_10m_ocean_scale_rank/ne_10m_ocean_scale_rank.shp'):
    pkl_file = Path("geodata/processed_map_elements.pkl")

    if not Path('geodata/').is_dir():
        os.mkdir('geodata/')

    if not pkl_file.is_file():
        print("Did not find processed map tiles")
        oceans = get_map_shape(filename, simplify=False, tolerance=0.5)
        grid_elements = create_grid_elements(oceans)

        with open('geodata/processed_map_elements.pkl', 'wb') as fp:
            pickle.dump(grid_elements, fp, pickle.HIGHEST_PROTOCOL)


def compressed_pickle(title, data):
    # save as .pbz2
    with bz2.BZ2File(f'{title}', 'wb') as fp:
        cPickle.dump(data, fp, pickle.HIGHEST_PROTOCOL)


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)

    return data
