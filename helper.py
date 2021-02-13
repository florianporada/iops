import cv2
from pathlib import Path
from sklearn.cluster import KMeans
import numpy as np
import urllib3
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from datetime import datetime
import cartopy.crs as ccrs


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def get_crop_rect(lon, lat, rect_width, rect_height, dimensions):
    img_height = dimensions[0]
    img_width = dimensions[1]
    lon_index = lon + 180
    lat_index = lat + 90

    y = img_height - (lat_index + 1) * rect_height
    yh = y + rect_height

    x = lon_index * rect_width
    xw = x + rect_width

    return y, yh, x, xw


def create_image_tiles(image_path, debug=False):
    # read image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # get dimensions of image
    dimensions = img.shape

    # height, width, number of channels in image
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2] if len(img.shape) > 2 else -1

    # map lon + lat to pixel dimensions
    longitude_size = int(width / 360)
    latitude_size = int(height / 180)
    grid_dict = {}

    if debug:
        print('Image Dimension    : ', dimensions)
        print('Image Height       : ', height)
        print('Image Width        : ', width)
        print('Number of Channels : ', channels)
        print('lon size : ', longitude_size)
        print('lat size : ', latitude_size)

    for latitude in range(-90, 90):
        for longitude in range(-180, 180):
            # get lonlat tile
            # https://stackoverflow.com/questions/9084609/
            y, yh, x, xw = get_crop_rect(
                longitude, latitude, longitude_size, latitude_size, img.shape)

            crop_img = img[y:yh, x:xw]

            grid_dict[str(longitude) + ',' + str(latitude)] = crop_img

    return grid_dict


def get_image_data():
    # Tile reference https://visibleearth.nasa.gov/grid
    http = urllib3.PoolManager()
    to_download = [
        {'name': 'heightmap.png', 'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_21600x10800.png'},
        {'name': 'blue_marble.png',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/76000/76487/world.200406.3x21600x10800.png'},
        {'name': 'heightmap_A1.tif',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_A1_grey_geo.tif'},
        {'name': 'heightmap_A2.tif',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_A2_grey_geo.tif'},
        {'name': 'heightmap_B1.tif',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_B1_grey_geo.tif'},
        {'name': 'heightmap_B2.tif',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_B2_grey_geo.tif'},
        {'name': 'heightmap_C1.tif',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_C1_grey_geo.tif'},
        {'name': 'heightmap_C2.tif',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_C2_grey_geo.tif'},
        {'name': 'heightmap_D1.tif',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_D1_grey_geo.tif'},
        {'name': 'heightmap_D2.tif',
            'url': 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_D2_grey_geo.tif'},
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


def merge_tif_tiles():
    # file_list = glob.glob("c:\data\....\*.tif")

    # files_string = " ".join(file_list)

    # command = "gdal_merge.py -o output.tif -of gtiff " + files_string

    # os.system(command)
    return 0


# ------------------------------------------------------- Generic Helper Methods
def create_video(point_array, details=None):
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='GA Coords', artist='florianporada',
                    comment='IOPS')
    writer = FFMpegWriter(fps=25, metadata=metadata)
    dt = datetime.now()

    # basemap = Basemap(projection='mill', lon_0=0)
    # basemap.drawcoastlines()
    # basemap.drawmapboundary(fill_color='aqua')
    # basemap.fillcontinents(color='coral', lake_color='aqua')

    fig = plt.figure(figsize=(18, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    plt.xlim(-180, 180)
    plt.ylim(-90, 90)

    sct = plt.scatter(point_array[0][:, 0],
                      point_array[0][:, 1], c='r', s=12)

    with writer.saving(fig, "./output/genetic_coords_viz_" + dt.strftime("%m_%d_%Y__%H_%M_%S") + ".mp4", 300):
        for i in range(len(point_array)):
            curr_pop = point_array[i]
            sct.set_offsets(curr_pop)

            if details is not None and i == len(point_array) - 1:
                print('Add detail data label')
                for ii, fitness_index in enumerate(details[0]):
                    plt.plot(curr_pop[int(details[0][0])][0],
                             curr_pop[int(details[0][0])][1], 'bo', markersize=15)
                    ax.annotate(
                        f'{details[1][ii]:.4f}', (curr_pop[int(fitness_index)][0], curr_pop[int(fitness_index)][1]))

            for i in range(2):
                writer.grab_frame()


def plot_population(population, details):
    fig = plt.figure(figsize=(18, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.stock_img()  # Background images ref http://earthpy.org/tag/cartopy.html
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.scatter(population[:, 0],
                population[:, 1], c='r', s=12)

    for ii, fitness_index in enumerate(details[0]):
        plt.plot(population[int(details[0][0])][0],
                 population[int(details[0][0])][1], 'bo', markersize=15)
        ax.annotate(
            f'{details[1][ii]:.4f}', (population[int(fitness_index)][0], population[int(fitness_index)][1]))

    plt.show()


def show_grid_elements(elements):
    fig, axs = plt.subplots()
    axs.set_aspect('equal', 'datalim')

    for key in elements:
        el = elements[key]

        if not el.is_empty:
            if el.geom_type == 'MultiPolygon':
                for geom in el.geoms:
                    xs, ys = geom.exterior.xy
                    axs.fill(xs, ys, alpha=0.5, c=np.random.rand(3,))

            if el.geom_type == 'Polygon':
                xs, ys = el.exterior.xy
                axs.fill(xs, ys, alpha=0.5, c=np.random.rand(3,))

    # for geom in land.geoms:
    #     xs, ys = geom.exterior.xy
    #     axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')

    plt.show()
