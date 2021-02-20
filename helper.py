import os
import cv2
from pathlib import Path
from sklearn.cluster import KMeans
import numpy as np
import urllib3
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from datetime import datetime
import cartopy.crs as ccrs
import requests
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from dotenv import load_dotenv

load_dotenv()


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


def is_in_rect(point, bbox):
    x = point[0]
    y = point[1]

    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    if x1 <= x <= x2 and y1 <= y <= y2:
        return True

    return False


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


def get_sentinelhub_token():
    # curl --request POST --url https://services.sentinel-hub.com/oauth/token --header "content-type: application/x-www-form-urlencoded" --data "grant_type=client_credentials&client_id=<your client id>" --data-urlencode "client_secret=<your client secret>"
    # Your client credentials
    client_id = os.environ.get('SENTINELHUB_CLIENT_ID')
    client_secret = os.environ.get('SENTINELHUB_CLIENT_SECRET')
    # Create a session
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)
    # Get token for the session
    token = oauth.fetch_token(token_url='https://services.sentinel-hub.com/oauth/token',
                              client_id=client_id, client_secret=client_secret)

    print('Access Token', token)
    # All requests using this session will have an access token automatically added
    resp = oauth.get("https://services.sentinel-hub.com/oauth/tokeninfo")
    print(resp.content)


def get_seninel_data(bbox):
    access_token = os.environ.get('SENTINELHUB_ACCESS_TOKEN')
    response = requests.post('https://services.sentinel-hub.com/api/v1/process',
                             headers={
                                 "Authorization": f"Bearer {access_token}"},
                             json={
                                 "input": {
                                     "bounds": {
                                         "bbox": [
                                             13.822174072265625,
                                             45.85080395917834,
                                             14.55963134765625,
                                             46.29191774991382
                                         ]
                                     },
                                     "data": [{
                                         "type": "S2L2A"
                                     }]
                                 },
                                 "evalscript": """
                                //VERSION=3

                                function setup() {
                                    return {
                                        input: ["B02", "B03", "B04"],
                                        output: {
                                        bands: 3
                                        }
                                    };
                                }

                                function evaluatePixel(
                                    sample,
                                    scenes,
                                    inputMetadata,
                                    customData,
                                    outputMetadata
                                ) {
                                    return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
                                }
                                """
                             })

    print(response)


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


def show_polygon_on_map(polygon):
    x, y = polygon.exterior.xy

    fig = plt.figure(1, figsize=(10, 5), dpi=90)
    ax = fig.add_subplot(111)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.stock_img()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.plot(x, y)
    ax.set_title('Polygon Edges')

    plt.show()
