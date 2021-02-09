import numpy as np
import cv2
from sklearn.cluster import KMeans


def analyze_image(img, cluster=3):
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
    Z = img.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = cluster
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2


def get_dominant_color(img, clusters=3, show=False):
    # convert to rgb from bgr
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # reshaping to a list of pixels
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    # using k-means to cluster pixels
    kmeans = KMeans(n_clusters=clusters)

    labels = kmeans.fit_predict(img)

    kmeans.fit(img)

    # the cluster centers are our dominant colors.
    colors = kmeans.cluster_centers_

    # save labels
    labels = kmeans.labels_

    # returning after converting to integer from float
    return colors.astype(int), labels
