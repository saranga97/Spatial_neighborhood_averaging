import cv2
import numpy as np


def spatial_average(image, neighborhood_size):

    padded_image = cv2.copyMakeBorder(image, neighborhood_size // 2, neighborhood_size // 2, neighborhood_size // 2,
                                      neighborhood_size // 2, cv2.BORDER_CONSTANT)

    kernel = np.ones((neighborhood_size, neighborhood_size), dtype=np.float32) / (neighborhood_size * neighborhood_size)

    averaged_image = cv2.filter2D(padded_image, -1, kernel)

    return averaged_image


