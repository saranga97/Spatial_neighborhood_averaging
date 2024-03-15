import cv2
import numpy as np


def spatial_average(image, neighborhood_size):

    padded_image = cv2.copyMakeBorder(image, neighborhood_size // 2, neighborhood_size // 2, neighborhood_size // 2,
                                      neighborhood_size // 2, cv2.BORDER_CONSTANT)

    kernel = np.ones((neighborhood_size, neighborhood_size), dtype=np.float32) / (neighborhood_size * neighborhood_size)

    averaged_image = cv2.filter2D(padded_image, -1, kernel)

    return averaged_image


def main():

    image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not open or find the image.")
        return

    # Spatial averaging with 3x3 neighborhood
    averaged_image_3x3 = spatial_average(image, 3)

    # Spatial averaging with 10x10 neighborhood
    averaged_image_10x10 = spatial_average(image, 10)

    # Spatial averaging with 20x20 neighborhood
    averaged_image_20x20 = spatial_average(image, 20)

    cv2.imshow('Original Image', image)
    cv2.imshow('3x3 Averaged Image', averaged_image_3x3)
    cv2.imshow('10x10 Averaged Image', averaged_image_10x10)
    cv2.imshow('20x20 Averaged Image', averaged_image_20x20)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()