#imports
import cv2

#define our one function. add documentation.
def calculate_homography(src_points, des_points, outlier_algorithm=cv2.RANSAC):
    cv2.calculateHomography(src_points, des_points, outlier_algorithm, 5.0)

