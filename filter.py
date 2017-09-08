import cv2
import numpy as np
from model import FaceKeypointsCaptureModel


def apply_filter(frame, pts_dict):
    left_eye_center_x = pts_dict["left_eye_center_x"]
    left_eye_center_y = pts_dict["left_eye_center_y"]
    left_eye_inner_corner_x = pts_dict["left_eye_inner_corner_x"]
    left_eye_inner_corner_y = pts_dict["left_eye_inner_corner_y"]
    radius_left = distance((left_eye_center_x, left_eye_center_y),
                           (left_eye_inner_corner_x, left_eye_inner_corner_y))

    right_eye_center_x = pts_dict["right_eye_center_x"]
    right_eye_center_y = pts_dict["right_eye_center_y"]
    right_eye_inner_corner_x = pts_dict["right_eye_inner_corner_x"]
    right_eye_inner_corner_y = pts_dict["right_eye_inner_corner_y"]
    radius_right = distance((right_eye_center_x, right_eye_center_y),
                           (right_eye_inner_corner_x, right_eye_inner_corner_y))
    
    frame = apply_filter_helper(frame, left_eye_center_x,
                                left_eye_center_y, int(radius_left))
    frame = apply_filter_helper(frame, right_eye_center_x,
                                right_eye_center_y, int(radius_right))
    return frame


def apply_filter_helper(frame, x, y, radius):
    cv2.circle(frame, (x, y), radius, (255, 0, 0), 5)
    return frame

def distance(pt1, pt2):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    return np.sqrt(sum((pt1-pt2)**2))

if __name__ == '__main__':
    model = FaceKeypointsCaptureModel("face_model.json", "face_model.h5")
    
    import matplotlib.pyplot as plt
    import cv2
    img = cv2.cvtColor(cv2.imread('dataset/trial1.jpg'), cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img, (96, 96))
    img1 = img1[np.newaxis, :, :, np.newaxis]
    
    print(img1.shape)

    pts, pts_dict = model.predict_points(img1)
    pts1, pred_dict1 = model.scale_prediction((0, 200), (0, 200))

    fr = apply_filter(img, pred_dict1)

    plt.figure(0)
    plt.imshow(fr, cmap='gray')
    plt.show()