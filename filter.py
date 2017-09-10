import cv2
import numpy as np
#from math import abs
from model import FaceKeypointsCaptureModel


def apply_filter(frame, pts_dict):

    #return apply_filter_eye(frame , pts_dict)
    return apply_filter_gentleman(frame , pts_dict)



def apply_filter_helper(frame, filter_img, x=(0, 2), y=(0, 2)):
    slice = frame[y[0]:y[1] , x[0]:x[1], :]
    for i in range(slice.shape[2]):
        for j in range(slice.shape[1]):
            slice[filter_img[:, j, i] != 0, j, i] = filter_img[filter_img[:, j, i]!=0, j, i]
    return slice


def apply_filter_eye_helper(frame, x, y, adjust_pos):
    filter_img = cv2.resize(cv2.imread("filters/sharingan.png"),
                            (2*adjust_pos, 2*adjust_pos))

    slice = apply_filter_helper(frame, filter_img,
                                x=(x-adjust_pos, x+adjust_pos),
                                y=(y-adjust_pos, y+adjust_pos))

    frame[y-adjust_rad:y+adjust_pos, x-adjust_pos:x+adjust_pos, :] = slice
    return frame


def apply_filter_eye(frame, pts_dict):
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

    frame = apply_filter_eye_helper(frame, int(left_eye_center_x),
                                int(left_eye_center_y), int(radius_left) // 2)
    frame = apply_filter_eye_helper(frame, int(right_eye_center_x),
                                int(right_eye_center_y), int(radius_right) // 2)

    return frame

def apply_filter_gentleman(frame, pts_dict):
    bow = cv2.imread("filters/gentleman_bow.png")
    glasses = cv2.imread("filters/gentleman_glass.png")
    hat = cv2.imread("filters/gentleman_hat.png")
    moustache = cv2.imread("filters/gentleman_moustache.png")

    nose_x = int(pts_dict["nose_tip_x"])
    top_lip_y = int(pts_dict["mouth_center_top_lip_y"])
    #x_dist = abs(top_lip_x - nose_x) 

    right_corner_y = int(pts_dict["mouth_right_corner_y"])
    left_corner_y = int(pts_dict["mouth_left_corner_y"])
    #y_dist = abs(left_corner_y - right_corner_y)
    #print(x_dist, y_dist)

    moustache = cv2.resize(moustache, (200, 10))
    slice = apply_filter_helper(frame, moustache,
                                x=(top_lip_y-100, top_lip_y+100),
                                y=(nose_x, nose_x+10))

    #frame[right_corner_y: left_corner_y, nose_x:top_lip_x, :] = slice
    frame[nose_x:nose_x+10, top_lip_y-100:top_lip_y+100, :] = slice
    return frame


def distance(pt1, pt2):
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    return np.sqrt(sum((pt1-pt2)**2))

if __name__ == '__main__':
    model = FaceKeypointsCaptureModel("face_model.json", "face_model.h5")
    
    import matplotlib.pyplot as plt
    import cv2
    img_ = cv2.imread('dataset/trial1.jpg')
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (96, 96))
    img = img[np.newaxis, :, :, np.newaxis]
    
    print(img.shape)

    pts, pts_dict = model.predict_points(img)
    pts, pred_dict = model.scale_prediction((0, 200), (0, 200))

    fr = apply_filter(img_, pred_dict)
    fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    print(fr.shape)

    plt.figure(0)
    plt.imshow(fr)
    plt.show()