from keras.models import model_from_json
import numpy as np


class FaceKeypointsCaptureModel(object):

    COLUMNS = ['left_eye_center_x', 'left_eye_center_y',
               'right_eye_center_x', 'right_eye_center_y',
               'left_eye_inner_corner_x', 'left_eye_inner_corner_y', 
               'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
               'right_eye_inner_corner_x', 'right_eye_inner_corner_y', 
               'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
               'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
               'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
               'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
               'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
               'nose_tip_x', 'nose_tip_y',
               'mouth_left_corner_x', 'mouth_left_corner_y',
               'mouth_right_corner_x', 'mouth_right_corner_y',
               'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
               'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        print("Model loaded from disk")
        self.loaded_model.summary()

    def predict_points(self, img):
        preds = self.loaded_model.predict(img) % 96

        pred_dict = dict([(point, val) for point, val in zip(FaceKeypointsCaptureModel.COLUMNS, preds[0])])
        print(pred_dict)

        return preds, pred_dict


def scale(data, out_range=(-1, 1)):
    range_ = [0, 96]
    normal_data = (data - range_[0]) / (range_[1] - range_[0])

    return (normal_data * (out_range[1] - out_range[0])) + out_range[0]


if __name__ == '__main__':
    model = FaceKeypointsCaptureModel("face_model.json", "face_model.h5")
    
    import matplotlib.pyplot as plt
    import cv2
    img = cv2.cvtColor(cv2.imread('trial1.jpg'), cv2.COLOR_BGR2GRAY)
    img1 = cv2.resize(img, (96, 96))
    img1 = img1[np.newaxis, :, :, np.newaxis]
    
    print(img1.shape)

    pts, pts_dict = model.predict_points(img1)
    pts1 = scale(pts[0], (0, 200))

    plt.figure(0)
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', interpolation=None)
    plt.scatter(pts1[range(0, 30, 2)], pts1[range(1, 30, 2)], marker='x')

    plt.subplot(1, 2, 2)
    plt.imshow(img1[0, :, :, 0], cmap='gray', interpolation=None)
    plt.scatter(pts[0, range(0, 30, 2)], pts[0, range(1, 30, 2)], marker='x')
    plt.show()