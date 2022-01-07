import numpy as np
import cv2
from sklearn.cluster import KMeans


class SIFT():
    def __init__(self, n_classes):
        self.sift = cv2.SIFT_create()
        self.k = n_classes * 10
        self.kmeans = KMeans(n_clusters=self.k)
        # self.descriptors = []

    def apply_sift(self, images):

        descriptors = []

        for im in images:
            image8bit = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            kp, des = self.sift.detectAndCompute(image8bit, None)
            if des is None:
                continue
            for d in des:
                descriptors.append(d)

        # clustering part
        self.kmeans.fit(descriptors)

    def get_features(self, images):

        # histogram list
        features_list = []

        for img in images:
            image8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

            kp, des = self.sift.detectAndCompute(image8bit, None)

            features = np.zeros(self.k)
            nkp = np.size(kp)

            if des is None:
                continue
            for d in des:
                idx = self.kmeans.predict([d])  # get cluster index
                features[idx] += 1 / nkp  # normalize histogram

            features_list.append(features)

        return features_list
