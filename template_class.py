import cv2
import numpy as np

class Template:
    
    def __init__(self, file_name, img_array, sift):
        self.sift = sift
        self.image = []
        self.gray = []
        self.keypoints = [] 
        self.descriptors = []
        self.file_name = file_name

        self.image = img_array if not file_name else cv2.imread(file_name)
        self.update_keypoints_descriptors(self.image)

    def update_keypoints_descriptors(self, image):
        if image is None:
            raise ValueError('File not found: "{}"'.format(self.file_name))
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.keypoints, self.descriptors = self.sift.detectAndCompute(self.gray, 
                                                                    mask=None)
        self.image = image
                
    