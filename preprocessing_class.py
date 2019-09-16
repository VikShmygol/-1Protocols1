import page_class as page
import template_class as templ
import img_processing_class as img_proc
import common as cm
import numpy as np
import cv2
import math

class PreProcess:
    def __init__(self, template_file_name_list, explored_file_name, img_num):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        self.explored_page = page.Page(explored_file_name, [], self.sift)
        self.template_list = []
        self.template_size = [cm.Size(716, 1077), cm.Size(716, 1077),
                                cm.Size(716, 1077), cm.Size(716, 1077)]
        self.img_num = img_num

        for file_name in template_file_name_list:
            self.template_list.append(templ.Template(file_name, [], self.sift)) 

    ### Voting for certain template
    def __vote(self):
        score_list = [0] * len(self.template_list)
        for count, template in enumerate(self.template_list):
            matches = self.bf_matcher.knnMatch(template.get_descriptors(),
                                             self.explored_page.get_descriptors(),
                                             k=2)
            for m, n in matches:
                if m.distance < 0.7*n.distance:
                    score_list[count] += 1
        
        max_score_index = score_list.index(max(score_list))
        return max_score_index
    
    ### Homography calculation and transformation
    def __transform(self, index):
        matches = self.bf_matcher.knnMatch(self.template_list[index].get_descriptors(),
                                             self.explored_page.get_descriptors(),
                                             k=2)
        keypoints1 = self.template_list[index].get_keypoints()
        keypoints2 = self.explored_page.get_keypoints()

        good = []
        good_without_list = []
        for m, n in matches:
            if m.distance < 0.55*n.distance:
                good.append(m)
                good_without_list.append([m])
        
        MIN_MATCH_COUNT = 3
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good
                                ]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good
                                ]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        ss = M[0, 1]
        sc = M[0, 0]
        scale_recovered = math.sqrt(ss * ss + sc * sc)
        theta_recovered = math.atan2(ss, sc) * 180 / math.pi

        #deskew image
        img_out = cv2.warpPerspective(self.explored_page.get_image(), np.linalg.inv(M),
                                    (self.template_list[index].get_shape()[1], 
                                    self.template_list[index].get_shape()[0]))
     
        return img_out


    def process(self):
        max_score_index = self.__vote()
        processed_img = self.__transform(max_score_index)
        self.explored_page.update_keypoints_descriptors(processed_img)
        img = img_proc.ImgProc(self.template_list[max_score_index],
                                self.explored_page)
                                
        return img.process(self.img_num)