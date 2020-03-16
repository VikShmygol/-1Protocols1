import common as cm
import template_class as templ
import page_class as page
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch, torchvision
from torch import nn, optim
from torch.autograd import Variable as var 
import torch.nn.functional as F
from torchvision import transforms
from torch.utils import data
from PIL import Image
import cnn_class as cnn


class ImgProc:

    """
    Processing of transformed image (aligned to the selected template) regions of 
    interest (ROIs): splitting numbers into digits (28*28 grayscale), evaluating 
    digit image with CNN 
    """
    def __init__(self, template_obj, explored_obj):
        self.template = template_obj 
        self.explored_page = explored_obj
        self.matches = []
        self.ROI_dict = {}    
        self.dict_with_nums = {} 
        self.model = cnn.digCNN().cuda()
        self.model.load_state_dict(torch.load('E:/#AI&CV camp/#Project/#1Protocols/mnist_digits.pt'))
        self.model.eval()

        self.transform = transforms.Compose([      
            transforms.ToTensor(),  
        ])
    
    # Extracting ROIs from transformed image acccording to
    # its location on template picture
    def __ROI_extraction(self, dict, img_num):
        img_gray = self.explored_page.gray
 
        for count, key in enumerate(dict):
            offset = np.array(dict[key][0])
       
            upper_corner = offset #first_page_coord + offset
            lower_corner = upper_corner + np.array(dict[key][1])

            key = "ROI_" + str(count)
            roi = img_gray[int(upper_corner[1]):int(lower_corner[1]), 
                            int(upper_corner[0]):int(lower_corner[0])]
            
            self.ROI_dict[key] = self.__adjust(roi)

    # Adjusting ROI exposure
    def __adjust(self, image):
        # Contrast adjusting
        sum_luminosity = np.sum(image)
        average_luminosity = sum_luminosity / (image.shape[0] * image.shape[1])
        ratio = np.sqrt(cm.kTemplateLuminosity / average_luminosity)
        image = np.clip(image*ratio, 0, 255)
        image = image.astype(np.uint8)

        # Filtering, historgram equalization, invertion
        image= cv2.bilateralFilter(image, -1, 2, 2)
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(10, 10))
        image = clahe.apply(image.astype(np.uint8))
        image = 255 - image
        offset = 130
        ret, image = cv2.threshold(image, offset, 255, cv2.THRESH_TOZERO)

        return image                           

    # Clearing up a pic, extracting digits' countours
    def __extract_character(self, image):
        chars_labels = []
        processed_chars = []

        high_thresh, bin_image = cv2.threshold(image, 0, 255, 
                                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Clearing up lines underneath the handwritten number
        edges = cv2.Canny(bin_image, 5, 15, apertureSize=3)
        lines = cv2.HoughLinesP(bin_image, 1, np.pi/180, 20, minLineLength=15, maxLineGap=1)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                if np.abs(y2-y1) > 3:
                    continue

                y_black = y1
                if y2 < y1:
                    y_black = y2

                if y_black > 0:
                    y_black -= 1

                black_x, black_y = x2-x1, bin_image.shape[0]-y_black
                blackener = np.zeros((black_y, black_x), np.uint8)
                bin_image[y_black:, x1:x2] = blackener
          
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_image, 
                                                                                connectivity=8)
        
        for count, cc in enumerate(stats):
            w_h_ratio = cc[2] / cc[3]
            if (w_h_ratio > cm.kWHRatioMin and 
                w_h_ratio < cm.kWHRatioMax and 
                cc[2] > 4 and cc[3] > 9 and
                cc[4] > 24 and cc[4] < 135):
                chars_labels.append(count)

        chars_labels.sort(key=lambda x: centroids[x][0])

        for label in chars_labels:
            output_image = np.zeros(cm.kOutputImgShape, dtype=np.uint8)

            lower_corner_x = stats[label][0] + stats[label][2]
            lower_corner_y = stats[label][1] + stats[label][3]
            char_image = image[stats[label][1]:lower_corner_y,
                        stats[label][0]:lower_corner_x]
            labeled_region = labels[stats[label][1]:lower_corner_y,
                        stats[label][0]:lower_corner_x]

            mask = (labeled_region == label)
            char_image = np.multiply(char_image, mask)

            if (char_image.shape[0] >= cm.kOutputImgShape[0] and
                char_image.shape[1] <= char_image.shape[0]):
                length = int(1.5*char_image.shape[0])
                output_image = cm.insert_pic((length, length), char_image, resize=True)
            elif char_image.shape[1] >= cm.kOutputImgShape[1]:
                length = int(1.5*char_image.shape[1])
                output_image = cm.insert_pic((length, length), char_image, resize=True)
            else:
                output_image = cm.insert_pic(cm.kOutputImgShape, char_image)
                
            processed_chars.append(output_image)
        return processed_chars

    # Interfering a picture with pretrained CNN
    def __evaluate_digit(self, digit_img):
        img_t = self.transform(digit_img).to(device='cuda')
        batch_t = torch.unsqueeze(img_t, 0)
        out = F.softmax(self.model(batch_t), dim=1)
        result_vec = out[0].cpu().detach().numpy()

        return str(np.argmax(result_vec))

    # This method combines all together
    def process(self, img_num):  
        self.__ROI_extraction(cm.kFirstPageDist, img_num)
        digits_images = []
        
        for key in self.ROI_dict:
            number = ''
            digits_images = self.__extract_character(self.ROI_dict[key])

            for cnt, pic in enumerate(digits_images):
                number += self.__evaluate_digit(pic)

            if number:
                self.dict_with_nums[key] = int(number)

        return self.dict_with_nums       
    
if __name__ == "__main__":
    pass