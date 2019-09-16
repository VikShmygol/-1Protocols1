import img_processing_class as img_proc
import preprocessing_class as preprocess
import common as cm
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

def main():
        for i in range(1, 3):
            img_name = 'Protocols/' + str(i) + '_1.jpg'
            template_list = ['Protocols/1_1_templ.jpg', 'Protocols/template_2.jpg']
            try:
                go = preprocess.PreProcess(template_list, img_name, i) 
                numbers = go.process()
                file_name = 'Protocols/Output/test_' + str(i) + '.txt'
                file = open(file_name, 'w')
                for key in numbers:
                    file.write(key + ' ' + str(numbers[key]) + '\n')
                file.close()
            except ValueError:
                print (sys.exc_info()[1])


if __name__ == "__main__":
        main()



