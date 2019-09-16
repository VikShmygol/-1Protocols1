from collections import namedtuple
from matplotlib import pyplot as plt
import numpy as np
import cv2


### Some named tuples
Coordinate = namedtuple('Coordinate', 'x y')
Offset = namedtuple('Offset', 'x y')
Size = namedtuple('Size', 'w h')

### Some constants
kTemplateLuminosity = 210
kOutputImgShape = (28, 28)
kWHRatioMin =  0.3 # min width/height ratio for char detection
kWHRatioMax =  1.9 # max width/height ratio for char detection

### First page dictionary with offsets for each ROI on the image
### (upper left corner, lower right corner)
kFirstPageDist = {"0_vyborcha_diln_num" : 
                    (Coordinate(245, 240), Coordinate(90, 40)),
                    "1_vyborcyi_okrug_num" : 
                    (Coordinate(620, 245), Coordinate(65, 40)),
                    "2_vyborchi_buleteni_num" : 
                    (Coordinate(605, 420), Coordinate(75, 60)),
                    "3_nevykorystani_buleteni_num" : 
                    (Coordinate(605, 490), Coordinate(80, 50)),
                    "4_vyborci_vneseni_spysok_num" : 
                    (Coordinate(607, 560), Coordinate(75, 50)),
                    "5_vyborci_za_miscem_num" : 
                    (Coordinate(615, 625), Coordinate(60, 50)),
                    "6_vyborci_otrymaly_u_prymischenni_num" : 
                    (Coordinate(607, 710), Coordinate(80, 50)),
                    "7_vyborci_otrymaly_za_miscem_num" : 
                    (Coordinate(615, 775), Coordinate(60, 50)),
                    "8_sumarna_kilkist_vyborciv_num" : 
                    (Coordinate(607, 850), Coordinate(75, 50)),
                    }

### Some functions
to_rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
to_bin = lambda x: cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)[1]

def plot_image(img, title=None, figsize=None):
    if not figsize:
        figsize = 4
        
    rgb_img = to_rgb(img)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(figsize, figsize))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if title and type(title) == str:
        ax.set_title(title)
    ax.imshow(rgb_img)
        
def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()
    
def check_num(list_to_check, num=10):
    length = len(list_to_check)
    res = num
    
    if length < num:
        res = length
        
    if num <= 0:
        raise ValueError('Invalid number of keypoints to be extracted: {}'.format(num))
    
    return res


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def insert_pic(output_size, pic, resize=False):
    width_offset = (output_size[1] - pic.shape[1]) // 2
    height_offset = (output_size[0] - pic.shape[0]) // 2 

    output_image = np.zeros(output_size, dtype=np.uint8)
    output_image[height_offset:(height_offset+pic.shape[0]),
                    width_offset:(width_offset+pic.shape[1])] = pic
    if resize:
        output_image = cv2.resize(output_image, kOutputImgShape, interpolation = cv2.INTER_AREA)

    return output_image

if __name__ == "__main__":
    img2 = cv2.imread('Protocols/1_1.jpg')
    img3 = rotate_bound(img2, 50)
    cv2.imwrite('Protocols/1_1_rot.jpg',img3)
    plt.imshow(to_rgb(img3))
    plt.show()