import cv2
import numpy as np
import math
from utils import *
from Sift import *
from OpenCVVideoStabilization import *
from VideoStabilization import *
from OpenCVImageTransform import *
from ImageTransform import *

################  HW2  #####################
# Nathana Facion                 RA:191079
# Rafael Mariottini Tomazela     RA:192803
############################################



def main():
    # Create image by video
    #create_image('input/p2-1-3.mp4')

    # Get interest points // Remover after
    #img1 = cv2.imread('input/p2-1-2.png')
    #img2 = cv2.imread('input/p2-1-2.png')
    #img2 = cv2.imread('input/p2-1-7.png')
    #img1 = cv2.copyMakeBorder(img1, 256, 256, 512, 256, cv2.BORDER_CONSTANT, 0)
    #img2 = cv2.copyMakeBorder(img2, 256, 256, 512, 322, cv2.BORDER_CONSTANT, 0)
    #stab = OpenCVImageTransform()
    #res = stab.show_matched(img1, img2)
    #debug("res img", res)
    # Get interest points
    #interest_points('input/p2-1-1.png','output/p2-1-3-'+ str(numFile()) + '.png',4)
    #stab = OpenCVVideoStabilization('input/p2-1-9.mp4', 'output/nometemp')
    #img1 = cv2.imread('input/p1-1-10.png')
    #img1 = cv2.imread('input/p2-1-3.png')
    #img2 = cv2.imread('input/p2-1-4.png')

    stab = VideoStabilization('input/p2-1-0.mp4', 'output/nometempour') #or 11 for nathana, 12 for mine again
    """
    img1 = cv2.imread('input/p2-1-8.png')
    img2 = cv2.imread('input/p2-1-7.png')
    sift = Sift()
    des1 = None
    des2 = None
    try:
        des1 = np.load('kpd1.npy')
        des2 = np.load('kpd2.npy')
        print('loaded')
    except IOError:
        des1 = sift.get_descriptors(img1, outname='im1')
        des2 = sift.get_descriptors(img2, outname='im2')
        np.save('kpd1.npy', des1)
        np.save('kpd2.npy', des2)
        print('ioerror')

    #print("des1", des1)
    #print("des2", des2)
    #return

    imt = ImageTransform()
    imt.find_and_apply_transformation_no_src(des1, des2, img1, img2)
    """

if __name__ == '__main__':
   main()


