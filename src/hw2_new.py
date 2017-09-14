import cv2
import numpy as np
import math
from utils import *
from Sift import *
from OpenCVVideoStabilization import *

################  HW2  #####################
# Nathana Facion                 RA:191079
# Rafael Mariottini Tomazela     RA:192803
############################################



def main():
    # Create image by video
    #create_image('input/p2-1-3.mp4')

    # Get interest points // Remover after
    #img1 = cv2.imread('input/p2-1-1.png')
    #img2 = cv2.imread('input/p2-1-2.png')
    #img1 = cv2.copyMakeBorder(img1, 256, 256, 512, 256, cv2.BORDER_CONSTANT, 0)
    #img2 = cv2.copyMakeBorder(img2, 256, 256, 512, 322, cv2.BORDER_CONSTANT, 0)
    #stab = OpenCVImageTransform()
    #res = stab.show_matched(img1, img2)
    #debug("res img", res)
    # Get interest points
    #interest_points('input/p2-1-1.png','output/p2-1-3-'+ str(numFile()) + '.png',4)
    #stab = OpenCVVideoStabilization('input/p2-1-9.mp4', 'output/nometemp')
    #img1 = cv2.imread('input/p1-1-10.png')
    img1 = cv2.imread('input/p2-1-1.png')
    sift = Sift()
    sift.get_descriptors(img1)

if __name__ == '__main__':
   main()


