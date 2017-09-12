import cv2
import numpy as np
import math
from utils import *
from OpenCVImageTransform import *

################  HW2  #####################
# Nathana Facion                 RA:191079
# Rafael Mariottini Tomazela     RA:192803
############################################



def main():
    # Create image by video
    #create_image('input/p2-1-3.mp4')

    # Get interest points // Remover after
    img1 = cv2.imread('input/p2-1-1.png')
    img2 = cv2.imread('input/p2-1-2.png')
    stab = OpenCVImageTransform(img1, img2)
    stab.show_matched()

    # Get interest points
    #interest_points('input/p2-1-1.png','output/p2-1-3-'+ str(numFile()) + '.png',4)


if __name__ == '__main__':
   main()


