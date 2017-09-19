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

def write_transform(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    sift = Sift()
    des1 = sift.get_descriptors(img1)
    des2 = sift.get_descriptors(img2)
    imt = ImageTransform()
    im = imt.find_and_apply_transformation(des1, des2, img1, img2, 600)
    write_image(4, im, True)

def main():
    print("First set of images on sift")
    write_transform('input/p2-1-0.png', 'input/p2-1-1.png')

    print("Second set of images on sift")
    write_transform('input/p2-1-2.png', 'input/p2-1-3.png')

    print("Third set of images on sift")
    write_transform('input/p2-1-4.png', 'input/p2-1-5.png')


    print("Stabilizaing video")
    stab = VideoStabilization('input/p2-5-6.mp4', 'output/p2-5-0') #or 11 for nathana, 12 for mine again    

if __name__ == '__main__':
   main()


