import cv2
import numpy as np
import math
from utils import *
import copy as cp
from scipy import ndimage
from pylab import *

################  HW2  #####################
# Nathana Facion                 RA:191079
# Rafael Mariottini Tomazela     RA:192803
############################################

def interest_points_opencv(file_name, file_final):
    img = cv2.imread(file_name)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(gray,kp,img)
    cv2.imwrite(file_final,img)

def interest_points(file_name, file_final, level):
    sigma = math.sqrt(2)
    img = cv2.imread(file_name)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Key localization
    vpyramid = []
    vdownsampling = []
    B = cp.copy(gray)
    # There-fore, we expand the input image by a factor of 2, using bilinear interpolation...
    height, width = B.shape[:2]
    B =  cv2.resize(B, (round(2*width), round(2*height)), interpolation = cv2.INTER_LINEAR )
    for i in range(0,level):
        gaussian_mask = create_gaussian_mask(7,sigma)
        B = cv2.filter2D(B, -1, gaussian_mask, borderType=cv2.BORDER_REPLICATE)
        A = cp.copy(B)
        debug('A',A.astype('uint8'))
        gaussian_mask = create_gaussian_mask(7,sigma)
        B = cv2.filter2D(B, -1, cv2.transpose(gaussian_mask), borderType=cv2.BORDER_REPLICATE)
        debug('B',B.astype('uint8'))
        difference = np.subtract(B , A)
        vpyramid.append(cp.deepcopy(difference))
        debug('pir',difference.astype('uint8'))
        height, width = B.shape[:2]
        B = cv2.resize(B, (round(0.5 * width), round(0.5 * height)),fx = 1.5, fy = 1.5, interpolation = cv2.INTER_LINEAR )
        # https://stackoverflow.com/questions/13242382/resampling-a-numpy-array-representing-an-image
        # http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#resize


    # http://www.cs.ubc.ca/~lowe/papers/iccv99.pdf

def create_image(file_name):
    cap = cv2.VideoCapture(file_name)
    while(cap.isOpened()):
        ret, frame = cap.read()
        #cv2.imshow('frame',frame)
        if ret == True:
            cv2.imwrite("output/p2-1-3-" + str(numFile()) + ".png", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
                break
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Create image by video
    create_image('input/p2-1-3.mp4')

    # Get interest points // Remover after
    interest_points_opencv('output/p2-1-3-0.png','output/p2-1-3-'+ str(numFile()) + '.png')

    # Get interest points
    interest_points('output/p2-1-3-0.png','output/p2-1-3-'+ str(numFile()) + '.png',4)


if __name__ == '__main__':
   main()


