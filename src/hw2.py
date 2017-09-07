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
    sigma = 3
    k = 2**(1/sigma)
    octave = 2
    img = cv2.imread(file_name)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # create Pyramids
    vpyramid = []
    vdownsampling = []
    image = cp.copy(gray)
    vpyramid.append(image)
    for i in range(1,level):
        image = ndimage.zoom(image,0.5, order =1)
        vpyramid.append(cp.deepcopy(image))
        #debug('pir',image.astype('uint8'))

    #Gaussian Blurring
    vgaussianblur = []
    vimgblur = []
    for j in range(octave):
        for i in range(level):
            sigma = math.pow(k,j)*1.7
            histogram_size= 2*(int(math.ceil(7*sigma)))+1
            imgblur = cv2.GaussianBlur(vpyramid[i],(histogram_size,histogram_size),sigma,sigma)
            vimgblur.append(imgblur)
        vgaussianblur.append(vimgblur)


    # Difference of Gaussian
    vimgdog = []
    vdog = []
    for i in range(0,octave):
        for j in range(0,level):
            difference = np.zeros(np.shape(vgaussianblur[i][0]))
            difference= np.subtract(vpyramid[j],vgaussianblur[i][j])
            #debug('hist',difference.astype('uint8'))
            #difference = vgaussianblur[i][j] - vgaussianblur[i-1][j]
            vimgdog.append(cp.deepcopy(difference))
        vdog.append(vimgdog)

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


