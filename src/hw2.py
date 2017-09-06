import cv2
import numpy as np
import math
from utils import *
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

def interest_points(file_name, file_final):
    s = 1.6 # set the standard deviation
    k = 1.3  # 2**(1/s)
    img = cv2.imread(file_name)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #1) Criar uma Gaussiana com convolucao com nossa imagem
    #mask = create_gaussian_mask(5, 3)
    #conv = cv2.filter2D(gray, -1, mask, borderType=cv2.BORDER_REPLICATE)
    blur5 = cv2.GaussianBlur(gray,(5,5),s*(k**4))
    blur3 = cv2.GaussianBlur(gray,(3,3),s*(k**2))
    diffGaussian = blur5 - blur3
    debug('diffGaussian',diffGaussian)
    # https://www.maxwell.vrac.puc-rio.br/17050/17050_5.PDF

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
    interest_points('output/p2-1-3-0.png','output/p2-1-3-'+ str(numFile()) + '.png')

if __name__ == '__main__':
   main()


