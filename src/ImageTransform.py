import cv2
import numpy as np
from utils import *
#from matplotlib import pyplot as plt

class ImageTransform:

    def _sift(self, img):
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray,None)

        return (kp, des)

    #def _fit_model(self):

    def _match(self, img1, img2):
        kp1, des1 = self._sift(img1)
        kp2, des2 = self._sift(img2)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            #verify if the distance between the first and second is big enough
            if m.distance < 0.7*n.distance: 
                good.append(m)
        # cv2.drawMatchesKnn expects list of lists as matches.
        #print("kp", kp1)
        #print("kp", kp1[1])
        #print("des", des1)
        #print("good", good)
        #img3 = None
        #img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good, img3,flags=2)
        #return img3
        #plt.imshow(img3),plt.show()
        return ((kp1, des1), (kp2, des2), good)

    def show_matched(self, img2, img1):
        img1 = cv2.copyMakeBorder(img1, 100, 100, 100, 100, cv2.BORDER_CONSTANT, 0)
        img2 = cv2.copyMakeBorder(img2, 100, 100, 100, 100, cv2.BORDER_CONSTANT, 0)
        ((kp1, des1), (kp2, des2), good) = self._match(img1, img2)
        #print(kp1, des1, good)
        if len(good)>8:
            dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            #matchesMask = mask.ravel().tolist()
            h,w,d = img1.shape
            #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.warpPerspective(img2,M, (w, h))
            


            #print(dst)
            #debug("transform", img2)
            #debug("transform", img1)            
            #debug("transform", dst)
            #dstf = np.maximum(img1, dst)
            #debug("transform", dstf)
            #img3 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            return dst
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None
            return None

        

        #debug("matched", self._match(img1, img2))


    def __init__(self):
        #img1 = cv2.copyMakeBorder(img1, 256, 256, 512, 256, cv2.BORDER_CONSTANT, 0)
        #img2 = cv2.copyMakeBorder(img2, 256, 256, 512, 322, cv2.BORDER_CONSTANT, 0)
        pass