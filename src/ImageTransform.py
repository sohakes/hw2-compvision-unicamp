import cv2
import numpy as np
from utils import *
from math import*
import operator
#from matplotlib import pyplot as plt


class ImageTransform:
    #receives two parameters which are lists of descriptors
    #returns a list with ((idx1, idx2), dist)
    def knn_match(self, des1l,des2l):
        def euclideanDistance(des1, des2):
            return np.sqrt(np.sum((des1 - des2)**2))

        def getNeighbors(des1, des2l, idx):
            distances = []
            for x in range(len(des2l)):
                dist = euclideanDistance(des1, des2l[x])
                distances.append(((idx, x), dist))
            distances.sort(key=operator.itemgetter(1))
            return distances[:2]

        list_matches =[]
        for x in range(len(des1l)):
            result =[]
            d1, d2 = getNeighbors(des1l[x],des2l, x)
            if d1[1] < 0.5 * d2[1]:
                list_matches.append(d1)

        return list_matches

    def draw_lines_matches(self, kpd1, kpd2, img1, img2):
        w1, h1, d = img1.shape
        w2, h2, d = img2.shape
        #print(w1, w2, h1, h2)
        comb_image = np.zeros((max(w1,w2), h1 + h2, d), np.uint8)
        comb_image[0:w1, 0:h1, :] = img1
        comb_image[0:w2, h1:h1+h2, :] = img2
        des1 = [k[1] for k in kpd1]
        des2 = [k[1] for k in kpd2]
        #print(des1)
        matches = self.knn_match(des1, des2)
        for idxs, dist in matches:
            idx1, idx2 = idxs
            pt1 = (kpd1[idx1][0][0], kpd1[idx1][0][1])
            pt2 = (kpd2[idx2][0][0] + h1, kpd2[idx2][0][1])
            print("points!", pt1, pt2)
            cv2.line(comb_image, pt1, pt2,40,1)

        #debug('combined image', comb_image)




    def _sift(self, img):
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray,None)
        
        img=cv2.drawKeypoints(img,kp, img)
        debug('sift', img)
        return (kp, des)

    #def _fit_model(self):

    def _match(self, img1, img2):
        kp1, des1 = self._sift(img1)
        kp2, des2 = self._sift(img2)
        # BFMatcher with default params
        #bf = cv2.BFMatcher()
        #matches = bf.knnMatch(des1,des2, k=2)
        # Apply ratio test
        matches = knn_match(des1,des2)

        good = []
        for m,n in matches:
           #verify if the distance between the first and second is big enough
           #if m.distance < 0.7*n.distance:
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
        return ((kp1, des1), (kp2, des2), matches)

    def show_matched(self, img2, img1):
        img1 = cv2.copyMakeBorder(img1, 100, 100, 100, 100, cv2.BORDER_CONSTANT, 0)
        img2 = cv2.copyMakeBorder(img2, 100, 100, 100, 100, cv2.BORDER_CONSTANT, 0)
        ((kp1, des1), (kp2, des2), good) = self._match(img1, img2)
        #print(kp1, des1, good)
        if len(good)>8:
            dst_pts = np.float32([ kp1[m[0]].pt for m in good ]).reshape(-1,1,2)
            src_pts = np.float32([ kp2[m[1]].pt for m in good ]).reshape(-1,1,2)
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
