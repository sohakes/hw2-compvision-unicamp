import cv2
import numpy as np
from utils import *
from math import*
import operator
import random
import itertools
from Sift import *
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
            if d1[1] < 0.7 * d2[1]:
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
            #print("points!", pt1, pt2)
            cv2.line(comb_image, pt1, pt2,40,1)

        debug('combined image', comb_image)
        cv2.imwrite('output/combinedimg2.png', comb_image)

    def draw_lines_matches_received(self, kpd1, kpd2, img1, img2, matches):
        w1, h1, d = img1.shape
        w2, h2, d = img2.shape
        #print(w1, w2, h1, h2)
        comb_image = np.zeros((max(w1,w2), h1 + h2, d), np.uint8)
        comb_image[0:w1, 0:h1, :] = img1
        comb_image[0:w2, h1:h1+h2, :] = img2
        #print(des1)
        for idxs, dist in matches:
            idx1, idx2 = idxs
            pt1 = (kpd1[idx1][0][0], kpd1[idx1][0][1])
            pt2 = (kpd2[idx2][0][0] + h1, kpd2[idx2][0][1])
            #print("points!", pt1, pt2)
            cv2.line(comb_image, pt1, pt2,40,1)

        debug('combined image', comb_image)
        cv2.imwrite('output/combinedimg2rec.png', comb_image)

    #return matrixes X and Y
    def fill_matrix_points_XY(self, kpd1, kpd2, matches):
        X = []
        Y = []
        for idxs, dist in matches:
            idx1, idx2 = idxs
            x1, y1 = (kpd1[idx1][0][0], kpd1[idx1][0][1])
            x2, y2 = (kpd2[idx2][0][0], kpd2[idx2][0][1])
            X.append([x1, y1, 1, 0, 0, 0])
            X.append([0, 0, 0, x1, y1, 1])
            Y.append([x2])
            Y.append([y2])

        return np.matrix(X), np.matrix(Y)

    def find_affine_matrix(self, X, Y):
        a = X.transpose() * X
        #print('affine', a)
        if a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]:
            return (np.linalg.inv(a)) * (X.transpose() * Y)
        return None

    #match is ((idx1, idx2), dist)
    def ransac(self, kpd1, kpd2, matches):
        thresh = 5
        p = 0.99
        n = 2000 #infinite, or just big enough

        X, Y = self.fill_matrix_points_XY(kpd1, kpd2, matches)
        best_inliers = []
        #best_A = []
        iterations = 0
        A = None
        while A is None and iterations < n*5:

            while n > iterations:
                #print(iterations, n)
                #find affine matrix for three points
                selected_matches = random.sample(matches, 3)
                Xsamp, Ysamp = self.fill_matrix_points_XY(kpd1, kpd2, selected_matches)
                #print('xsamp, ysamp',Xsamp, Ysamp)
                Asamp = self.find_affine_matrix(Xsamp, Ysamp)
                #print('asamp', Asamp)
                if Asamp is None:
                    continue

                #check the rest
                Ytest = X*Asamp
                inliers = []
                #print('start', Asamp)
                for i in range(1, len(Ytest), 2):
                    x1, y1 = Ytest[i-1, 0], Ytest[i, 0]
                    x2, y2 = Y[i-1,0], Y[i,0]
                    dist = math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2))
                    #print('dist', dist, x1, y1, x2, y2)
                    if dist < thresh:
                        #print('done', (i-1)/2)
                        inliers.append(matches[int((i-1)/2)]) #append the match

                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    #best_A = Asamp
                    w = len(inliers)/len(matches)
                    #n = np.log(1-p)/np.log(1-w**n)
                #print('inliers, best',inliers, best_inliers)

                iterations += 1

            #find affine for all the inliers now
            best_inliers2 = [((x[0][1], x[0][0]), x[1]) for x in best_inliers]
            Xf, Yf = self.fill_matrix_points_XY(kpd2, kpd1, best_inliers2)
            A = self.find_affine_matrix(Xf, Yf)
            #Xf, Yf = self.fill_matrix_points_XY(kpd1, kpd2, best_inliers)
            #A = self.find_affine_matrix(Xf, Yf)


        return A, best_inliers

    def transform_image_into(self, A, img_src, img_dest, offset_x = 0, offset_y = 0):
        hei, wid, dep = img_dest.shape
        heis, wids, deps = img_src.shape
        hei -= 2*offset_y
        wid -= 2*offset_x
        f1 = lambda x1, y1: [x1, y1, 1, 0, 0, 0]
        f2 = lambda x1, y1: [0, 0, 0, x1, y1, 1]
        #f1 = lambda x1, y1: [x1]
        #f2 = lambda x1, y1: [y1]
        #original image
        X = np.matrix([f(x1, y1) for x1, y1 in itertools.product(range(-wid*2, wid*2), range(-hei*2, hei*2)) for f in (f1, f2)])
        #print(X)
        Y = X * A
        #print(X, Y)
        #print(A)
        #a, b, c, d, e, f = A[0,0], A[1,0], A[2,0], A[3,0], A[4,0], A[5,0]

        for i in range(1, len(X), 2):
            xd, yd = X[i,3], X[i,4]
            q, k = yd, xd

            x, y =Y[i-1,0] , Y[i,0]

            #x = (-b * f + b * q + c * e - e * k)/(b * d - a * e)
            #y = (a*(f-q) - c * d + d * k)/(b * d - a * e)
            
            #x, y = Y[i-1,0], Y[i,0]
            x1, y1, x2, y2 = floor(x), floor(y), ceil(x), ceil(y)
            if x1 >= wids or x2 >= wids or y1 >= heis or y2 >= heis or np.min([x1, x2, y1, y2]) < 0:
                continue
            for d in range(dep):
                
                #pq11, pq12, pq21, pq22 = (x1, y1), (x1, y2), (x2, y1), (x2, y2)
                fq11, fq12, fq21, fq22 = img_src[y1, x1, d], img_src[y1, x2, d], img_src[y2, x1, d], img_src[y2, x2, d]
                #print('fqs', fq11, fq12, fq21, fq22)
                frac = x2 - x1
                fxy1 = fq11
                fxy2 = fq12
                if frac != 0:          
                    fxy1 = ((x2 - x)/(frac))*fq11 + ((x - x1)/(frac))*fq21
                    fxy2 = ((x2 - x)/(frac))*fq12 + ((x - x1)/(frac))*fq22
                

                frac2 = y2-y1
                fxy = fxy1
                if frac2 != 0:
                    fxy = ((y2-y)/(y2-y1))*fxy1 + ((y-y1)/(y2-y1))*fxy2

                

                #print(int(round(fxy)))
                
                img_dest[yd+offset_y, xd+offset_x, d] = int(round(fxy))


        
        return img_dest

    def find_and_apply_transformation(self, kpd1, kpd2, img1, img2):
        des1 = [k[1] for k in kpd1]
        des2 = [k[1] for k in kpd2]
        #print(des1)
        matches = self.knn_match(des1, des2)
        A, n_matches = self.ransac(kpd1, kpd2, matches)
        dest = cv2.copyMakeBorder(img2, 600, 600, 600, 600, cv2.BORDER_CONSTANT, 0)
        self.draw_lines_matches(kpd1, kpd2, img1, img2)
        self.draw_lines_matches_received(kpd1, kpd2, img1, img2, n_matches)
        img_dest = self.transform_image_into(A, img1, dest, 600, 600)

        debug('transformed_image', img_dest)

    def find_and_apply_transformation_no_src(self, kpd1, kpd2, img1, img2):
        des1 = [k[1] for k in kpd1]
        des2 = [k[1] for k in kpd2]
        #print(des1)
        matches = self.knn_match(des1, des2)
        A, n_matches = self.ransac(kpd1, kpd2, matches)
        if A is None:
            return None
        wid, hei, dep = img1.shape
        dest = np.zeros((wid + 250, hei + 250, dep), np.uint8)
        self.draw_lines_matches(kpd1, kpd2, img1, img2)
        self.draw_lines_matches_received(kpd1, kpd2, img1, img2, n_matches)
        img_dest = self.transform_image_into(A, img1, dest, 125, 125)

        debug('transformed_image', img_dest)

        return img_dest
        
    def show_matched(self, img2, img1):
        sift = Sift()
        des1 = sift.get_descriptors(img1, outname='im1')
        des2 = sift.get_descriptors(img2, outname='im2')
        dest = self.find_and_apply_transformation_no_src(des1, des2, img1, img2)
        debug('img_dest', dest)
        return dest

    def set_first_frame(self, img2):
        sift = Sift()
        self.des2 = sift.get_descriptors(img2, outname='im2')


    def show_matched_reuse(self, img2, img1):
        sift = Sift()
        des1 = sift.get_descriptors(img1, outname='im1')
        dest = self.find_and_apply_transformation_no_src(des1, self.des2, img1, img2)
        debug('img_dest', dest)
        return dest
     
    """
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

    """

    def __init__(self):
        #img1 = cv2.copyMakeBorder(img1, 256, 256, 512, 256, cv2.BORDER_CONSTANT, 0)
        #img2 = cv2.copyMakeBorder(img2, 256, 256, 512, 322, cv2.BORDER_CONSTANT, 0)
        pass
