import cv2
import numpy as np
from utils import *
from math import*
import operator
import random
import itertools
from Sift import *

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

    def draw_lines_matches_received(self, kpd1, kpd2, img1, img2, matches, question):
        w1, h1, d = img1.shape
        w2, h2, d = img2.shape
        comb_image = np.zeros((max(w1,w2), h1 + h2, d), np.uint8)
        comb_image[0:w1, 0:h1, :] = img1
        comb_image[0:w2, h1:h1+h2, :] = img2

        for idxs, dist in matches:
            idx1, idx2 = idxs
            pt1 = (kpd1[idx1][0][0], kpd1[idx1][0][1])
            pt2 = (kpd2[idx2][0][0] + h1, kpd2[idx2][0][1])
            cv2.line(comb_image, pt1, pt2,(0, 255, 255),1)

        write_image(question, comb_image, self.save)


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
        if a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]:
            return (np.linalg.inv(a)) * (X.transpose() * Y)
        return None

    #match is ((idx1, idx2), dist)
    def ransac(self, kpd1, kpd2, matches):
        thresh = 4
        p = 0.99
        n = 10000 #infinite, or just big enough

        X, Y = self.fill_matrix_points_XY(kpd1, kpd2, matches)
        best_inliers = []
        iterations = 0
        A = None
        best_sum_dist = 100000
        while A is None and iterations < n*2:
            n = 10000
            while n > iterations:
                #find affine matrix for three points
                selected_matches = random.sample(matches, 3)
                Xsamp, Ysamp = self.fill_matrix_points_XY(kpd1, kpd2, selected_matches)
                Asamp = self.find_affine_matrix(Xsamp, Ysamp)
                if Asamp is None:
                    continue

                #check the rest
                Ytest = X*Asamp
                inliers = []
                sum_dist = 0
                for i in range(1, len(Ytest), 2):
                    x1, y1 = Ytest[i-1, 0], Ytest[i, 0]
                    x2, y2 = Y[i-1,0], Y[i,0]
                    dist = math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2))
                    if dist < thresh:
                        inliers.append(matches[int((i-1)/2)]) #append the match
                        sum_dist += dist

                if len(inliers) > len(best_inliers) or (len(inliers) == len(best_inliers) and best_sum_dist > sum_dist) :
                    best_sum_dist = sum_dist
                    best_inliers = inliers
                    #best_A = Asamp
                    w = len(inliers)/len(matches)

                iterations += 1

            #find affine for all the inliers now
            best_inliers2 = [((x[0][1], x[0][0]), x[1]) for x in best_inliers]
            Xf, Yf = self.fill_matrix_points_XY(kpd2, kpd1, best_inliers2)
            A = self.find_affine_matrix(Xf, Yf)

        return A, best_inliers

    def transform_image_into(self, A, img_src, img_dest, offset_x = 0, offset_y = 0):
        hei, wid, dep = img_dest.shape
        heis, wids, deps = img_src.shape
        hei -= 2*offset_y
        wid -= 2*offset_x
        f1 = lambda x1, y1: [x1, y1, 1, 0, 0, 0]
        f2 = lambda x1, y1: [0, 0, 0, x1, y1, 1]

        #original image
        X = np.matrix([f(x1, y1) for x1, y1 in itertools.product(range(0-offset_x, wid+offset_x), range(0-offset_y, hei+offset_y)) for f in (f1, f2)])

        Y = X * A

        for i in range(1, len(X), 2):
            xd, yd = X[i,3], X[i,4]
            q, k = yd, xd

            x, y = Y[i-1,0], Y[i,0]

            x, y  = int(round(x)), int(round(y))

            if x <=0 or y <= 0 or x >= wids or y >= heis:
                continue
            p =img_src[y,x, :]
            if (p == 0).all():
                continue
            img_dest[yd+offset_y, xd+offset_x, :] = p
         
        return img_dest

    def find_and_apply_transformation(self, kpd1, kpd2, img1, img2, padding=0):
        des1 = [k[1] for k in kpd1]
        des2 = [k[1] for k in kpd2]
        matches = self.knn_match(des1, des2)
        A, n_matches = self.ransac(kpd1, kpd2, matches)
        dest = cv2.copyMakeBorder(img2, padding, padding, padding, padding, cv2.BORDER_CONSTANT, 0)
        self.draw_lines_matches_received(kpd1, kpd2, img1, img2, matches, 2)
        self.draw_lines_matches_received(kpd1, kpd2, img1, img2, n_matches, 3)
        img_dest = self.transform_image_into(A, img1, dest, padding, padding)

        debug('transformed_image', img_dest)
        return img_dest

    def find_and_apply_transformation_no_src(self, kpd1, kpd2, img1, img2, padding=0):
        des1 = [k[1] for k in kpd1]
        des2 = [k[1] for k in kpd2]
        matches = self.knn_match(des1, des2)
        A, n_matches = self.ransac(kpd1, kpd2, matches)
        if A is None:
            return None
        wid, hei, dep = img1.shape
        dest = np.zeros((wid + 2*padding, hei + 2*padding, dep), np.uint8)
        self.draw_lines_matches_received(kpd1, kpd2, img1, img2, matches, 2)
        self.draw_lines_matches_received(kpd1, kpd2, img1, img2, n_matches, 3)
        img_dest = self.transform_image_into(A, img1, dest, padding, padding)

        debug('transformed_image', img_dest)

        return img_dest
        
    def show_matched(self, img2, img1):
        sift = Sift()
        des1 = sift.get_descriptors(img1, save = self.save)
        des2 = sift.get_descriptors(img2, save = self.save)
        dest = self.find_and_apply_transformation_no_src(des1, des2, img1, img2)
        
        debug('img_dest', dest)
        return dest

    def set_first_frame(self, img2):
        sift = Sift()
        self.des2 = sift.get_descriptors(img2, save = self.save)


    def show_matched_reuse(self, img2, img1):
        sift = Sift()
        des1 = sift.get_descriptors(img1, save = self.save)
        dest = self.find_and_apply_transformation_no_src(des1, self.des2, img1, img2)
        debug('img_dest', dest)
        return dest

    def __init__(self, save = True):
        self.save = save
        pass
