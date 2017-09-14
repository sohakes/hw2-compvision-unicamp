import cv2
import numpy as np
from utils import *
import math

class Sift:
    #in the original paper, he recommends 5 levels
    def _get_gaussian_pyramid(self, img, n_octaves=4, n_levels_octave=5, kernel_size = 7, gaussian = math.sqrt(2))
        assert n_levels_octave >= 4
        assert n_octaves >= 3
        octaves = []
        processing_img = img
        for i in range(n_octaves):
            current_octave = [processing_img]
            w, h = processing_img.shape
            for j in range(1, n_levels_octave):
                processing_img = cv2.GaussianBlur(processing_img, kernel_size, kernel_size)
                current_octave.append(processing_img)

            octaves.append(current_octave)

            #it's the second level of the octave
            processing_img = cv2.resize(current_octave[1], (w/2, h/2), interpolation= cv2.INTER_LINEAR)

        return octaves

        #difference of gaussians
    def _get_dog_pyramid(self, g_octaves):
        dog_octaves = []
        for current_g_octave in g_octaves:
            current_dog_octave = []
            for j in range(1, len(current_g_octave)):
                current_dog_octave.append(current_g_octave[i] - current_g_octave[i-1])

            dog_octaves.append(current_dog_octave)

        return dog_octaves

    #there is probably a better way to code this
    #used the ones here http://www.ipol.im/pub/art/2014/82/article_lr.pdf
    def _get_hessian_values(w, s, m, n):
        h = {}
        h[(1, 1)] = w[s+1][m,n] + w[s-1][m,n] - 2*w[s][m,n]
        h[(2, 2)] = w[s][m+1,n] + w[s][m-1,n] - 2*w[s][m,n]
        h[(3, 3)] = w[s][m,n+1] + w[s][m,n-1] - 2*w[s][m,n]
        h[(1, 2)] = (w[s+1][m+1,n] - w[s+1][m-1,n] - w[s-1][m+1,n] + w[s-1][m-1,n])/4
        h[(1, 3)] = (w[s+1][m,n+1] - w[s+1][m,n-1] - w[s-1][m,n+1] + w[s-1][m,n-1])/4
        h[(2, 3)] = (w[s][m+1,n+1] - w[s][m+1,n-1] - w[s][m-1,n+1] + w[s][m-1,n-1])/4
        return h

    def _calc_interpolate_hessian(w, s, m, n):
        h = self._get_hessian_values(w, s, m, n)
        return np.matrix([[h[(i+1, j+1) for j in range(3)]] for i in range(3)])

    def _calc_edge_hessian(w, s, m, n):
        h = self._get_hessian_values(w, s, m, n)
        return np.matrix([[h[(i+1, j+1) for j in range(2)]] for i in range(2)])

    def _calc_gradient(w, s, m, n):
        return np.matrix([  [(w[s+1][m,n] - w[s-1][m,n])/2],
                            [(w[s][m+1,n] - w[s][m-1,n])/2],
                            [(w[s][m,n+1] - w[s][m,n-1])/2]])

    def _calc_alpha(hess, grad):
        return -linalg.inv(hess)*grad

    def _get_subpixel_keypoint(w, s, m, n):
        hess = self._calc_interpolate_hessian(w,s,m,n)
        grad = self.__calc_gradient(w,s,m,n)
        alpha = _calc_alpha(hess, grad)
        i = 1
        #extrema is another point
        while max(alpha) > 0.5 and i <= 5:
            i += 1
            s, m, n = (math.round(s + alpha[0]), math.round(m + alpha[1]), math.round(n + alpha[2]))
            w, h = w[0].shape
            #point outside the scope, discard it
            if s < 1 or s >= len(w) or m < 1 or m >= w - 1 or n < 1 or n >= h - 1:
                return None

            hess = self._calc_interpolate_hessian(w,s,m,n)
            grad = self.__calc_gradient(w,s,m,n)
            alpha = _calc_alpha(hess, grad)

        #it's no extrema
        if i == 6:
            return None
        
        s, m, n = (math.round(s + alpha[0]), math.round(m + alpha[1]), math.round(n + alpha[2]))
        kpval = w[s][m,n] + 0.5 * alpha.transpose() * grad

        return kpval, s, m, n
            

    def _get_minimas_maximas(self, dog_octaves):
        maximas = []
        minimas = []
        #important to know: the downscale must be /2, if I change that, must be changed here
        for i in range(len(dog_octaves)):
            current_dog_octave = dog_octaves[i]
            #just the middle ones
            for j in range(1, len(current_dog_octave) - 1)
                current_img = current_dog_octave[j]
                w, h = current_img.shape
                for x, y in zip(range(1, w - 1), range(1, h - 1)):
                    combs = [(x - a, y - b, j - c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
                    combs.remove(0, 0, 0)

                    is_minima = True
                    is_maxima = True
                    #not sure if there can me more than one extrema with same value, I think it can
                    #also, easy to optimize by removing already compared points
                    for c in combs:
                        nx, ny, nj = c
                        if current_img[x, y] > current_dog_octave[nj][nx, ny]:
                            is_minima = False
                            break

                    for c in combs:
                        nx, ny, nj = c
                        if current_img[x, y] < current_dog_octave[nj][nx, ny]:
                            is_maxima = False
                            break                   

                    kpval, s, m, n = self._get_subpixel_keypoint(current_img, x, y)

                    #discard values lower that 0.03 as the paper says (8 in range [0, 255])
                    if math.abs(kpval) <= 8:
                        continue
                    
                    #really rough approximation, can skew things honestly, but tries to get the middle pixels
                    #size_factor = 2**(i - 1)
                    #(x, y) = (int((size_factor*x + size_factory*(x+1)/2)), int((size_factor*y + size_factory*(y+1)/2)))

                    if is_maxima:
                        maximas.append((kpval, s, m, n))
                    if is_minima:
                        minimas.append((kpval, s, m, n))
        
    return (minimas, maximas)
                    

    def get_descriptors(self, img_color, n_octaves=4, n_levels_octave=4, kernel_size=7, gaussian=math.sqrt(2)):
        #img must be black and white
        img = cv2.CvtColor(img_color, cv2.COLOR_BGR2GRAY).astype(float)
        w, h = img.shape
        img = cv2.resize(img, (w*2, h*2), interpolation= cv2.INTER_LINEAR)
        g_octaves = self._get_gaussian_pyramid(img, n_octaves, n_levels_octave, kernel_size, gaussian)
        dog_octaves = self._get_dog_pyramid(g_octaves)
        (minimas, maximas) = self._get_minimas_maximas(dog_octaves)
        #TODO: step to improve by removing low contrast areas, but maybe whatever this

        return (kp, des)

    def __init__():
        pass