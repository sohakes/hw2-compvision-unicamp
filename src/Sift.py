import cv2
import numpy as np
from utils import *
import math
import itertools

class Sift:
    #I let it be constant at 0.5, I doubt there is a reason to change that    
    def _calc_delta(self, octave_level): 
        return 0.5*(2**(octave_level))

    #actually I meant sigma by gaussian, but I wil not change now
    def _get_gaussian_to_apply_given_level(self, gaussian, level, n_levels_octave):
        #return 1.7*gaussian**level
        #return (2**i)*gaussian*(2**(float(j)/float(n_levels_octave)))
        return (gaussian*2)*math.sqrt((2**((2*level)/(n_levels_octave-2))) - (2**(((2*(level-1)))/(n_levels_octave-2))))

    def _get_real_gaussian_given_level(self, gaussian, level, octave_level, n_levels_octave):
        return 2*self._calc_delta(octave_level)*gaussian*(2**(level/(n_levels_octave-2)))

    def _apply_gaussian(self, img, gaussian, level, n_levels_octave):
        return cv2.GaussianBlur(img, (0, 0), self._get_gaussian_to_apply_given_level(gaussian, level, n_levels_octave))

    #in the original paper, he recommends 5 levels
    def _get_gaussian_pyramid(self, img, n_octaves=4, n_levels_octave=5, kernel_size = 7, gaussian = 0.8):
        assert n_levels_octave >= 4
        assert n_octaves >= 3
        octaves = []
        first_sigma = 2*math.sqrt((0.8)**2-(0.5)**2)
        processing_img = cv2.GaussianBlur(img, (0, 0), first_sigma)
        for i in range(n_octaves):
            current_octave = [processing_img]
            w, h = processing_img.shape
            for j in range(1, n_levels_octave):
                #TEMP
                #sigma=math.pow(math.sqrt(2),j)*1.7
                #histogram_size= int(math.ceil(7*sigma))
                #histogram_size= 2*histogram_size+1                
                #processing_img = cv2.GaussianBlur(current_octave[0], (histogram_size, histogram_size), sigma, sigma)
                #debug('gauss', processing_img)
                ###
                #processing_img = cv2.GaussianBlur(processing_img, (0, 0), gaussian)
                processing_img = self._apply_gaussian(current_octave[-1], gaussian, j, n_levels_octave)
                #processing_img = cv2.GaussianBlur(current_octave[0], (0, 0), ((2**i)*gaussian*2**(float(j)/float(n_levels_octave))))
                current_octave.append(processing_img)

            octaves.append(current_octave)

            #it's the second level of the octave
            processing_img = cv2.resize(current_octave[n_levels_octave - 3], (int(h/2), int(w/2)), interpolation= cv2.INTER_LINEAR)

        return octaves

        #difference of gaussians
    def _get_dog_pyramid(self, g_octaves):
        dog_octaves = []
        for current_g_octave in g_octaves:
            current_dog_octave = []
            for j in range(1, len(current_g_octave)):
                current_dog_octave.append(current_g_octave[j] - current_g_octave[j-1])

            dog_octaves.append(current_dog_octave)

        return dog_octaves

    #if i want to invert:
    #w\[(s.*?)\]\[m(.*?),n(.*?)\]
    #w[$1][m$3,n$2]
    #there is probably a better way to code this
    #used the ones here http://www.ipol.im/pub/art/2014/82/article_lr.pdf
    def _get_hessian_values(self, w, s, m, n):
        h = {}
        h[(1, 1)] = w[s+1][m,n] + w[s-1][m,n] - 2*w[s][m,n]
        h[(2, 2)] = w[s][m+1,n] + w[s][m-1,n] - 2*w[s][m,n]
        h[(3, 3)] = w[s][m,n+1] + w[s][m,n-1] - 2*w[s][m,n]
        h[(1, 2)] = (w[s+1][m+1,n] - w[s+1][m-1,n] - w[s-1][m+1,n] + w[s-1][m-1,n])/4
        h[(1, 3)] = (w[s+1][m,n+1] - w[s+1][m,n-1] - w[s-1][m,n+1] + w[s-1][m,n-1])/4
        h[(2, 3)] = (w[s][m+1,n+1] - w[s][m+1,n-1] - w[s][m-1,n+1] + w[s][m-1,n-1])/4
        #print(w[s+1][m,n], w[s-1][m,n], 2*w[s][m,n])
        #print('w', w, s, m, n)
        #print(h)
        return h

    def _calc_interpolate_hessian(self, w, s, m, n):
        h = self._get_hessian_values(w, s, m, n)
        return np.matrix([  [h[(1, 1)], h[(1, 2)], h[(1, 3)]],
                            [h[(1, 2)], h[(2, 2)], h[(2, 3)]],
                            [h[(1, 3)], h[(2, 3)], h[(3, 3)]]])

    def _calc_edge_hessian(self, w, s, m, n):
        h = self._get_hessian_values(w, s, m, n)
        return np.matrix([  [h[(1, 1)], h[(1, 2)]],
                            [h[(1, 2)], h[(2, 2)]]])
        

    def _calc_gradient(self, w, s, m, n):
        return np.matrix([  [(w[s+1][m,n] - w[s-1][m,n])/2],
                            [(w[s][m+1,n] - w[s][m-1,n])/2],
                            [(w[s][m,n+1] - w[s][m,n-1])/2]])

    #gotten from here https://stackoverflow.com/questions/17931613/how-to-decide-a-whether-a-matrix-is-singular-in-python-numpy
    def _is_invertible(self, a):
        return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

    def _calc_alpha(self, hess, grad):
        #print(hess)
        #print(grad)
        if self._is_invertible(hess):
            return -np.linalg.inv(hess)*grad
        else:
            return None

    

    def _get_subpixel_keypoint(self, w, s, m, n, octa, gaussian, total_octaves):
        hess = self._calc_interpolate_hessian(w,s,m,n)
        grad = self._calc_gradient(w,s,m,n)
        alpha = self._calc_alpha(hess, grad)
        if alpha is None:
            #print("not invertible", hess)
            #print(w[s][m-3:m+3,n-3:n+3])
            return None
        i = 1
        #extrema is another point
        while abs(alpha).max() > 0.5 and i <= 5:
            i += 1
            #print('alpha', alpha)
            s, m, n = (int(round(s + alpha[0,0])), int(round(m + alpha[1,0])), int(round(n + alpha[2,0])))
            wid, h = w[0].shape
            #point outside the scope, discard it
            if s < 1 or s >= len(w) - 1 or m < 1 or m >= wid - 1 or n < 1 or n >= h - 1:
                #print("out of bounds on rec alpha")
                return None

            hess = self._calc_interpolate_hessian(w,s,m,n)
            grad = self._calc_gradient(w,s,m,n)
            alpha = self._calc_alpha(hess, grad)
            if alpha is None:
                #print("not invertible2", hess)
                return None

        #it's no extrema
        if i == 6:
            #print("no matches")
            return None

        scale = self._calc_delta(octa)    
        
        #real x, y in the real image
        x, y = scale * (m + alpha[1,0]), scale * (n + alpha[2,0])

        #real gamma
        gamma = self._get_real_gaussian_given_level(gaussian, s + alpha[0,0], octa, total_octaves)

        #discrete keypoint in the scale
        s, m, n = (int(round(s + alpha[0,0])), int(round(m + alpha[1,0])), int(round(n + alpha[2,0])))
        #print(alpha, s, m, n, i)
        kpval = w[s][m,n] + 0.5 * alpha.transpose() * grad

        return kpval, s, m, n, x, y, gamma

    def _calc_edgeness(self, w, s, m, n):
        hess = self._calc_edge_hessian(w, s, m, n)
        trace = hess.trace()
        det = np.linalg.det(hess)
        return (trace**2)/det

    def _does_discard_keypoint(self, w, s, m, n, c_edge):
        return self._calc_edgeness(w,s,m,n) > ((c_edge +1)**2)/c_edge

    def _get_contribution(self, dog, keypoint, lam):
        kpval, s, m, n, doc_oct, rx, ry, gamma = keypoint

    #lam=1.5
    def _calculate_histogram(self, dog, keypoint, lam):
        _, s, _, _, doc_oct, x, y, gamma = keypoint
        delta = self._calc_delta(dog_oct)
        w = dog[doc_oct]
        wid, h = w[s].shape
        rang = (3 * lam * gamma)
        minx = math.ceil((-rang+x)/delta)
        maxx = math.floor(1 + (rang+x)/delta)
        miny = math.ceil((-rang+y)/delta)
        maxy = math.floor(1 + (rang+y)/delta)
        if minx < 0 or maxx > wid or miny < 0 or maxy > h:
            return None

        
        
        #return w[s][minx:maxx, miny:maxy]
        

    def _gradient_m_n(self, w, s, m, n):
        return (0.5*(w[s][m+1][n] - w[s][m-1][n]), 0.5*(w[s][m][n+1] - w[s][m][n-1]))
            

    def _get_minimas_maximas(self, dog_octaves, c_edge):
        maximas = []
        minimas = []
        #print('important!', len(dog_octaves), len(dog_octaves[0]))
        #important to know: the downscale must be /2, if I change that, must be changed here
        for i in range(len(dog_octaves)):
            doc_oct = i
            current_dog_octave = dog_octaves[i]
            #just the middle ones
            kk = 0
            #for ts in range(0, len(current_dog_octave)):
            #    debug('curr_img', (current_dog_octave[ts]*255).astype('uint8'))

            for ts in range(1, len(current_dog_octave) - 1):
                current_img = current_dog_octave[ts]
                tmp_img = (current_img.copy()*255).astype('uint8')
                #debug('curr_img', (current_img*255).astype('uint8'))
                w, h = current_img.shape
                for x, y in itertools.product(range(1, w - 1), range(1, h - 1)):
                    s = ts
                    kk +=1
                    combs = [(x - a, y - b, s - c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (0, 0, 0)]
                    combs.remove((x, y, s))

                    is_minima = True
                    is_maxima = True
                    #not sure if there can me more than one extrema with same value, I think it can
                    #also, easy to optimize by removing already compared points
                    #print("try", s, x, y, current_img.shape)
                    for c in combs:
                        
                        nx, ny, nj = c
                        if current_img[x, y] > current_dog_octave[nj][nx, ny]:
                            #print(current_img[x, y], current_dog_octave[nj][nx, ny])
                            #print("not minima", c, x, y)
                            is_minima = False
                            break

                    for c in combs:
                        nx, ny, nj = c
                        if current_img[x, y] < current_dog_octave[nj][nx, ny]:
                            #print(current_img[x, y], current_dog_octave[nj][nx, ny])
                            #print("not maxima", c)
                            is_maxima = False
                            break               

                    if is_minima == False and is_maxima == False:
                        #print("not either")
                        continue  
                    
                    #print(current_img[x,y], x, y, "CURR IMG XY")
                    subpixel = self._get_subpixel_keypoint(current_dog_octave, s, x, y, doc_oct, gaussian, total_octaves)
                    if subpixel is None:
                        #print('no subpixel')
                        continue

                    #kpval, s, m, n = current_img[x, y], s, x, y
                    kpval, s, m, n, rx, ry, gamma = subpixel
                    #kpval = kpval[0,0]
                    #probably needs to check if not already added

                    tmp_img = cv2.circle(tmp_img,(n,m), 1, (255,255,255), 1)

                    #discard values lower that 0.03 as the paper says
                    if abs(kpval) <= 0.03 or self._does_discard_keypoint(current_dog_octave, s, m, n, c_edge):
                        #print('nope', abs(kpval) <= 0.03, kpval)
                        continue
                    #print('HERE!')
                    #really rough approximation, can skew things honestly, but tries to get the middle pixels
                    #size_factor = 2**(i - 1)
                    #(x, y) = (int((size_factor*x + size_factory*(x+1)/2)), int((size_factor*y + size_factory*(y+1)/2)))

                    if is_maxima:
                        maximas.append((kpval, s, m, n, doc_oct, rx, ry, gamma))
                    if is_minima:
                        minimas.append((kpval, s, m, n, doc_oct, rx, ry, gamma))
                #debug('circles', tmp_img)
        #print("the mighty", kk)
        return (minimas, maximas)

    def _describe_keypoints(self, dog_octaves, c_edge, minimas, maximas):

                    
    #IMPORTANT: the c_edge value is for [0,1], I think it's still ok since it's a ratio
    def get_descriptors(self, img_color, n_octaves=4, n_levels_octave=5, kernel_size=7, gaussian=0.8, c_edge=10):
        #img must be black and white
        img_u = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY).astype(float)/255      
        w, h = img_u.shape 
        img = cv2.resize(img_u, dsize=(h*2, w*2), interpolation= cv2.INTER_LINEAR)        
        g_octaves = self._get_gaussian_pyramid(img, n_octaves, n_levels_octave, kernel_size, gaussian)
        dog_octaves = self._get_dog_pyramid(g_octaves)
        (minimas, maximas) = self._get_minimas_maximas(dog_octaves, c_edge)
        #TODO: step to improve by removing low contrast areas, but maybe whatever this
        print('minmax',minimas + maximas)
        img_temp = img_color
        print(img_temp.shape)
        for kpval, s, m, n, octv, rx, ry, gamma in minimas + maximas:
            print((m*(2**(octv-1)), n*(2**(octv-1))))
            img_temp = cv2.circle(img_temp,(int(round(ry)), int(round(rx))), 2, (0,255,0), 2)

        debug('img', img_temp)

        #return (kp, des)

    def __init__(self):
        pass