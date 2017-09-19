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
        return (gaussian*2)*math.sqrt((2**((2*level)/(n_levels_octave-2))) - (2**(((2*(level-1)))/(n_levels_octave-2))))

    def _get_real_gaussian_given_level(self, gaussian, level, octave_level, n_levels_octave):
        return 2*self._calc_delta(octave_level)*gaussian*(2**(level/(n_levels_octave-2)))

    def _apply_gaussian(self, img, gaussian, level, n_levels_octave):
        return cv2.GaussianBlur(img, (0, 0), self._get_gaussian_to_apply_given_level(gaussian, level, n_levels_octave))

    #in the original paper, he recommends 5 levels
    def _get_gaussian_pyramid(self, img, n_octaves=4, n_levels_octave=5, kernel_size = 7, gaussian = 0.8):
        assert n_levels_octave >= 3
        assert n_octaves >= 3
        octaves = []
        first_sigma = 2*math.sqrt((0.8)**2-(0.5)**2)
        processing_img = cv2.GaussianBlur(img, (0, 0), first_sigma)
        for i in range(n_octaves):
            current_octave = [processing_img]
            w, h = processing_img.shape
            for j in range(1, n_levels_octave):
                processing_img = self._apply_gaussian(current_octave[-1], gaussian, j, n_levels_octave)
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
        if self._is_invertible(hess):
            return -np.linalg.inv(hess)*grad
        else:
            return None

    

    def _get_subpixel_keypoint(self, w, s, m, n, octa, gaussian, total_octaves):
        hess = self._calc_interpolate_hessian(w,s,m,n)
        grad = self._calc_gradient(w,s,m,n)
        alpha = self._calc_alpha(hess, grad)
        if alpha is None:
            return None
        i = 1
        #extrema is another point
        while abs(alpha).max() > 0.5 and i <= 5:
            i += 1
            s, m, n = (int(round(s + alpha[0,0])), int(round(m + alpha[1,0])), int(round(n + alpha[2,0])))
            wid, h = w[0].shape
            #point outside the scope, discard it
            if s < 1 or s >= len(w) - 1 or m < 1 or m >= wid - 1 or n < 1 or n >= h - 1:
                return None

            hess = self._calc_interpolate_hessian(w,s,m,n)
            grad = self._calc_gradient(w,s,m,n)
            alpha = self._calc_alpha(hess, grad)
            if alpha is None:
                return None

        #it's no extrema
        if i == 6:
            return None

        scale = self._calc_delta(octa)    
        
        #real x, y in the real image
        x, y = scale * (m + alpha[1,0]), scale * (n + alpha[2,0])

        #real gamma
        gamma = self._get_real_gaussian_given_level(gaussian, s + alpha[0,0], octa, total_octaves)

        #discrete keypoint in the scale
        s, m, n = (int(round(s + alpha[0,0])), int(round(m + alpha[1,0])), int(round(n + alpha[2,0])))
        kpval = w[s][m,n] + 0.5 * alpha.transpose() * grad

        return kpval, s, m, n, x, y, gamma

    def _calc_edgeness(self, w, s, m, n):
        hess = self._calc_edge_hessian(w, s, m, n)
        trace = hess.trace()
        det = np.linalg.det(hess)
        return (trace**2)/det

    def _does_discard_keypoint(self, w, s, m, n, c_edge):
        return self._calc_edgeness(w,s,m,n) > ((c_edge +1)**2)/c_edge

    def _gradient_m_n(self, w, s, m, n):
        return (0.5*(w[s][m+1][n] - w[s][m-1][n]), 0.5*(w[s][m][n+1] - w[s][m][n-1]))

    #lam=1.5, n_bins=36, threshold=0.8
    def _calculate_reference(self, dog, keypoint, lam, n_bins, threshold):
        _, s, _, _, dog_oct, x, y, gamma = keypoint
        delta = self._calc_delta(dog_oct)
        w = dog[dog_oct]
        wid, h = w[s].shape
        rang = (3 * lam * gamma)
        minx = math.ceil((-rang+x)/delta)
        maxx = math.floor(1 + (rang+x)/delta)
        miny = math.ceil((-rang+y)/delta)
        maxy = math.floor(1 + (rang+y)/delta)

        if minx <= 0 or maxx >= wid or miny <= 0 or maxy >= h:
            return None
        
        bins = [0] * n_bins

        #calculating the histogram
        for m in range(minx, maxx):
            for n in range(miny,maxy):
                exponent = -(math.sqrt(math.pow(delta * m - x,2) + math.pow(delta * n - y,2)))/2*((lam*gamma)**2)
                grad = self._gradient_m_n(w, s, m, n)
                grad_norm = math.sqrt(grad[0]**2 + grad[1]**2)
                contribution = math.exp(exponent) * grad_norm

                bin_index = int(round((n_bins/(2*math.pi))*np.arctan2(grad[0], grad[1])))

                bins[bin_index] += contribution

        #smoothing histogram with 6 times circular convolution
        kernel = [1/3, 1/3, 1/3]
        bins = np.array(bins)
        for i in range(6):
            temp_bins = np.concatenate(([bins[-1]], bins, [bins[0]]))
            bins = np.convolve(temp_bins, kernel, mode='valid')
        
        #calculate reference point
        maxbin = max(bins)
        keypoints = []
        for k in range(len(bins)):
            hk = bins[k]
            hkm = bins[(k - 1) % n_bins]
            hkp = bins[(k + 1) % n_bins]
            if hk > hkm and hk > hkp and hk >= threshold*maxbin:
                thetak = 2*math.pi*(k-1)/n_bins
                theta = (thetak + (math.pi/n_bins)* ((hkm-hkp)/(hkm-2*hk+hkp)))
                keypoints.append(keypoint + (theta,))

        return keypoints

    #lam_descr = 6, n_hist = 4, n_ori = 8
    def _normalized_descriptor(self, dog, keypoint_theta, lam_descr, n_hist, n_ori):
        kpval, s, mt, nt, dog_oct, x, y, gamma, theta = keypoint_theta
        delta = self._calc_delta(dog_oct)
        def calc_normalized_cord(m, n):
            nx = ((m * delta - x) * math.cos(theta) + (n * delta - y) * math.sin(theta))/gamma
            ny = (-(m * delta - x) * math.sin(theta) + (n * delta - y) * math.cos(theta))/gamma
            return (nx, ny)

        wid, hei = dog[dog_oct][s].shape
        rang = math.sqrt(2) * lam_descr * gamma               

        ration_hist = ((n_hist+1)/n_hist)
        rang2 = rang * ration_hist
        minx = math.ceil((-rang2+x)/delta)
        maxx = math.floor(1 + (rang2+x)/delta)
        miny = math.ceil((-rang2+y)/delta)
        maxy = math.floor(1 + (rang2+y)/delta)

        if minx <= 0 or maxx >= wid - 1 or miny <= 0 or maxy >= hei - 1:
            return None 

        histograms = [[[0]*n_ori]*n_hist]*n_hist

        for m in range(minx, maxx):
            for n in range(miny, maxy):
                nx, ny = calc_normalized_cord(m, n)
                #Verify if the sample (m, n) is inside the normalized patc
                if max(abs(nx), abs(ny)) >= lam_descr * ration_hist:
                    continue
                
                #Compute normalized gradient orientation
                grad = self._gradient_m_n(dog[dog_oct], s, m, n)
                ntheta = (np.arctan2(grad[0], grad[1]) - theta) % (2*math.pi)

                #compute contribution
                exponent = -((math.pow(delta * m - x,2) + math.pow(delta * n - y,2)))/(2*((lam_descr*gamma)**2))
                grad_norm = math.sqrt(grad[0]**2 + grad[1]**2)
                contribution = math.exp(exponent) * grad_norm
                for i in range(1, n_hist +1):
                    for j in range(1, n_hist + 1):
                        xi = (i - (1+n_hist)/2) * (2*lam_descr/n_hist)
                        yj = (j - (1+n_hist)/2) * (2*lam_descr/n_hist)
                        if abs(xi - nx) > (2*lam_descr/n_hist) or abs(yj - ny) > (2*lam_descr/n_hist):
                            continue

                        for k in range(1, n_ori + 1):
                            thetak = 2*math.pi*(k-1)/n_ori
                            if abs((thetak - ntheta)%(2*math.pi)) >= 2*math.pi/n_ori:
                                continue
                            histograms[i-1][j-1][k-1] = (histograms[i-1][j-1][k-1]
                                + (1 - ((n_hist)/(2*lam_descr))*abs(xi-nx))
                                * (1 - ((n_hist)/(2*lam_descr))*abs(yj-ny))
                                * (1 - ((n_ori)/(2*math.pi))*abs((thetak - ntheta)%2*math.pi))
                                * contribution) 

        feature_vector = [0]*(n_hist*n_hist*n_ori)
        for i in range(n_hist):
            for j in range(n_hist):
                for k in range(n_ori):
                    feature_vector[i*n_hist*n_ori+j*n_ori+k] = histograms[i][j][k]

        feature_vector = np.array(feature_vector)
        fnorm = np.linalg.norm(feature_vector)
        norm_thresh = 0.2 * fnorm        

        feature_vector[feature_vector > norm_thresh] = norm_thresh
        feature_vector = feature_vector*512/fnorm
        feature_vector[feature_vector > 255] = 255

        return feature_vector
                

    def _describe_keypoints(self, dog, minimas, maximas, lam, n_bins, threshold, lam_descr, n_hist, n_ori):
        keypoints = minimas + maximas
        final_features = []
        for kp in keypoints:
            kpthetas = self._calculate_reference(dog, kp, lam, n_bins, threshold)
            if kpthetas is None:
                continue
            for kpt in kpthetas:
                if kpt is None:
                    continue
                fv = self._normalized_descriptor(dog, kpt, lam_descr, n_hist, n_ori)
                if fv is not None:
                    final_features.append((kpt, fv))

        return final_features
                
            

    def _get_minimas_maximas(self, dog_octaves, c_edge, gaussian):
        maximas = []
        minimas = []
        #important to know: the downscale must be /2, if I change that, must be changed here
        for i in range(len(dog_octaves)):
            doc_oct = i
            current_dog_octave = dog_octaves[i]
            #just the middle ones
            kk = 0

            for ts in range(1, len(current_dog_octave) - 1):
                current_img = current_dog_octave[ts]
                tmp_img = (current_img.copy()*255).astype('uint8')
                w, h = current_img.shape
                for x, y in itertools.product(range(1, w - 1), range(1, h - 1)):
                    s = ts
                    kk +=1
                    combs = [(x - a, y - b, s - c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (0, 0, 0)]
                    combs.remove((x, y, s))
                    has_min_max = True
                    is_minima = True
                    is_maxima = True

                    for c in combs:                        
                        nx, ny, nj = c
                        if current_img[x, y] > current_dog_octave[nj][nx, ny] and is_minima == True:
                            is_minima = False

                        if current_img[x, y] < current_dog_octave[nj][nx, ny] and is_maxima == True:
                            is_maxima = False

                        if is_maxima == False and is_minima == False:
                            has_min_max = False
                            break          

                    if has_min_max == False:
                        continue  
                    
                    subpixel = self._get_subpixel_keypoint(current_dog_octave, s, x, y, doc_oct, gaussian, len(dog_octaves))
                    if subpixel is None:
                        continue

                    kpval, s, m, n, rx, ry, gamma = subpixel
                    tmp_img = cv2.circle(tmp_img,(n,m), 1, (255,255,255), 1)

                    #discard values lower that 0.03 as the paper says
                    if abs(kpval) <= 0.015 or self._does_discard_keypoint(current_dog_octave, s, m, n, c_edge):
                        continue

                    if is_maxima:
                        maximas.append((kpval, s, m, n, doc_oct, rx, ry, gamma))
                    if is_minima:
                        minimas.append((kpval, s, m, n, doc_oct, rx, ry, gamma))

        return (minimas, maximas)

    def get_descriptors(self, img_color, n_octaves=8, n_levels_octave=4, kernel_size=7,
            gaussian=0.8, c_edge=20, lam=1.5, n_bins=36, threshold=0.8, lam_descr = 6, n_hist = 3, n_ori = 8, save = True):
        #img must be black and white        
        img_u = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY).astype(float)/255      
        w, h = img_u.shape 
        n_octaves = min(math.floor(np.log2(min(w, h)/(0.5/12)) + 1), n_octaves)
        img = cv2.resize(img_u, dsize=(h*2, w*2), interpolation= cv2.INTER_LINEAR)        
        g_octaves = self._get_gaussian_pyramid(img, n_octaves, n_levels_octave, kernel_size, gaussian)
        dog_octaves = self._get_dog_pyramid(g_octaves)
        (minimas, maximas) = self._get_minimas_maximas(dog_octaves, c_edge, gaussian)
    
        kpt_fv = self._describe_keypoints(dog_octaves, minimas, maximas, lam, n_bins, threshold, lam_descr, n_hist, n_ori)

        img_temp = img_color.copy()
        for kpt, fv in kpt_fv:
            kpval, s, mt, nt, doc_oct, rx, ry, gamma, theta = kpt
            img_temp = cv2.circle(img_temp,(int(round(ry)), int(round(rx))), 2, (0,255,0), 1)

        write_image(1, img_temp, save)

        kpt_fv_r = [((int(round(k[0][6])), int(round(k[0][5])), k[0][6], k[0][5], k[0][8]), k[1]) for k in kpt_fv]

        return kpt_fv_r

    def __init__(self):
        pass
