import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold

def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    dp = np.zeros((6,1), dtype = np.float64)
    
    H = np.zeros((6,6))
    A = np.array([[1+p[0][0], p[2][0], p[4][0]],[p[1][0],1+p[3][0],p[5][0]]])
    
    sy = np.arange(0, img2.shape[0], 1)
    sx = np.arange(0, img2.shape[1], 1)
    spl = RectBivariateSpline(sy, sx, img2)
    
    for x in range(img1.shape[1]):
        for y in range(img1.shape[0]):
            hc = np.array([[x],[y],[1]]) #homogeneous coordinate
            wx = (A@hc)[0][0] #warped x
            wy = (A@hc)[1][0] #warped y
            if (wx < 0 or wx >= img2.shape[1] or wy < 0 or wy >= img2.shape[0]):
                continue
            gradI = np.array([[Gx[y][x], Gy[y][x]]])
            jacob = np.array([[x,0,y,0,1,0],[0,x,0,y,0,1]])
            H += (np.transpose(gradI@jacob))@(gradI@jacob)
            dp += (np.transpose(gradI@jacob))@(np.array([[ img1[y][x] - spl.ev(wy, wx) ]]))
            
    dp = (np.linalg.inv(H))@dp

    return dp

p = np.zeros((6,1), dtype = np.float64)
def subtract_dominant_motion(img1, img2):
    Gx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize = 5)
    Gy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize = 5)
    
    global p
    p += lucas_kanade_affine(img1, img2, p, Gx, Gy)
    M = np.array([[1+p[0][0], p[2][0], p[4][0]],[p[1][0],1+p[3][0],p[5][0]],[0,0,1]])
    
    igs_warp = np.zeros((img1.shape[0], img1.shape[1]), dtype = np.float32)
    wp = np.ones((3, img1.shape[1]*img1.shape[0]), dtype = np.float32)
    ip = np.ones((3, img1.shape[1]*img1.shape[0]), dtype = np.float32)

    for y in range(img1.shape[0]):
        for x in range(img1.shape[1]):
            wp[0][img1.shape[1]*y + x] = x
            wp[1][img1.shape[1]*y + x] = y

    ip = np.matmul(np.linalg.inv(M), wp)

    ip[0] = ip[0] / ip[2]
    ip[1] = ip[1] / ip[2]

    for y in range(0, img1.shape[0]):
        for x in range(img1.shape[1]):
            if(ip[0][img1.shape[1]*y + x] < 0 or ip[1][img1.shape[1]*y + x] < 0 or ip[0][img1.shape[1]*y + x] >= (img1.shape[1]-1) or ip[1][img1.shape[1]*y + x] >= (img1.shape[0]-1)):
                continue

            i = int(ip[0][img1.shape[1]*y + x])
            j = int(ip[1][img1.shape[1]*y + x])
            a = ip[0][img1.shape[1]*y + x] - i
            b = ip[1][img1.shape[1]*y + x] - j

            igs_warp[y][x] = (1-a)*(1-b)*img1[j][i] + a*(1-b)*img1[j][i+1] + a*b*img1[j+1][i+1] + (1-a)*b*img1[j+1][i]

    moving_image = np.abs(img2 - igs_warp)
    
    th_hi = 0.24 * 256
    th_lo = 0.18 * 256

    hyst = apply_hysteresis_threshold(moving_image, th_lo, th_hi)
    return hyst

if __name__ == "__main__":
    data_dir = 'data'
    video_path = 'motion.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 150/20, (636, 318))
    tmp_path = os.path.join(data_dir, "organized-{}.jpg".format(0))
    T = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)
    for i in range(0, 50):
        img_path = os.path.join(data_dir, "organized-{}.jpg".format(i))
        I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        clone = I.copy()
        moving_img = subtract_dominant_motion(T, I)
        clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
        clone[moving_img, 2] = 522
        out.write(clone)
        T = I
    out.release()
    