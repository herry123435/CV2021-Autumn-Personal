
import math
import glob
import numpy as np
from PIL import Image
import imageio

# parameters

datadir = './data'
resultdir='./results'

sigma=4
highThreshold=0.07
lowThreshold=0.03
rhoRes=2
thetaRes=math.pi/180
nLines=20


def ConvFilter(Igs, G):
    hp = int(G.shape[0]/2)
    wp = int(G.shape[1]/2)
    padded = np.empty((Igs.shape[0]+2*hp, Igs.shape[1]+2*wp), object)

    padded[hp:Igs.shape[0]+hp, wp:Igs.shape[1]+wp] = Igs[:,:]
    for i in range(0, hp):
        for j in range(0,wp):
            padded[i,j] = Igs[0,0]
        for j in range(wp, Igs.shape[1] + wp):
            padded[i,j] = Igs[0,j-wp]
        for j in range(Igs.shape[1] + wp, padded.shape[1]):
            padded[i,j] = Igs[0, Igs.shape[1] - 1]
    for i in range(hp, Igs.shape[0] + hp):
        for j in range(0,wp):
            padded[i,j] = Igs[i-hp, 0]
        for j in range(Igs.shape[1]+wp, padded.shape[1]):
            padded[i,j] = Igs[i-hp, 0]
    for i in range(Igs.shape[0]+hp, padded.shape[0]):
        for j in range(0,wp):
            padded[i,j] = Igs[Igs.shape[0]-1, 0]
        for j in range(wp,Igs.shape[1]+wp):
            padded[i,j] = Igs[Igs.shape[0]-1, j-wp]
        for j in range(Igs.shape[1]+wp, padded.shape[1]):
            padded[i,j] = Igs[Igs.shape[0]-1, Igs.shape[1]-1]

    Iconv = np.empty(Igs.shape, object)
    temp = []
    h = padded.shape[0] - G.shape[0] + 1
    w = padded.shape[1] - G.shape[1] + 1

    for i in range(h):
        for j in range(w):
            conv = padded[i:i+G.shape[0], j:j+G.shape[1]] * G
            temp.append(np.sum(conv))
    
    Iconv[:,:] = np.array(temp).reshape(Igs.shape[0], Igs.shape[1])

    return Iconv

def EdgeDetection(Igs, sigma, highThreshold, lowThreshold):
    size = 3*sigma
    h = int((size-1)/2)
    w = int((size-1)/2)
    y,x = np.ogrid[-h:h+1, -w:w+1]
    z = np.exp(-( (x*x)/(2.*sigma*sigma) + (y*y)/(2.*sigma*sigma) ))
    gk = z / np.sum(z)
    smoothed = ConvFilter(Igs, gk)

    sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Ix = ConvFilter(smoothed, sx)

    sy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    Iy = ConvFilter(smoothed, sy)

    Imag = np.sqrt(Ix.astype(float) * Ix.astype(float) + Iy.astype(float) *Iy.astype(float))
    Io = np.arctan2(Iy.astype(float), Ix.astype(float))

    angle = Io * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(0,Imag.shape[0]):
        for j in range(0,Imag.shape[1]):
            try:
                p = 255
                r = 255

                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    p = Imag[i,j+1]
                    r = Imag[i,j-1]
                elif (22.5 <= angle[i,j] < 67.5):
                    p = Imag[i+1,j-1]
                    r = Imag[i-1,j+1]
                elif (67.5 <= angle[i,j] < 112.5):
                    p = Imag[i+1,j]
                    r = Imag[i-1,j]
                elif (112.5 <= angle[i,j] < 157.5):
                    p = Imag[i-1,j-1]
                    r = Imag[i+1,j+1]

                if(Imag[i,j] >= p) and (Imag[i,j] >= r):
                    Imag[i,j] = Imag[i,j]
                else:
                    Imag[i,j] = 0

            except IndexError as e:
                pass
            
    Im = np.empty(Imag.shape, object)

    for i in range(0,Imag.shape[0]):
        for j in range(0,Imag.shape[1]):
            if (Imag[i,j] >= highThreshold):
                Im[i,j] = 1
            else:
                Im[i,j] = 0

    for i in range(0,Imag.shape[0]):
        for j in range(0,Imag.shape[1]):
            if (Imag[i,j] >= lowThreshold and Im[i,j] == 0):
                try:
                    if (Im[i-1,j-1] == 1) or (Im[i-1,j] == 1) or (Im[i-1,j+1] == 1) or (Im[i,j-1] == 1) or (Im[i,j+1] == 1) or (Im[i+1,j-1] == 1) or (Im[i+1,j] == 1) or (Im[i+1,j+1] == 1):
                        Im[i,j] = 1;
                except IndexError as e:
                    pass

    return Im, Io, Ix, Iy

def HoughTransform(Im, rhoRes, thetaRes):
    rho_max = np.sqrt(Im.shape[0]*Im.shape[0] + Im.shape[1]*Im.shape[1])
    rho_dim = (int)(rho_max / rhoRes)
    theta_dim = (int)(2*np.pi / thetaRes)

    H = np.zeros((rho_dim, theta_dim))

    for i in range(0, Im.shape[0]):
        for j in range(0, Im.shape[1]):
            if(Im[i,j] == 1):
                for t in range(0, theta_dim):
                    theta = (2*np.pi) * (1.0 * t / theta_dim)
                    rho = i*np.cos(theta) + j*np.sin(theta)
                    r = (int)(rho_dim * (1.0 * rho / rho_max))
                    H[r,t] += 1
    
    return H

def HoughLines(H,rhoRes,thetaRes,nLines):
    for i in range(0, H.shape[0]):
        for j in range(0, H.shape[1]):
            try:
                if(H[i-1,j-1] >= H[i,j]) or (H[i-1,j] >= H[i,j]) or (H[i-1,j+1] >= H[i,j]) or (H[i,j-1] >= H[i,j]) or (H[i,j+1] >= H[i,j]) or (H[i+1,j-1] >= H[i,j]) or (H[i+1,j] >= H[i,j]) or (H[i+1,j+1] >= H[i,j]):
                    H[i,j] = 0
            except IndexError as e:
                pass

    lRho = np.array([])
    lTheta = np.array([])

    for k in range(0, nLines):
        max = -3000
        li = -1
        lj = -1
        for i in range(0, H.shape[0]):
            for j in range(0, H.shape[1]):
                if H[i,j] > max:
                    max = H[i,j]
                    li = i
                    lj = j
        lRho = np.append(lRho, li * rhoRes)
        lTheta = np.append(lTheta, lj * thetaRes)
        H[li,lj] = 0

    return lRho,lTheta

def HoughLineSegments(lRho, lTheta, Im):
    l = []
    for k in range(0, lRho.shape[0]):
        coor = []
        for i in range(0, Im.shape[0]):
                for j in range(0, Im.shape[1]):
                    if (-1 <= lRho[k] - (i*np.cos(lTheta[k]) + j*np.sin(lTheta[k])) <= 1):
                        for ri in range(-1, 2):
                            for rj in range(-1,2):
                                try:
                                    if(Im[i+ri,j+rj] == 1):
                                        coor.append([i,j])
                                except IndexError as e:
                                    pass
        coor.sort(key=lambda x:x[0])
        try:
            sx = coor[0][0]
            sy = coor[0][1]
            ex = coor[len(coor)-1][0]
            ey = coor[len(coor)-1][0]
            l.append({"start":(sx,sy), "end":(ex,ey)})
        except IndexError as e:
            pass
    
    l = np.array(l)

    return l

def main():
    number = 1
    # read images
    for img_path in glob.glob(datadir+'/*.jpg'):
        # load grayscale image
        img = Image.open(img_path).convert("L")

        Igs = np.array(img)
        Igs = Igs / 255.

        # Hough function
        Im, Io, Ix, Iy = EdgeDetection(Igs, sigma, highThreshold, lowThreshold)
        H= HoughTransform(Im, rhoRes, thetaRes)
        lRho,lTheta =HoughLines(H,rhoRes,thetaRes,nLines)
        l = HoughLineSegments(lRho, lTheta, Im)

        # saves the outputs to files
        # Im, H, Im + hough line , Im + hough line segments
        imageio.imwrite(("Im_" + str(number) + ".jpeg"), Im.astype(float))
        imageio.imwrite(("H_" + str(number) + ".jpeg"), H.astype(float))
        original = np.asarray(Image.open(img_path).convert('RGB'))
        for k in range(0,nLines):
            for i in range(0, original.shape[0]):
                for j in range(0, original.shape[1]):
                    if (-1 <= lRho[k] - (i*np.cos(lTheta[k]) + j*np.sin(lTheta[k])) <= 1):
                        original[i,j,0] = 0
                        original[i,j,1] = 0
                        original[i,j,2] = 0
        imageio.imwrite(("hl_" + str(number) + ".jpeg"), original.astype(float))
        original = np.asarray(Image.open(img_path).convert('RGB'))
        for k in range(0,nLines):
            for i in range(0, original.shape[0]):
                for j in range(0, original.shape[1]):
                    if (-1 <= lRho[k] - (i*np.cos(lTheta[k]) + j*np.sin(lTheta[k])) <= 1):
                        try:
                            if(l[k]['start'][0] <= i <= l[k]['end'][0]):
                                original[i,j,0] = 0
                                original[i,j,1] = 0
                                original[i,j,2] = 0
                        except IndexError as e:
                            pass
        imageio.imwrite(("hls_" + str(number) + ".jpeg"), original.astype(float))
        number += 1


if __name__ == '__main__':
    main()