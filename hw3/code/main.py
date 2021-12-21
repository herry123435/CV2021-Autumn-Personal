import math
import numpy as np
from PIL import Image

def compute_h(p1, p2):
    A = np.zeros((p1.shape[0]*2, 9), dtype = np.float32)
    for i in range(0,2*p1.shape[0]):
        if (i%2 == 0):
            A[i][0] = p2[int(i/2)][0]
            A[i][1] = p2[int(i/2)][1]
            A[i][2] = 1
            A[i][6] = -p1[int(i/2)][0] * p2[int(i/2)][0]
            A[i][7] = -p1[int(i/2)][0] * p2[int(i/2)][1]
            A[i][8] = -p1[int(i/2)][0]
        else:
            A[i][3] = p2[int((i-1)/2)][0]
            A[i][4] = p2[int((i-1)/2)][1]
            A[i][5] = 1
            A[i][6] = -p1[int((i-1)/2)][1] * p2[int((i-1)/2)][0]
            A[i][7] = -p1[int((i-1)/2)][1] * p2[int((i-1)/2)][1]
            A[i][8] = -p1[int((i-1)/2)][1]
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    V = Vt.transpose(1,0)
    H = np.zeros((3,3), dtype = np.float32)
    for i in range(0,3):
        for j in range(0,3):
            H[i][j] = V[3*i + j][8]

    return H

def compute_h_norm(p1, p2):
    p1n = np.zeros(p1.shape, dtype = np.float32)
    p2n = np.zeros(p2.shape, dtype = np.float32)
    p1n[:,0] = p1[:,0] / 1600
    p1n[:,1] = p1[:,1] / 1200
    p2n[:,0] = p2[:,0] / 1600
    p2n[:,1] = p2[:,1] / 1200

    H = compute_h(p1n, p2n)

    H[0][1] = (1600/1200) * H[0][1]
    H[0][2] = 1600 * H[0][2]
    H[1][0] = (1200/1600) * H[1][0]
    H[1][2] = 1200 * H[1][2]
    H[2][0] = H[2][0] / 1600
    H[2][1] = H[2][1] / 1200

    return H

def warp_image(igs_in, igs_ref, H):
    igs_warp = np.zeros((1200, 1600, 3), dtype = np.float32)
    wp = np.ones((3, 1600*1200), dtype = np.float32)
    ip = np.ones((3, 1600*1200), dtype = np.float32)

    for y in range(0, 1200):
        for x in range(0, 1600):
            wp[0][1600*y + x] = x
            wp[1][1600*y + x] = y

    ip = np.matmul(np.linalg.inv(H), wp)

    ip[0] = ip[0] / ip[2]
    ip[1] = ip[1] / ip[2]

    for y in range(0, 1200):
        for x in range(0, 1600):
            if(ip[0][1600*y + x] < 0 or ip[1][1600*y + x] < 0 or ip[0][1600*y + x] >= 1599 or ip[1][1600*y + x] >= 1199):
                continue

            i = int(ip[0][1600*y + x])
            j = int(ip[1][1600*y + x])
            a = ip[0][1600*y + x] - i
            b = ip[1][1600*y + x] - j

            igs_warp[y][x] = (1-a)*(1-b)*igs_in[j][i] + a*(1-b)*igs_in[j][i+1] + a*b*igs_in[j+1][i+1] + (1-a)*b*igs_in[j+1][i]

    igs_merge = np.zeros((2200, 3200, 3), dtype = np.float32)
    wp2 = np.ones((3, 2200*3200), dtype = np.float32)
    ip2 = np.ones((3, 2200*3200), dtype = np.float32)

    igs_merge[500:1700,1600:3200,:] = igs_ref

    for y in range(0, 2200):
        for x in range(0, 3200):
            wp2[0][3200*y + x] = x - 1600
            wp2[1][3200*y + x] = y - 500

    ip2 = np.matmul(np.linalg.inv(H), wp2)

    ip2[0] = ip2[0] / ip2[2]
    ip2[1] = ip2[1] / ip2[2]

    for y in range(0, 2200):
        for x in range(0, 3200):
            if(ip2[0][3200*y + x] < 0 or ip2[1][3200*y + x] < 0 or ip2[0][3200*y + x] >= 1599 or ip2[1][3200*y + x] >= 1199):
                continue

            i = int(ip2[0][3200*y + x])
            j = int(ip2[1][3200*y + x])
            a = ip2[0][3200*y + x] - i
            b = ip2[1][3200*y + x] - j

            igs_merge[y][x] = (1-a)*(1-b)*igs_in[j][i] + a*(1-b)*igs_in[j][i+1] + a*b*igs_in[j+1][i+1] + (1-a)*b*igs_in[j+1][i]

    return igs_warp, igs_merge

def rectify(igs, p1, p2):
    H = compute_h(p1, p2)
    igs_rec = np.zeros(igs.shape, dtype = np.float32)

    wp = np.ones((3, igs.shape[1]*igs.shape[0]), dtype = np.float32)
    ip = np.ones((3, igs.shape[1]*igs.shape[0]), dtype = np.float32)

    for y in range(0, igs.shape[0]):
        for x in range(0, igs.shape[1]):
            wp[0][igs.shape[1]*y + x] = x
            wp[1][igs.shape[1]*y + x] = y

    ip = np.matmul(np.linalg.inv(H), wp)

    ip[0] = ip[0] / ip[2]
    ip[1] = ip[1] / ip[2]

    for y in range(0, igs.shape[0]):
        for x in range(0, igs.shape[1]):
            if(ip[0][igs.shape[1]*y + x] < 0 or ip[1][igs.shape[1]*y + x] < 0 or ip[0][igs.shape[1]*y + x] >= igs.shape[1] - 1 or ip[1][igs.shape[1]*y + x] >= igs.shape[0] - 1):
                continue

            i = int(ip[0][igs.shape[1]*y + x])
            j = int(ip[1][igs.shape[1]*y + x])
            a = ip[0][igs.shape[1]*y + x] - i
            b = ip[1][igs.shape[1]*y + x] - j

            igs_rec[y][x] = (1-a)*(1-b)*igs[j][i] + a*(1-b)*igs[j][i+1] + a*b*igs[j+1][i+1] + (1-a)*b*igs[j+1][i]

    return igs_rec

def set_cor_mosaic():
    p_in = np.array([[985,445], [1293,541], [1543,723], [891,867], [1255,959]])
    p_ref = np.array([[229,433], [545,541], [755,709], [117,895], [509,947]])

    return p_in, p_ref

def set_cor_rec():
    c_in = np.array([[1399,161],[1399,835],[1090,161],[1090,835]])
    c_ref = np.array([[1399,161],[1399,835],[1059,195],[1049,835]])

    return c_in, c_ref

def main():
    ##############
    # step 1: mosaicing
    ##############
    
    # read images
    img_in = Image.open('data/porto1.png').convert('RGB')
    img_ref = Image.open('data/porto2.png').convert('RGB')

    # shape of igs_in, igs_ref: [y, x, 3]
    igs_in = np.array(img_in)
    igs_ref = np.array(img_ref)

    # lists of the corresponding points (x,y)
    # shape of p_in, p_ref: [N, 2]
    p_in, p_ref = set_cor_mosaic()

    # p_ref = H * p_in
    H = compute_h_norm(p_ref, p_in)
    igs_warp, igs_merge = warp_image(igs_in, igs_ref, H)

    # plot images
    img_warp = Image.fromarray(igs_warp.astype(np.uint8))
    img_merge = Image.fromarray(igs_merge.astype(np.uint8))

    # save images
    img_warp.save('porto1_warped.png')
    img_merge.save('porto_mergeed.png')
    
    ##############
    # step 2: rectification
    ##############

    img_rec = Image.open('data/iphone.png').convert('RGB')
    igs_rec = np.array(img_rec)

    c_in, c_ref = set_cor_rec()

    igs_rec = rectify(igs_rec, c_in, c_ref)

    img_rec = Image.fromarray(igs_rec.astype(np.uint8))
    img_rec.save('iphone_rectified.png')

if __name__ == '__main__':
    main()
