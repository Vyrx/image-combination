from cmath import inf
import cv2
from matplotlib.pyplot import pause
import numpy as np
import random
import math
import sys

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # the example of image window
    # creat_im_window("Result",img)
    # im_show()

    # you can use this function to store the result
    # cv2.imwrite("result.jpg",img)

    # img=cv2.drawKeypoints(img_gray[0],kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('sift_keypoints.jpg',img)

    # img_1 = base image
    img_1, img_gray_1 = read_img('./test/m1.jpg')
    img_1 = np.array(img_1) # 756 x 1008 x 3
    img_gray_1 = np.array(img_gray_1) # 756 x 1008

    img_2, img_gray_2 = read_img('./test/m2.jpg')
    img_2 = np.array(img_2) # 756 x 1008 x 3
    img_gray_2 = np.array(img_gray_2) # 756 x 1008

    SIFT_Detector = cv2.SIFT_create()

    kp_1, des_1 = SIFT_Detector.detectAndCompute(img_gray_1, None)
    kp_2, des_2 = SIFT_Detector.detectAndCompute(img_gray_2, None)

    # kp : number of keypoints
    # des : number of keypoints x 128

    ### Feature Matching
    
    matching_kp = [] # each entry is a matching kp [image_1_index, image_2_index] between both iamges

    for j in range(len(kp_1)):

        min_diff = [inf, inf] # min norm diff
        min_index = [-1, -1] # index of min norm diff

        for k in range(len(kp_2)):
            for l in range(2):
                diff = np.abs(np.linalg.norm(des_1[j] - des_2[k]))
                if diff < min_diff[l]:
                    min_diff[l] = diff
                    min_index[l] = k
                    break
        
        ### Lowe's Ratio Test
        if min_diff[0] < 0.7 * min_diff[1]:
            matching_kp.append([j, min_index[0]])
        
    
    np.save('matching_kp_1.npy', matching_kp)


    p1 = []
    p2 = []

    # Each point (x,y) in p1 of image_1 corresponds to point p2 of image_2 
    
    for i in range(len(matching_kp)):
        p1.append([kp_1[matching_kp[i][0]].pt[0], kp_1[matching_kp[i][0]].pt[1]])
        p2.append([kp_2[matching_kp[i][1]].pt[0], kp_2[matching_kp[i][1]].pt[1]])
    
    np.save('p1.npy', p1)
    np.save('p2.npy', p2)

    p1 = np.load('p1.npy')
    p2 = np.load('p2.npy')

    ### RANSAC

    max_support = 0

    while True:
        # Select 4 random points

        inlier_index = random.sample(range(len(p1)), 4)

        A = []
        b = []
        
        for i in range(4):
            x1 = p1[inlier_index[i]][0]
            y1 = p1[inlier_index[i]][1]
            x2 = p2[inlier_index[i]][0]
            A.append([x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2])
            b.append([x2])
        
        for i in range(4):
            x1 = p1[inlier_index[i]][0]
            y1 = p1[inlier_index[i]][1]
            y2 = p2[inlier_index[i]][1]
            A.append([0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2])
            b.append([y2])
        A = np.array(A)
        b = np.array(b)

        H = np.matmul( np.matmul( np.linalg.inv( np.matmul(A.T, A) ), A.T), b)
        # u, s, vh = np.linalg.svd(A)
        # H = vh[7]

        H = np.append(H, 1)
        H.resize(3,3)

        #Test every p2' against p2

        num_support = 0

        for i in range(len(p1)):
            x_p1 = p1[i][0]
            y_p1 = p1[i][1]
            x_p2 = p2[i][0]
            y_p2 = p2[i][1]

            p_temp = np.matmul(H, [[x_p1],[y_p1],[1]]) # projection result
            x_temp = p_temp[0] / p_temp[2]
            y_temp = p_temp[1] / p_temp[2]
            
            if i not in inlier_index:
                dist = math.dist([x_p2,y_p2],[x_temp,y_temp])
                if dist < 1:
                    num_support += 1

        if num_support > 250:
            break
        
            

    x_corner = [0, 0, np.shape(img_1)[0], np.shape(img_1)[0]]
    y_corner = [0, np.shape(img_1)[1], 0, np.shape(img_1)[1]]
    
    # Transform the corners

    for i in range(2):
        x_p1 = x_corner[i]
        y_p1 = y_corner[i]
        p_temp = np.matmul(H, [[x_p1],[y_p1],[1]]) # projection result
        x_corner[i] = (p_temp[0] / p_temp[2])[0]
        y_corner[i] = (p_temp[1] / p_temp[2])[0]
    
    x1_prime = min(int(min(x_corner)),0)
    y1_prime = min(int(min(y_corner)),0)

    size = [np.shape(img_2)[1] + np.abs(x1_prime), np.shape(img_2)[0] + np.abs(y1_prime)]

    A = np.array([[1.0, 0, -x1_prime],[0, 1.0, -y1_prime],[0, 0, 1.0]],dtype=np.float64)
    H = np.matmul(A, H)
    warped1 = cv2.warpPerspective(src=img_1,M=H,dsize=size)
    warped2 = cv2.warpPerspective(src=img_2,M=A,dsize=size)

    # creat_im_window("img_1",warped1)
    # creat_im_window("img_2",warped2)

    alpha = 0.5
    beta = 0.5
    dst = cv2.addWeighted(warped1, alpha, warped2, beta, 0.0)
    creat_im_window("Image",dst)
    cv2.imwrite("Blend_1.jpg",dst)
    im_show()




