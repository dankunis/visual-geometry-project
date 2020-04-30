from utils import *

import cv2
import numpy as np


#SIFT
def get_SIFT_key_points(input_frames, output_folder):
    if cv2.__version__ != '3.4.2.16':
        print("[ERROR] : Your opencv version must be 3.4.2.16 for using SIFT. Please try 'pip install opencv-python==3.4.2.16' and 'pip install opencv-contrib-python==3.4.2.16'")
        return
    
    print("[POINT CORRESPONDENCES] : SIFT - get key points")
    for counter in range(len(os.listdir(input_frames)) - 1):
        name = os.path.join(input_frames, 'IMG_' + str(counter + 6363) + '.jpg')
        img = cv2.imread(name)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        #keypoints + descriptor
        kp, des = sift.detectAndCompute(gray,None)

        img=cv2.drawKeypoints(gray,kp)

        cv2.imwrite(os.path.join(output_folder + "{:05d}.png".format(counter)), img) #save in output folder
        cv2.imshow(name,img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

#Harris Corner Detector (HCD)
def get_key_points(input_frames, output_folder):

    print("[POINT CORRESPONDENCES] : Harris Corner Detection - get key points")
    for counter in range(len(os.listdir(input_frames)) - 1):
        #os.path.join(input_frames, 'IMG_' + str(counter + 6363) + '.jpg') #for "frames" pictures
        #os.path.join(input_frames, str(counter) + '.png') #for chessboard pictures
        #change function parameter as well
        name = os.path.join(input_frames, 'IMG_' + str(counter + 6363) + '.jpg')
        img = cv2.imread(name)

        # resize the image so it would be no bigger than 1920x1080
        height, width = img.shape[:2]
        if max(width, height) > 2000:
            if height > width:
                new_width = 1080
            else:
                new_width = 1920

            img = image_resize(img, width=new_width)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray) #float32 type needed for HCD
        #gray = input image
        #blockSize = It is the size of neighbourhood considered for corner detection
        #ksize = Aperture parameter of Sobel derivative used.
        #k = Harris detector free parameter in the equation.
        #values taken from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
        dst = cv2.cornerHarris(gray,2,3,0.04)

        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)

        # Threshold for an optimal value, it may vary depending on the image.
        img[dst>0.01*dst.max()]=[0,0,255]

        cv2.imwrite(os.path.join(output_folder + "{:05d}.png".format(counter)), img) #save in output folder
        cv2.imshow(name,img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()





        #optional: more accuracy with SubPixel:

        # ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        # dst = np.uint8(dst)
        # # find centroids
        # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        # # define the criteria to stop and refine the corners
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        # corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
        # # Now draw them
        # res = np.hstack((centroids,corners))
        # res = np.int0(res)
        # img[res[:,1],res[:,0]]=[0,0,255]
        # img[res[:,3],res[:,2]] = [0,255,0]
        # #show
        # cv2.imshow(name,img)
        # cv2.waitKey(500)
        # cv2.destroyAllWindows()
