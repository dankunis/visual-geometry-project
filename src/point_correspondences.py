from utils import *

import cv2
import numpy as np
from distutils.version import LooseVersion
from matplotlib import pyplot as plt
import fnmatch

def feature_matching(input_frames, output_folder):
    print("[FEATURE MATCHING] : start matching. This will take some time...")
    pics = fnmatch.filter(os.listdir(input_frames), '*.jpg')
    amount_pics = len(pics)
    MIN_MATCH_COUNT = 10
    #compare current image with successorimage
    for i in tqdm(range(amount_pics - 1)):
        img1 = resizeImg(cv2.imread(os.path.join(input_frames, pics[i]), 0)) # queryImage
        img2 = resizeImg(cv2.imread(os.path.join(input_frames, pics[i+1]), 0)) # trainImage

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)

            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

        else:
            print ("[FEATURE MATCHING] : Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
            matchesMask = None
        
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

        plt.imshow(img3, 'gray'),plt.savefig(output_folder + "img%s" % (i))

    print("[FEATURE MATCHING] : finished matching. Output in " + output_folder)    
        


#SIFT
def get_SIFT_key_points(input_frames, output_folder):
    if LooseVersion(cv2.__version__) > LooseVersion('3.4.2.16'):
        print("[ERROR] : Your opencv version must be 3.4.2.16 for using SIFT. Please try 'pip install opencv-python==3.4.2.16' and 'pip install opencv-contrib-python==3.4.2.16'. Your current opencv version is " + cv2.__version__)
        return
    
    print("[POINT CORRESPONDENCES] : SIFT - get key points")
    for counter in tqdm(range(len(os.listdir(input_frames)) - 1)):
        name = os.path.join(input_frames, 'IMG_' + str(counter + 6363) + '.jpg')
        img = cv2.imread(name)

        img = resizeImg(img)

        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create()
        #keypoints + descriptor
        kp, des = sift.detectAndCompute(gray,None)

        img=cv2.drawKeypoints(gray,kp, None)

        cv2.imwrite(os.path.join(output_folder + "{:05d}.png".format(counter)), img) #save in output folder
        #cv2.imshow(name,img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
    print("[POINT CORRESPONDENCES] : SIFT - output in " + output_folder)

#Harris Corner Detector (HCD)
def get_key_points(input_frames, output_folder):

    print("[POINT CORRESPONDENCES] : Harris Corner Detection - get key points")
    for counter in tqdm(range(len(os.listdir(input_frames)) - 1)):
        #os.path.join(input_frames, 'IMG_' + str(counter + 6363) + '.jpg') #for "frames" pictures
        #os.path.join(input_frames, str(counter) + '.png') #for chessboard pictures
        #change function parameter as well
        name = os.path.join(input_frames, 'IMG_' + str(counter + 6363) + '.jpg')
        img = cv2.imread(name)

        img = resizeImg(img)

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
        #cv2.imshow(name,img)
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

def resizeImg(img):
    # resize the image so it would be no bigger than 1920x1080
    height, width = img.shape[:2]
    if max(width, height) > 2000:
        if height > width:
            new_width = 1080
        else:
            new_width = 1920

        return image_resize(img, width=new_width)