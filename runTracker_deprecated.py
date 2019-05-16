import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import glob
import re
import os
import sys
import time
#import ipdb

import net


#gpu = 0
#os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

def main():

    def extBBvot(annParse):
        bbx1 = min(annParse[0],annParse[2],annParse[4],annParse[6])
        bby1 = min(annParse[1],annParse[3],annParse[5],annParse[7])
        bbx2 = max(annParse[0],annParse[2],annParse[4],annParse[6])
        bby2 = max(annParse[1],annParse[3],annParse[5],annParse[7])
        return [bbx1, bby1, bbx2, bby2]

    def bb_intersection_over_union(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def view(img, vidOut):
        vidOut.write(img)
        plt.imshow(img[:,:,::-1])
        fig.show()
        plt.pause(0.000005)
        #plt.pause(0.5)
        plt.clf()

    def predict(sess, tracknet, targetFrame, midFrame, searchFrame, bbTarget, annParseMid, annParseSearch, firstSemSuperIter=False):

        [bbx1, bby1, bbx2, bby2] = bbTarget

        # Prepeare the net inputs
        cx = bbx1 + ((bbx2 - bbx1)/2)
        cy = bby1 + ((bby2 - bby1)/2)
        bbPadsH = 3*((bby2 - bby1)/2)
        bbPadsW = 3*((bbx2 - bbx1)/2)

        startCropY = 0 if cy-bbPadsH < 0 else cy-bbPadsH
        endCropY = targetFrame.shape[0]-1 if cy+bbPadsH > targetFrame.shape[0]-1 else cy+bbPadsH
        startCropX = 0 if cx-bbPadsW < 0 else cx-bbPadsW
        endCropX = targetFrame.shape[1]-1 if cx+bbPadsW > targetFrame.shape[1]-1 else cx+bbPadsW

        targetCrop = targetFrame[startCropY:endCropY, startCropX:endCropX]

        endCropY = midFrame.shape[0]-1 if cy+bbPadsH > midFrame.shape[0]-1 else cy+bbPadsH
        endCropX = midFrame.shape[1]-1 if cx+bbPadsW > midFrame.shape[1]-1 else cx+bbPadsW

        midCrop = midFrame[startCropY:endCropY, startCropX:endCropX]

        endCropY = searchFrame.shape[0]-1 if cy+bbPadsH > searchFrame.shape[0]-1 else cy+bbPadsH
        endCropX = searchFrame.shape[1]-1 if cx+bbPadsW > searchFrame.shape[1]-1 else cx+bbPadsW

        searchCrop = searchFrame[startCropY:endCropY, startCropX:endCropX]

        # opencv reads as BGR and tensorflow gets RGB
        target = cv2.resize(targetCrop[:,:,::-1], (227,227))
        mid = cv2.resize(midCrop[:,:,::-1], (227,227))
        search = cv2.resize(searchCrop[:,:,::-1], (227,227))
        target = np.expand_dims(target, axis=0)
        mid = np.expand_dims(mid, axis=0)
        search = np.expand_dims(search, axis=0)

        # Infer
        [res] = sess.run([tracknet.result], feed_dict={tracknet.target: target, tracknet.mid: mid, tracknet.search: search})

        # Convert resulted BB to image cords
        resBBoxMid = np.squeeze(res['bbox_mid'])
        resBBoxSearch = np.squeeze(res['bbox_search'])
        bbx1PredMid = int(resBBoxMid[0]*midCrop.shape[1]) + startCropX
        bby1PredMid = int(resBBoxMid[1]*midCrop.shape[0]) + startCropY
        bbx2PredMid = int(resBBoxMid[2]*midCrop.shape[1]) + startCropX
        bby2PredMid = int(resBBoxMid[3]*midCrop.shape[0]) + startCropY
        bbx1PredSearch = int(resBBoxSearch[0]*searchCrop.shape[1]) + startCropX
        bby1PredSearch = int(resBBoxSearch[1]*searchCrop.shape[0]) + startCropY
        bbx2PredSearch = int(resBBoxSearch[2]*searchCrop.shape[1]) + startCropX
        bby2PredSearch = int(resBBoxSearch[3]*searchCrop.shape[0]) + startCropY
        cv2.rectangle(midFrame, (bbx1PredMid,bby1PredMid), (bbx2PredMid,bby2PredMid), (0,0,255), 3)
        if firstSemSuperIter:
            cv2.rectangle(searchFrame, (bbx1PredSearch,bby1PredSearch), (bbx2PredSearch,bby2PredSearch), (0,0,255), 3)
        predBB_mid = [bbx1PredMid, bby1PredMid, bbx2PredMid, bby2PredMid]
        predBB_search = [bbx1PredSearch, bby1PredSearch, bbx2PredSearch, bby2PredSearch]

        # Calculate IOU
        [bbx1GT_mid, bby1GT_mid, bbx2GT_mid, bby2GT_mid] = extBBvot(annParseMid)
        [bbx1GT_search, bby1GT_search, bbx2GT_search, bby2GT_search] = extBBvot(annParseSearch)
        cv2.rectangle(midFrame, (bbx1GT_mid,bby1GT_mid), (bbx2GT_mid,bby2GT_mid), (0,255,0), 3)
        if firstSemSuperIter:
            cv2.rectangle(searchFrame, (bbx1GT_search,bby1GT_search), (bbx2GT_search,bby2GT_search), (0,255,0), 3)
        gtBB_mid = [bbx1GT_mid, bby1GT_mid, bbx2GT_mid, bby2GT_mid]
        gtBB_search = [bbx1GT_search, bby1GT_search, bbx2GT_search, bby2GT_search]
        iouMid = bb_intersection_over_union(predBB_mid, gtBB_mid)
        iouSearch = bb_intersection_over_union(predBB_search, gtBB_search)

        return iouMid, iouSearch, predBB_mid, predBB_search


    # Load the video frames and annotations
    video = sys.argv[1]
    model = "./models/final.ckpt" #sys.argv[2]
    tf.reset_default_graph()
    if not os.path.exists(video):
        print "No such a video!"
        exit(1)

    videoAnn = video + "/groundtruth.txt"
    framesAnn = [line.rstrip() for line in open(videoAnn).readlines()]
    frames = glob.glob(video + "/*.jpg")
    frames.sort()

    vidName = os.path.basename(os.path.normpath(video))
    vidHeight, vidWidth, _ = (cv2.imread(frames[0])).shape
    vidOut = cv2.VideoWriter('{}.avi'.format(vidName), cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (vidWidth,vidHeight))

    with tf.Session() as sess:
        # Initiate the network
        tracknet = net.TRACKNET(1)
        saver = tf.train.Saver()
        saver.restore(sess, model)

        fig = plt.figure()
        predBB = [0,0,0,0]
        initCounter = -1
        iouTot = 0
        iouSearch = 0
        timerCount = 0
        for frameAnnIndx in xrange(4, len(framesAnn), 4):
            startTimer = time.time()

            #----------------------------#
            # Working on frames x1,x3,x5 #
            #----------------------------#
            # Extract GT annotation or predicted annotation
            annParseTarget = [int(float(number)) for number in framesAnn[frameAnnIndx-4].split(',')]
            annParseMid = [int(float(number)) for number in framesAnn[frameAnnIndx-2].split(',')]
            annParseSearch = [int(float(number)) for number in framesAnn[frameAnnIndx].split(',')]

            if iouSearch < 0.3:
                initCounter += 1
                bbTarget = extBBvot(annParseTarget)
            else:
                bbTarget = predBB_search

            # Read frames
            targetFrame = cv2.imread(frames[frameAnnIndx-4])
            midFrame = cv2.imread(frames[frameAnnIndx-2])
            searchFrame = cv2.imread(frames[frameAnnIndx])

            iouMid, iouSearch, predBB_mid, predBB_search = predict(sess, tracknet, targetFrame, midFrame, searchFrame, bbTarget, annParseMid, annParseSearch, True)
            iouTot += iouMid
            iouTot += iouSearch

            #----------------------------#
            # Working on frames x1,x2,x3 #
            #----------------------------#
            annParseMid1 = [int(float(number)) for number in framesAnn[frameAnnIndx-3].split(',')]
            midFrame1 = cv2.imread(frames[frameAnnIndx-3])

            iouMid1, _, _, _ = predict(sess, tracknet, targetFrame, midFrame1, midFrame, bbTarget, annParseMid1, annParseMid)
            iouTot += iouMid1

            #----------------------------#
            # Working on frames x3,x4,x5 #
            #----------------------------#
            annParseMid2 = [int(float(number)) for number in framesAnn[frameAnnIndx-1].split(',')]
            midFrame2 = cv2.imread(frames[frameAnnIndx-1])

            iouMid2, _, _, _ = predict(sess, tracknet, midFrame, midFrame2, searchFrame, predBB_mid, annParseMid2, annParseSearch)
            iouTot += iouMid2

            # View frames
            endTimer = time.time()
            timerCount += endTimer-startTimer
            plt.title("Frame - Mid1, FPS - {}, IoU - {}, initNum - {}".format(round(1/(endTimer-startTimer), 3), round(iouMid1,2), initCounter))
            view(midFrame1, vidOut)
            plt.title("Frame - Mid, FPS - {}, IoU - {}, initNum - {}".format(round(1/(endTimer-startTimer), 3), round(iouMid,2), initCounter))
            view(midFrame, vidOut)
            plt.title("Frame - Mid2, FPS - {}, IoU - {}, initNum - {}".format(round(1/(endTimer-startTimer), 3), round(iouMid2,2), initCounter))
            view(midFrame2, vidOut)
            plt.title("Frame - Search, FPS - {}, IoU - {}, initNum - {}".format(round(1/(endTimer-startTimer), 3), round(iouSearch,2), initCounter))
            view(searchFrame, vidOut)

    # Calculate accuracy, robustness and overall error
    avgFPS = round((len(framesAnn)-1)/timerCount, 3)
    A = round(iouTot/(len(framesAnn)-1), 3)
    R = round(1-(float(initCounter)/(len(framesAnn)-1)), 3)
    print "Clip: {}\nFPS = {}\nAverage IoU error = {}\nRobustnes error = {}\nOverall error = {}".format(video, avgFPS, 1-A, 1-R, 1-((A+R)/2))
    vidOut.release()


if __name__ == '__main__':
    sys.exit(main())
