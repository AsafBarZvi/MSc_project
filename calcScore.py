import cv2
import tensorflow as tf
import numpy as np
import glob
import re
import os
import sys
import time
import importlib
#import ipdb


def main():

    # Setup a free GPU core
    availableGPU = None
    for gpuId in range(int(os.popen("nvidia-smi -L | wc -l").readlines()[0])):
        if int(os.popen("nvidia-smi -i {} -q --display=MEMORY | grep -m 1 Free | grep -o '[0-9]*'".format(gpuId)).readlines()[0]) < 1000:
            continue
        availableGPU = gpuId
        break
    if availableGPU == None:
        print "No available GPU device!"
        sys.exit(1)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(availableGPU)
    print('Runing on GPU: {}'.format(availableGPU))

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

    # Load all VOT videos
    videos = glob.glob("./data/votTestData/*")
    tf.reset_default_graph()
    model = sys.argv[1]
    dataScheme = int(sys.argv[2])
    if dataScheme == 1:
        net = importlib.import_module('net_scheme1')
        k1 = 3
        k2 = 3
    else:
        net = importlib.import_module('net')
        k1 = 1
        k2 = 4

    robTot = 0
    iouTot = 0
    videoCounter = 0
    with tf.Session() as sess:
        # Initiate the network
        tracknet = net.TRACKNET(1)
        saver = tf.train.Saver()
        saver.restore(sess, model)

        predBB = [0,0,0,0]
        for video in videos:
            if re.search("list\.txt", video):
                print "skip category list file"
                continue

            videoAnn = video + "/groundtruth.txt"
            framesAnn = [line.rstrip() for line in open(videoAnn).readlines()]
            frames = glob.glob(video + "/*.jpg")
            frames.sort()

            initCounterSingleVid = -1
            iouTotSingleVid = 0
            iou = 0
            startTimer = time.time()
            for frameAnnIndx in xrange(2, len(framesAnn), 2):
                # Extract GT annotation or predicted annotation
                annParseTarget = [int(float(number)) for number in framesAnn[frameAnnIndx-2].split(',')]
                annParseMid = [int(float(number)) for number in framesAnn[frameAnnIndx-1].split(',')]
                annParseSearch = [int(float(number)) for number in framesAnn[frameAnnIndx].split(',')]

                if iou < 0.5:
                    initCounterSingleVid += 1
                    [bbx1, bby1, bbx2, bby2] = extBBvot(annParseTarget)
                else:
                    [bbx1, bby1, bbx2, bby2] = predBB_search

                # Read the pair of frames
                targetFrame = cv2.imread(frames[frameAnnIndx-2])
                midFrame = cv2.imread(frames[frameAnnIndx-1])
                searchFrame = cv2.imread(frames[frameAnnIndx])

                # Prepeare the net inputs
                cx = bbx1 + ((bbx2 - bbx1)/2)
                cy = bby1 + ((bby2 - bby1)/2)
                bbPadsH = k1*((bby2 - bby1)/2)
                bbPadsW = k1*((bbx2 - bbx1)/2)

                startCropY = 0 if cy-bbPadsH < 0 else cy-bbPadsH
                endCropY = targetFrame.shape[0]-1 if cy+bbPadsH > targetFrame.shape[0]-1 else cy+bbPadsH
                startCropX = 0 if cx-bbPadsW < 0 else cx-bbPadsW
                endCropX = targetFrame.shape[1]-1 if cx+bbPadsW > targetFrame.shape[1]-1 else cx+bbPadsW

                targetCrop = targetFrame[startCropY:endCropY, startCropX:endCropX]
                if dataScheme == 1:
                    if endCropY - startCropY < 2*bbPadsH:
                        zPads = np.zeros(((2*bbPadsH)-(endCropY-startCropY), targetCrop.shape[1], 3), dtype=np.uint8)
                        if startCropY == 0:
                            targetCrop = np.concatenate((zPads, targetCrop), axis=0)
                        else:
                            targetCrop = np.concatenate((targetCrop, zPads), axis=0)

                    if endCropX - startCropX < 2*bbPadsW:
                        zPads = np.zeros((targetCrop.shape[0], (2*bbPadsW)-(endCropX-startCropX), 3), dtype=np.uint8)
                        if startCropX == 0:
                            targetCrop = np.concatenate((zPads, targetCrop), axis=1)
                        else:
                            targetCrop = np.concatenate((targetCrop, zPads), axis=1)


                bbPadsH = k2*((bby2 - bby1)/2)
                bbPadsW = k2*((bbx2 - bbx1)/2)

                startCropY = 0 if cy-bbPadsH < 0 else cy-bbPadsH
                endCropY = midFrame.shape[0]-1 if cy+bbPadsH > midFrame.shape[0]-1 else cy+bbPadsH
                startCropX = 0 if cx-bbPadsW < 0 else cx-bbPadsW
                endCropX = midFrame.shape[1]-1 if cx+bbPadsW > midFrame.shape[1]-1 else cx+bbPadsW

                midCrop = midFrame[startCropY:endCropY, startCropX:endCropX]

                startCropY = 0 if cy-bbPadsH < 0 else cy-bbPadsH
                endCropY = searchFrame.shape[0]-1 if cy+bbPadsH > searchFrame.shape[0]-1 else cy+bbPadsH
                startCropX = 0 if cx-bbPadsW < 0 else cx-bbPadsW
                endCropX = searchFrame.shape[1]-1 if cx+bbPadsW > searchFrame.shape[1]-1 else cx+bbPadsW

                searchCrop = searchFrame[startCropY:endCropY, startCropX:endCropX]

                # opencv reads as BGR and tensorflow gets RGB
                if dataScheme == 1:
                    target = cv2.resize(targetCrop[:,:,::-1], (227,227))
                    mid = cv2.resize(midCrop[:,:,::-1], (227,227))
                    search = cv2.resize(searchCrop[:,:,::-1], (227,227))
                else:
                    target = cv2.resize(targetCrop[:,:,::-1], (100,100))
                    mid = cv2.resize(midCrop[:,:,::-1], (400,400))
                    search = cv2.resize(searchCrop[:,:,::-1], (400,400))

                target = np.expand_dims(target, axis=0)
                mid = np.expand_dims(mid, axis=0)
                search = np.expand_dims(search, axis=0)

                # Infer
                [res] = sess.run([tracknet.result], feed_dict={tracknet.target: target, tracknet.mid: mid, tracknet.search: search})

                # Convert resulted BB to image cords
                resBBoxMid = np.squeeze(res['bbox_mid'])
                resBBoxSearch = np.squeeze(res['bbox_search'])
                bbx1Pred = int(resBBoxMid[0]*midCrop.shape[1]) + startCropX
                bby1Pred = int(resBBoxMid[1]*midCrop.shape[0]) + startCropY
                bbx2Pred = int(resBBoxMid[2]*midCrop.shape[1]) + startCropX
                bby2Pred = int(resBBoxMid[3]*midCrop.shape[0]) + startCropY
                bbx1PredSearch = int(resBBoxSearch[0]*searchCrop.shape[1]) + startCropX
                bby1PredSearch = int(resBBoxSearch[1]*searchCrop.shape[0]) + startCropY
                bbx2PredSearch = int(resBBoxSearch[2]*searchCrop.shape[1]) + startCropX
                bby2PredSearch = int(resBBoxSearch[3]*searchCrop.shape[0]) + startCropY
                predBB_mid = [bbx1Pred, bby1Pred, bbx2Pred, bby2Pred]
                predBB_search = [bbx1PredSearch, bby1PredSearch, bbx2PredSearch, bby2PredSearch]

                # Calculate IOU
                [bbx1GT_mid, bby1GT_mid, bbx2GT_mid, bby2GT_mid] = extBBvot(annParseMid)
                gtBB = [bbx1GT_mid, bby1GT_mid, bbx2GT_mid, bby2GT_mid]
                iou = bb_intersection_over_union(predBB_mid, gtBB)
                iouTotSingleVid += iou
                if iou < 0.3:
                    initCounterSingleVid += 1

                [bbx1GT_search, bby1GT_search, bbx2GT_search, bby2GT_search] = extBBvot(annParseSearch)
                gtBB = [bbx1GT_search, bby1GT_search, bbx2GT_search, bby2GT_search]
                iou = bb_intersection_over_union(predBB_search, gtBB)
                iouTotSingleVid += iou


            # Calculate accuracy, robustness and overall error per current video
            endTimer = time.time()
            avgFPS = round((len(framesAnn)-1)/(endTimer-startTimer), 3)
            A = round(iouTotSingleVid/(len(framesAnn)-1), 3)
            R = round(1-(float(initCounterSingleVid)/(len(framesAnn)-1)), 3)
            iouTot += A
            robTot += R
            videoCounter += 1
            print "Clip: {}\nFPS = {}\nAverage IoU error = {}\nRobustnes error = {}\nOverall error = {}".format(video, avgFPS, 1-A, 1-R, 1-((A+R)/2))


    # Calculate accuracy, robustness and overall error
    A = round(iouTot/videoCounter, 3)
    R = round(robTot/videoCounter, 3)
    print "\nVOT results summary:\nAverage IoU error = {}\nRobustnes error = {}\nOverall error = {}".format(1-A, 1-R, 1-((A+R)/2))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: python calcScore.py <pathToModel> <dataScheme(1|2)>"
        sys.exit(0)

    sys.exit(main())
