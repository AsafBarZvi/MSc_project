import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import glob
import re
import os
import sys
import time
import importlib
#import ipdb

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

    def view(img, vidOut=None):
        #vidOut.write(img)
        plt.imshow(img[:,:,::-1])
        fig.show()
        plt.pause(0.000005)
        #plt.pause(0.5)
        plt.clf()


    # Load the video frames and annotations
    video = sys.argv[1]
    model = sys.argv[2]
    tf.reset_default_graph()
    if not os.path.exists(video):
        print "No such a video!"
        exit(1)

    dataScheme = int(sys.argv[3])
    if dataScheme == 1:
        net = importlib.import_module('net_scheme1')
        k1 = 3
        k2 = 3
    else:
        net = importlib.import_module('net')
        k1 = 1
        k2 = 4


    videoAnn = video + "/groundtruth.txt"
    framesAnn = [line.rstrip() for line in open(videoAnn).readlines()]
    frames = glob.glob(video + "/*.jpg")
    frames.sort()

    #vidName = os.path.basename(os.path.normpath(video))
    #vidHeight, vidWidth, _ = (cv2.imread(frames[0])).shape
    #vidOut = cv2.VideoWriter('./temp/{}_mulView.avi'.format(vidName), cv2.VideoWriter_fourcc('M','J','P','G'), 20.0, (vidWidth*2,vidHeight))

    with tf.Session() as sess:
        # Initiate the network
	#meta = model + ".meta"
        #saver = tf.train.import_meta_graph(meta, clear_devices=True)
	#saver.restore(sess, model)
	#targetPH = sess.graph.get_tensor_by_name("Placeholder:0")
	#midPH = sess.graph.get_tensor_by_name("Placeholder_1:0")
	#searchPH = sess.graph.get_tensor_by_name("Placeholder_2:0")
	#outputPH = sess.graph.get_tensor_by_name("fc_nets/strided_slice:0")
        #result = {'bbox_mid': outputPH[:,:4], 'bbox_search': outputPH[:,4:]}

        tracknet = net.TRACKNET(1)
        saver = tf.train.Saver()
        saver.restore(sess, model)

        fig = plt.figure()
        predBB = [0,0,0,0]
        initCounter = -1
        iouTot = 0
        iou = 0
        timerCount = 0
        for frameAnnIndx in xrange(2, len(framesAnn), 1):
            startTimer = time.time()
            # Extract GT annotation or predicted annotation
            annParseTarget = [int(float(number)) for number in framesAnn[frameAnnIndx-2].split(',')]
            annParseMid = [int(float(number)) for number in framesAnn[frameAnnIndx-1].split(',')]
            annParseSearch = [int(float(number)) for number in framesAnn[frameAnnIndx].split(',')]

            if frameAnnIndx == 1 or iou < 0.5:
                initCounter += 1
                [bbx1, bby1, bbx2, bby2] = extBBvot(annParseTarget)
            else:
                [bbx1, bby1, bbx2, bby2] = predBB_mid

            # Read frames
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
            #[res] = sess.run([result], feed_dict={targetPH: target, midPH: mid, searchPH: search})
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
            cv2.rectangle(midFrame, (bbx1Pred,bby1Pred), (bbx2Pred,bby2Pred), (0,255,255), 3)
            cv2.rectangle(searchFrame, (bbx1PredSearch,bby1PredSearch), (bbx2PredSearch,bby2PredSearch), (0,0,255), 3)
            predBB_mid = [bbx1Pred, bby1Pred, bbx2Pred, bby2Pred]
            predBB_search = [bbx1PredSearch, bby1PredSearch, bbx2PredSearch, bby2PredSearch]

            # Calculate IOU
            [bbx1GT_mid, bby1GT_mid, bbx2GT_mid, bby2GT_mid] = extBBvot(annParseMid)
            #[bbx1GT_search, bby1GT_search, bbx2GT_search, bby2GT_search] = extBBvot(annParseSearch)
            cv2.rectangle(midFrame, (bbx1GT_mid,bby1GT_mid), (bbx2GT_mid,bby2GT_mid), (0,255,0), 3)
            #cv2.rectangle(searchFrame, (bbx1GT_search,bby1GT_search), (bbx2GT_search,bby2GT_search), (0,255,0), 3)
            #gtBB = [bbx1GT_search, bby1GT_search, bbx2GT_search, bby2GT_search]
            #iou = bb_intersection_over_union(predBB_search, gtBB)
            #iouTot += iou
            gtBB = [bbx1GT_mid, bby1GT_mid, bbx2GT_mid, bby2GT_mid]
            iou = bb_intersection_over_union(predBB_mid, gtBB)
            iouTot += iou

            # View frame
            endTimer = time.time()
            timerCount += endTimer-startTimer
            plt.title("FPS - {}, IoU - {}, initNum - {}".format(round(1/(endTimer-startTimer), 3), round(iou,2), initCounter))
            #view(np.concatenate((midFrame,searchFrame), axis=1))#, vidOut)
            view(midFrame)

    # Calculate accuracy, robustness and overall error
    avgFPS = round((len(framesAnn)-1)/timerCount, 3)
    A = round(iouTot/(len(framesAnn)-1), 3)
    R = round(1-(float(initCounter)/(len(framesAnn)-1)), 3)
    print "Clip: {}\nFPS = {}\nAverage IoU error = {}\nRobustnes error = {}\nOverall error = {}".format(video, avgFPS, 1-A, 1-R, 1-((A+R)/2))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print "Usage: python runTracker_mulView.py <pathToVideo> <pathToModel> <dataScheme(1|2)>"
        sys.exit(0)

    sys.exit(main())
