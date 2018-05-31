import cv2
import tensorflow as tf
import numpy as np
#from matplotlib import pyplot as plt
import glob
import re
import os
import sys
import time
#import ipdb

import goturn_net


gpu = 2
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

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

    #def view(img):
    #    plt.imshow(img[:,:,::-1])
    #    fig.show()
    #    plt.pause(0.000005)
    #    plt.clf()


    # Load all VOT videos
    videos = glob.glob("./data/votTestData/*")
    #model = "./snaps/goturnTrain_trainConvHighLR/33_11487.ckpt" #sys.argv[2]
    model = "./snaps/goturnTrain_noAugOnlyAlov/final.ckpt" #sys.argv[2]

    robTot = 0
    iouTot = 0
    videoCounter = 0
    with tf.Session() as sess:
        # Initiate the network
        tracknet = goturn_net.TRACKNET(1, 0.0005, False)
        tracknet.build()
        #init = tf.global_variables_initializer()
        #init_local = tf.local_variables_initializer()
        #sess.run(init)
        #sess.run(init_local)
        saver = tf.train.Saver()
        saver.restore(sess, model)

        #fig = plt.figure()
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
            startTimer = time.time()
            for frameAnnIndx in xrange(1,len(framesAnn)):
                # Extract GT annotation or predicted annotation
                annParseTarget = [int(float(number)) for number in framesAnn[frameAnnIndx-1].split(',')]
                annParseSearch = [int(float(number)) for number in framesAnn[frameAnnIndx].split(',')]

                if frameAnnIndx == 1 or iou < 0.3:
                    initCounterSingleVid += 1
                    [bbx1, bby1, bbx2, bby2] = extBBvot(annParseTarget)
                else:
                    [bbx1, bby1, bbx2, bby2] = predBB

                # Read the pair of frames
                targetFrame = cv2.imread(frames[frameAnnIndx-1])
                searchFrame = cv2.imread(frames[frameAnnIndx])

                # Prepeare the net inputs
                cx = bbx1 + ((bbx2 - bbx1)/2)
                cy = bby1 + ((bby2 - bby1)/2)
                bbPadsH = 2*((bby2 - bby1)/2)
                bbPadsW = 2*((bbx2 - bbx1)/2)

                startCropY = 0 if cy-bbPadsH < 0 else cy-bbPadsH
                endCropY = targetFrame.shape[0]-1 if cy+bbPadsH > targetFrame.shape[0]-1 else cy+bbPadsH
                startCropX = 0 if cx-bbPadsW < 0 else cx-bbPadsW
                endCropX = targetFrame.shape[1]-1 if cx+bbPadsW > targetFrame.shape[1]-1 else cx+bbPadsW

                targetCrop = targetFrame[startCropY:endCropY, startCropX:endCropX]
                searchCrop = searchFrame[startCropY:endCropY, startCropX:endCropX]

                # opencv reads as BGR and tensorflow gets RGB
                target = cv2.resize(targetCrop[:,:,::-1], (227,227))
                search = cv2.resize(searchCrop[:,:,::-1], (227,227))
                target = np.expand_dims(target, axis=0)
                search = np.expand_dims(search, axis=0)

                # Infer
                [res] = sess.run([tracknet.fc4], feed_dict={tracknet.image: search, tracknet.target: target})

                # Convert resulted BB to image cords
                res = np.squeeze(res)
                res = res/10
                bbx1Pred = int(res[0]*searchCrop.shape[1]) + startCropX
                bby1Pred = int(res[1]*searchCrop.shape[0]) + startCropY
                bbx2Pred = int(res[2]*searchCrop.shape[1]) + startCropX
                bby2Pred = int(res[3]*searchCrop.shape[0]) + startCropY
                #cv2.rectangle(searchFrame, (bbx1Pred,bby1Pred), (bbx2Pred,bby2Pred), (0,0,255), 3)
                predBB = [bbx1Pred, bby1Pred, bbx2Pred, bby2Pred]

                # Calculate IOU
                [bbx1GT, bby1GT, bbx2GT, bby2GT] = extBBvot(annParseSearch)
                #cv2.rectangle(searchFrame, (bbx1GT,bby1GT), (bbx2GT,bby2GT), (0,255,0), 3)
                gtBB = [bbx1GT, bby1GT, bbx2GT, bby2GT]
                iou = bb_intersection_over_union(predBB, gtBB)
                iouTotSingleVid += iou

                # View frame
                #plt.title("IoU - {}, initNum - {}".format(round(iou,2), initCounter))
                #view(searchFrame)

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
    sys.exit(main())
