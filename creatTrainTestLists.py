import cv2
import numpy as np
import os
import sys
from random import shuffle

def creatLists(dataExtDir):

    imageNetTrainFileLines = [line.rstrip() for line in open(dataExtDir + "/imageNetTrainSet.txt").readlines()]
    alovTrainFileLines = [line.rstrip() for line in open(dataExtDir + "/alovTrainSet.txt").readlines()]
    trainLists = [imageNetTrainFileLines, alovTrainFileLines]
    shuffle(imageNetTrainFileLines)
    shuffle(alovTrainFileLines)
    trainSet = open(dataExtDir + "/trainSet.txt", "w")
    maxTrainSetLenght = max(len(imageNetTrainFileLines), len(alovTrainFileLines))
    minTrainSetLenght = min(len(imageNetTrainFileLines), len(alovTrainFileLines))
    largerListIdx = 0 if len(imageNetTrainFileLines) > len(alovTrainFileLines) else 1
    appearRatioForShortList = maxTrainSetLenght/minTrainSetLenght
    l = -1
    for i in range(maxTrainSetLenght):

        # Write from short list
        if not i % appearRatioForShortList:
            l += 1
            idxForShortList = l % minTrainSetLenght
            parseLine = trainLists[1-largerListIdx][idxForShortList].split(',')
            search = cv2.imread(parseLine[2])
            height = search.shape[0]
            width = search.shape[1]
            [bbx1, bby1, bbx2, bby2] = [float(parseLine[3])/width, float(parseLine[4])/height, float(parseLine[5])/width, float(parseLine[6])/height]
            trainSet.write("{},{},{},{},{},{},{}\n".format(parseLine[0], parseLine[1], parseLine[2], bbx1, bby1, bbx2, bby2))

        # Write from large list
        parseLine = trainLists[largerListIdx][i].split(',')
        search = cv2.imread(parseLine[2])
        height = search.shape[0]
        width = search.shape[1]
        [bbx1, bby1, bbx2, bby2] = [float(parseLine[3])/width, float(parseLine[4])/height, float(parseLine[5])/width, float(parseLine[6])/height]
        trainSet.write("{},{},{},{},{},{},{}\n".format(parseLine[0], parseLine[1], parseLine[2], bbx1, bby1, bbx2, bby2))

    trainSet.close()

    testFileLines = [line.rstrip() for line in open(dataExtDir + "/votTestSet.txt").readlines()]
    shuffle(testFileLines)
    testSet = open(dataExtDir + "/testSet.txt", "w")
    for i in range(len(testFileLines)):
        parseLine = testFileLines[i].split(',')
        midImage = cv2.imread(parseLine[1])
        searchImage = cv2.imread(parseLine[2])
        # Note: same augmentation for mid as for search image, thus, using same width and height to scale 0-1
        width = searchImage.shape[1]
        height = searchImage.shape[0]
        # Write VOT
        [bbx1M, bby1M, bbx2M, bby2M] = [float(parseLine[3])/width, float(parseLine[4])/height, float(parseLine[5])/width, float(parseLine[6])/height]
        [bbx1S, bby1S, bbx2S, bby2S] = [float(parseLine[7])/width, float(parseLine[8])/height, float(parseLine[9])/width, float(parseLine[10])/height]
        testSet.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(parseLine[0], parseLine[1], parseLine[2], bbx1M, bby1M, bbx2M, bby2M, bbx1S, bby1S, bbx2S, bby2S))

    testSet.close()
