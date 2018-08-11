import cv2
import numpy as np
import os
import sys
from random import shuffle

alovTrainFileLines = [line.rstrip() for line in open("./alovTrainSet.txt").readlines()]
imageNetTrainFileLines = [line.rstrip() for line in open("./imageNetTrainSet.txt").readlines()]
shuffle(alovTrainFileLines)
shuffle(imageNetTrainFileLines)
trainSet = open("trainSet.txt", "w")
maxTrainSetLenght = max(len(imageNetTrainFileLines), len(alovTrainFileLines))
minTrainSetLenght = min(len(imageNetTrainFileLines), len(alovTrainFileLines))
appearRatioForShortList = maxTrainSetLenght/(minTrainSetLenght*2)
l = -1
for i in range(maxTrainSetLenght):

    if not i % appearRatioForShortList:
        l += 1
        idxForShortList = l % minTrainSetLenght
        parseLineImageNet = imageNetTrainFileLines[idxForShortList].split(',') # TODO change list file to be generic
        searchImageNet = cv2.imread(parseLineImageNet[3])
        heightImageNet = searchImageNet.shape[0]
        widhtImageNet = searchImageNet.shape[1]
        # Write ImageNet
        [bbx1, bby1, bbx2, bby2] = [float(parseLineImageNet[4])/widhtImageNet, float(parseLineImageNet[5])/heightImageNet, float(parseLineImageNet[6])/widhtImageNet, float(parseLineImageNet[7])/heightImageNet]
        trainSet.write("{},{},{},{},{},{},{},{}\n".format(parseLineImageNet[0], parseLineImageNet[1], parseLineImageNet[2], parseLineImageNet[3], bbx1, bby1, bbx2, bby2))

    parseLineAlov = alovTrainFileLines[i].split(',')
    searchAlov = cv2.imread(parseLineAlov[3])
    heightAlov = searchAlov.shape[0]
    widhtAlov = searchAlov.shape[1]
    # Write Alov
    [bbx1, bby1, bbx2, bby2] = [float(parseLineAlov[4])/widhtAlov, float(parseLineAlov[5])/heightAlov, float(parseLineAlov[6])/widhtAlov, float(parseLineAlov[7])/heightAlov]
    trainSet.write("{},{},{},{},{},{},{},{}\n".format(parseLineAlov[0], parseLineAlov[1], parseLineAlov[2], parseLineAlov[3], bbx1, bby1, bbx2, bby2))

trainSet.close()

testFileLines = [line.rstrip() for line in open("./votTestSet.txt").readlines()]
shuffle(testFileLines)
testSet = open("testSet.txt", "w")
for i in range(len(testFileLines)):
    parseLine = testFileLines[i].split(',')
    mid1Image = cv2.imread(parseLine[1])
    mid2Image = cv2.imread(parseLine[2])
    searchImage = cv2.imread(parseLine[3])
    # Note: same augmentation for mid1 and mid2 as for search image, thus, using same width and height to scale 0-1
    widht = searchImage.shape[1]
    height = searchImage.shape[0]
    # Write VOT
    [bbx1M1, bby1M1, bbx2M1, bby2M1] = [float(parseLine[4])/widht, float(parseLine[5])/height, float(parseLine[6])/widht, float(parseLine[7])/height]
    [bbx1M2, bby1M2, bbx2M2, bby2M2] = [float(parseLine[8])/widht, float(parseLine[9])/height, float(parseLine[10])/widht, float(parseLine[11])/height]
    [bbx1S, bby1S, bbx2S, bby2S] = [float(parseLine[12])/widht, float(parseLine[13])/height, float(parseLine[14])/widht, float(parseLine[15])/height]
    testSet.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(parseLine[0], parseLine[1], parseLine[2], parseLine[3], bbx1M1, bby1M1, bbx2M1, bby2M1, bbx1M2, bby1M2, bbx2M2, bby2M2, bbx1S, bby1S, bbx2S, bby2S))

testSet.close()
