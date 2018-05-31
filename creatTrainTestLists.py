import cv2
import numpy as np
import os
import sys
from random import shuffle

alovTrainFileLines = [line.rstrip() for line in open("./alovTrainSet.txt").readlines()]
imageNetTrainFileLines = [line.rstrip() for line in open("./imageNetTrainSet.txt").readlines()]
shuffle(alovTrainFileLines)
shuffle(imageNetTrainFileLines)
trainSet = open("trainSet1.txt", "w")
trainSetLenght = max(len(imageNetTrainFileLines), len(alovTrainFileLines))
for i in range(trainSetLenght):
    idxForShortList = i % min(len(imageNetTrainFileLines), len(alovTrainFileLines))
    parseLineAlov = alovTrainFileLines[idxForShortList].split(',')
    parseLineImageNet = imageNetTrainFileLines[i].split(',')
    searchAlov = cv2.imread(parseLineAlov[1])
    searchImageNet = cv2.imread(parseLineImageNet[1])
    heightAlov = searchAlov.shape[0]
    widhtAlov = searchAlov.shape[1]
    heightImageNet = searchImageNet.shape[0]
    widhtImageNet = searchImageNet.shape[1]
    # Write Alov
    [bbx1, bby1, bbx2, bby2] = [float(parseLineAlov[2])/widhtAlov, float(parseLineAlov[3])/heightAlov, float(parseLineAlov[4])/widhtAlov, float(parseLineAlov[5])/heightAlov]
    trainSet.write("{},{},{},{},{},{}\n".format(parseLineAlov[0], parseLineAlov[1], bbx1, bby1, bbx2, bby2))
    # Write ImageNet
    [bbx1, bby1, bbx2, bby2] = [float(parseLineImageNet[2])/widhtImageNet, float(parseLineImageNet[3])/heightImageNet, float(parseLineImageNet[4])/widhtImageNet, float(parseLineImageNet[5])/heightImageNet]
    trainSet.write("{},{},{},{},{},{}\n".format(parseLineImageNet[0], parseLineImageNet[1], bbx1, bby1, bbx2, bby2))

trainSet.close()

testFileLines = [line.rstrip() for line in open("./votTestSet.txt").readlines()]
shuffle(testFileLines)
testSet = open("testSet1.txt", "w")
for i in range(len(testFileLines)):
    parseLine = testFileLines[i].split(',')
    searchImage = cv2.imread(parseLine[1])
    widht = searchImage.shape[1]
    height = searchImage.shape[0]
    # Write VOT
    [bbx1, bby1, bbx2, bby2] = [float(parseLine[2])/widht, float(parseLine[3])/height, float(parseLine[4])/widht, float(parseLine[5])/height]
    testSet.write("{},{},{},{},{},{}\n".format(parseLine[0], parseLine[1], bbx1, bby1, bbx2, bby2))

testSet.close()
