import cv2
import tensorflow as tf
import tensorboard
import numpy as np
#from matplotlib import pyplot as plt
import glob
import re
import os
import xml.etree.ElementTree as ET


# Augmentation params as in the paper
augShift = 1./5
augScale = 1./15
minScale = 0.6
maxScale = 1.4
k1 = k2 = 2
k3 = 10

alovData = "./data/alovData"
alovDataGT = "./data/alovDataGT"
imageNetData = "./data/imageNetData"
imageNetDataGT = "./data/imageNetDataGT"
votData = "./data/votTestData"

alovExtdata = "./data/alovExtData"
if not os.path.exists(alovExtdata):
    os.mkdir(alovExtdata)

alovExtdataTarget = alovExtdata + "/target"
if not os.path.exists(alovExtdataTarget):
    os.mkdir(alovExtdataTarget)

alovExtdataSearching = alovExtdata + "/searching"
if not os.path.exists(alovExtdataSearching):
    os.mkdir(alovExtdataSearching)

imageNetExtdata = "./data/imageNetExtData"
if not os.path.exists(imageNetExtdata):
    os.mkdir(imageNetExtdata)

imageNetExtdataTarget = imageNetExtdata + "/target"
if not os.path.exists(imageNetExtdataTarget):
    os.mkdir(imageNetExtdataTarget)

imageNetExtdataSearching = imageNetExtdata + "/searching"
if not os.path.exists(imageNetExtdataSearching):
    os.mkdir(imageNetExtdataSearching)

votExtdata = "./data/votExtData"
if not os.path.exists(votExtdata):
    os.mkdir(votExtdata)

votExtdataTarget = votExtdata + "/target"
if not os.path.exists(votExtdataTarget):
    os.mkdir(votExtdataTarget)

votExtdataSearching = votExtdata + "/searching"
if not os.path.exists(votExtdataSearching):
    os.mkdir(votExtdataSearching)


def extBB(annParse):
    bbx1 = min(annParse[1],annParse[3],annParse[5],annParse[7])
    bby1 = min(annParse[2],annParse[4],annParse[6],annParse[8])
    bbx2 = max(annParse[1],annParse[3],annParse[5],annParse[7])
    bby2 = max(annParse[2],annParse[4],annParse[6],annParse[8])
    return [bbx1, bby1, bbx2, bby2]

def extBBvot(annParse):
    bbx1 = min(annParse[0],annParse[2],annParse[4],annParse[6])
    bby1 = min(annParse[1],annParse[3],annParse[5],annParse[7])
    bbx2 = max(annParse[0],annParse[2],annParse[4],annParse[6])
    bby2 = max(annParse[1],annParse[3],annParse[5],annParse[7])
    return [bbx1, bby1, bbx2, bby2]

def viewer(img):
    fig = plt.figure()
    #plt.imshow(cv2.resize(img[:,:,::-1], (227,227)))
    plt.imshow(img[:,:,::-1])
    fig.show()

def goturnAugmentation(augIdx, cxPrev, cyPrev, bbx1Prev, bby1Prev, bbx2Prev, bby2Prev, frameRes, bbx1Curr, bby1Curr, bbx2Curr, bby2Curr):
    width = bbx2Prev - bbx1Prev
    height = bby2Prev - bby1Prev
    cxCurr = bbx1Prev + (width/2)
    cyCurr = bby1Prev + (height/2)
    newWidth = -1
    newHeight = -1
    newCx = -1
    newCy = -1

    if augIdx == 0:
        newHeight = frameCurr.shape[0]-1 if k2*height > frameCurr.shape[0]-1 else k2*height
        newWidth = frameCurr.shape[1]-1 if k2*width > frameCurr.shape[1]-1 else k2*width
        newCx = cxCurr
        newCy = cyCurr
        startCropCurrY = 0 if cyCurr-newHeight/2 < 0 else cyCurr-newHeight/2
        endCropCurrY = frameCurr.shape[0]-1 if cyCurr+newHeight/2 > frameCurr.shape[0]-1 else cyCurr+newHeight/2
        startCropCurrX = 0 if cxCurr-newWidth/2 < 0 else cxCurr-newWidth/2
        endCropCurrX = frameCurr.shape[1]-1 if cxCurr+newWidth/2 > frameCurr.shape[1]-1 else cxCurr+newWidth/2
        bbx1New = 0 if newCx-newWidth/2 < 0 else newCx-newWidth/2
        bby1New = 0 if newCy-newHeight/2 < 0 else newCy-newHeight/2

    else:
        numOfTries = 10
        while (newWidth < 0 or newWidth > frameRes[1]-1) and numOfTries:
            scaleW = max(minScale, min(maxScale, np.random.laplace(1, augScale)))
            newWidth = int(k2*width*scaleW)
            numOfTries -= 1
            if numOfTries == 0:
                return [0, 0, 0, 0, 0, 0, 0, 0, False]

        numOfTries = 10
        while (newHeight < 0 or newHeight > frameRes[0]-1) and numOfTries:
            scaleH = max(minScale, min(maxScale, np.random.laplace(1, augScale)))
            newHeight = int(k2*height*scaleH)
            numOfTries -= 1
            if numOfTries == 0:
                return [0, 0, 0, 0, 0, 0, 0, 0, False]

        numOfTries = 10
        firstIter = True
        while (firstIter or newCx < cxCurr-((width*k2)/2) or newCx > cxCurr+((width*k2)/2) or newCx-newWidth/2 < 0 or newCx+newWidth/2 > frameRes[1]-1) and numOfTries:
            newCx = cxCurr + width*np.random.laplace(0, augShift)
            newCx = int(min(frameRes[1]-newWidth/2, max(newWidth/2,newCx)))
            numOfTries -= 1
            firstIter = False
            if numOfTries == 0:
                return [0, 0, 0, 0, 0, 0, 0, 0, False]

        numOfTries = 10
        firstIter = True
        while (firstIter or newCy < cyCurr-((height*k2)/2) or newCy > cyCurr+((height*k2)/2) or newCy-newHeight/2 < 0 or newCy+newHeight/2 > frameRes[0]-1) and numOfTries:
            newCy = cyCurr + height*np.random.laplace(0, augShift)
            newCy = int(min(frameRes[1]-newHeight/2, max(newHeight/2,newCy)))
            numOfTries -= 1
            firstIter = False
            if numOfTries == 0:
                return [0, 0, 0, 0, 0, 0, 0, 0, False]

        bbx1New = newCx - newWidth/2
        bbx2New = newCx + newWidth/2
        bby1New = newCy - newHeight/2
        bby2New = newCy + newHeight/2

        startCropCurrY = bby1New
        endCropCurrY = bby2New
        startCropCurrX = bbx1New
        endCropCurrX = bbx2New


    bbx1CurrCrop = 0 if bbx1Curr-bbx1New < 0 else bbx1Curr-bbx1New
    bbx2CurrCrop = 0 if bbx2Curr-bbx1New < 0 else bbx2Curr-bbx1New
    bby1CurrCrop = 0 if bby1Curr-bby1New < 0 else bby1Curr-bby1New
    bby2CurrCrop = 0 if bby2Curr-bby1New < 0 else bby2Curr-bby1New

    return [bbx1CurrCrop, bby1CurrCrop, bbx2CurrCrop, bby2CurrCrop, startCropCurrY, endCropCurrY, startCropCurrX, endCropCurrX, True]


# Extract Alov data
print "Extracting Alov data..."

alovTrainSet = open("alovTrainSet.txt", "w")

categories = glob.glob(alovData + "/*")
for category in categories:
    if re.search("notToUseForTrain", category):
        print "skip ALOV duplication with VOT..."
        continue
    videos = glob.glob(category + "/*")
    for video in videos:
        videoAnn = re.sub("\/alovData\/", "/alovDataGT/", video)
        videoAnn = re.sub("$", ".ann", videoAnn)
        framesAnn = [line.rstrip() for line in open(videoAnn).readlines()]
        frames = glob.glob(video + "/*.jpg")
        frames.sort()
        if len(framesAnn) < 2:
            print "video {} have less then 2 annotated frames".format(video.split("/")[-1])
            continue
        for frameAnnIndx in xrange(1, len(framesAnn), 1):
            annParsePrev = [int(float(number)) for number in framesAnn[frameAnnIndx-1].split()]
            annParseCurr = [int(float(number)) for number in framesAnn[frameAnnIndx].split()]
            framePrev = cv2.imread(frames[annParsePrev[0]-1])
            frameCurr = cv2.imread(frames[annParseCurr[0]-1])
            [bbx1Prev, bby1Prev, bbx2Prev, bby2Prev] = extBB(annParsePrev)
            [bbx1Curr, bby1Curr, bbx2Curr, bby2Curr] = extBB(annParseCurr)

            cxPrev = bbx1Prev + ((bbx2Prev - bbx1Prev)/2)
            cyPrev = bby1Prev + ((bby2Prev - bby1Prev)/2)
            bbPadsPrevH = k1*((bby2Prev - bby1Prev)/2)
            bbPadsPrevW = k1*((bbx2Prev - bbx1Prev)/2)

            startCropPrevY = 0 if cyPrev-bbPadsPrevH < 0 else cyPrev-bbPadsPrevH
            endCropPrevY = framePrev.shape[0]-1 if cyPrev+bbPadsPrevH > framePrev.shape[0]-1 else cyPrev+bbPadsPrevH
            startCropPrevX = 0 if cxPrev-bbPadsPrevW < 0 else cxPrev-bbPadsPrevW
            endCropPrevX = framePrev.shape[1]-1 if cxPrev+bbPadsPrevW > framePrev.shape[1]-1 else cxPrev+bbPadsPrevW

            framePrevCropPads = framePrev[startCropPrevY:endCropPrevY, startCropPrevX:endCropPrevX]
            fileNameTarget = "{}/{}_prev_{}".format(alovExtdataTarget, (frames[annParsePrev[0]-1].split("/"))[-2], annParsePrev[0])
            cv2.imwrite(fileNameTarget + ".jpg", framePrevCropPads)

            #viewer(framePrev)
            #cv2.rectangle(framePrev, (bbx1Prev,bby1Prev), (bbx2Prev,bby2Prev), (0,255,0), 3)
            #viewer(framePrevCropPads)
            invalidAugCounter = 0
            for augIdx in range(k3):
                [bbx1CurrCrop, bby1CurrCrop, bbx2CurrCrop, bby2CurrCrop, startCropCurrY, endCropCurrY, startCropCurrX, endCropCurrX, valid] = goturnAugmentation(augIdx, cxPrev, cyPrev, bbx1Prev, bby1Prev, bbx2Prev, bby2Prev, frameCurr.shape[:2], bbx1Curr, bby1Curr, bbx2Curr, bby2Curr)
                if not valid:
                    invalidAugCounter += 1
                    continue
                frameCurrCropPads = frameCurr[startCropCurrY:endCropCurrY, startCropCurrX:endCropCurrX]
                fileNameSearch = "{}/{}_curr_{}_{}".format(alovExtdataSearching, (frames[annParseCurr[0]-1].split("/"))[-2], annParseCurr[0], augIdx)
                cv2.imwrite(fileNameSearch + ".jpg", frameCurrCropPads)
                annFile = open(fileNameSearch + ".ann", "w")
                annFile.write("{},{},{},{}".format(bbx1CurrCrop, bby1CurrCrop, bbx2CurrCrop, bby2CurrCrop))
                annFile.close()
                alovTrainSet.write("{}.jpg,{}.jpg,{},{},{},{}\n".format(fileNameTarget, fileNameSearch, bbx1CurrCrop, bby1CurrCrop, bbx2CurrCrop, bby2CurrCrop))
                #cv2.rectangle(frameCurrCropPads, (bbx1CurrCrop,bby1CurrCrop), (bbx2CurrCrop,bby2CurrCrop), (0,255,0), 3)
                #cv2.rectangle(frameCurr, (bbx1Curr,bby1Curr), (bbx2Curr,bby2Curr), (0,255,0), 3)
                #viewer(frameCurrCropPads)

            if invalidAugCounter:
                print "For frame {}_{}, not all augmentation succeeded ({} augmentation)".format(frames[annParseCurr[0]-1].split("/")[-2], annParseCurr[0], 10-invalidAugCounter)


alovTrainSet.close()


# Extract ImageNet data
print "Extracting ImageNet data..."

imageNetTrainSet = open("imageNetTrainSet.txt", "w")

images = glob.glob(imageNetData + "/*/" + "/*")
imagesGT = glob.glob(imageNetDataGT + "/*/" + "/*")
images.sort()
imagesGT.sort()

for image in images:
    imageGT = re.sub("\/imageNetData\/", "/imageNetDataGT/", image)
    imageGT = re.sub("JPEG$", "xml", imageGT)

    xmlTree = ET.parse(imageGT)
    root = xmlTree.getroot()

    if len(root.findall('object')) != 1:
        print "{} has no object annotated".format(image)
        continue

    framePrev = cv2.imread(image)

    obj = root.find('object')
    bbElement = obj.find('bndbox')
    [bbx1Prev, bby1Prev, bbx2Prev, bby2Prev] = [int(bbElement.find('xmin').text), int(bbElement.find('ymin').text), int(bbElement.find('xmax').text), int(bbElement.find('ymax').text)]

    cxPrev = bbx1Prev + ((bbx2Prev - bbx1Prev)/2)
    cyPrev = bby1Prev + ((bby2Prev - bby1Prev)/2)
    bbPadsPrevH = k1*((bby2Prev - bby1Prev)/2)
    bbPadsPrevW = k1*((bbx2Prev - bbx1Prev)/2)

    startCropPrevY = 0 if cyPrev-bbPadsPrevH < 0 else cyPrev-bbPadsPrevH
    endCropPrevY = framePrev.shape[0]-1 if cyPrev+bbPadsPrevH > framePrev.shape[0]-1 else cyPrev+bbPadsPrevH
    startCropPrevX = 0 if cxPrev-bbPadsPrevW < 0 else cxPrev-bbPadsPrevW
    endCropPrevX = framePrev.shape[1]-1 if cxPrev+bbPadsPrevW > framePrev.shape[1]-1 else cxPrev+bbPadsPrevW

    framePrevCropPads = framePrev[startCropPrevY:endCropPrevY, startCropPrevX:endCropPrevX]
    fileNameTarget = "{}/{}_prev".format(imageNetExtdataTarget, image.split("/")[-1][:-5])
    cv2.imwrite(fileNameTarget + ".jpg", framePrevCropPads)

    #cv2.rectangle(framePrev, (bbx1Prev,bby1Prev), (bbx2Prev,bby2Prev), (0,255,0), 3)
    #viewer(framePrevCropPads)
    invalidAugCounter = 0
    for augIdx in range(1,k3):
        [bbx1CurrCrop, bby1CurrCrop, bbx2CurrCrop, bby2CurrCrop, startCropCurrY, endCropCurrY, startCropCurrX, endCropCurrX, valid] = goturnAugmentation(augIdx, cxPrev, cyPrev, bbx1Prev, bby1Prev, bbx2Prev, bby2Prev, framePrev.shape[:2], bbx1Prev, bby1Prev, bbx2Prev, bby2Prev)
        if not valid:
            invalidAugCounter += 1
            continue
        frameCurrCropPads = framePrev[startCropCurrY:endCropCurrY, startCropCurrX:endCropCurrX]
        fileNameSearch = "{}/{}_curr_{}".format(imageNetExtdataSearching, image.split("/")[-1][:-5], augIdx)
        cv2.imwrite(fileNameSearch + ".jpg", frameCurrCropPads)
        annFile = open(fileNameSearch + ".ann", "w")
        annFile.write("{},{},{},{}".format(bbx1CurrCrop, bby1CurrCrop, bbx2CurrCrop, bby2CurrCrop))
        annFile.close()
        imageNetTrainSet.write("{}.jpg,{}.jpg,{},{},{},{}\n".format(fileNameTarget, fileNameSearch, bbx1CurrCrop, bby1CurrCrop, bbx2CurrCrop, bby2CurrCrop))
        #cv2.rectangle(frameCurrCropPads, (bbx1CurrCrop,bby1CurrCrop), (bbx2CurrCrop,bby2CurrCrop), (0,255,0), 3)
        #viewer(frameCurrCropPads)

    if invalidAugCounter:
        print "For frame {}, not all augmentation succeeded ({} augmentation)".format(image.split("/")[-1][:-50], 10-invalidAugCounter)

imageNetTrainSet.close()


# Extract vot data
print "Extracting VOT data..."

votTestSet = open("votTestSet.txt", "w")

videos = glob.glob(votData + "/*")
for video in videos:
    if re.search("list\.txt", video):
        print "skip category list file"
        continue
    videoAnn = video + "/groundtruth.txt"
    framesAnn = [line.rstrip() for line in open(videoAnn).readlines()]
    frames = glob.glob(video + "/*.jpg")
    frames.sort()
    if len(framesAnn) < 5:
        print "video {} is shorter then 5 frames".format(video.split("/")[-1])
        continue
    for frameAnnIndx in xrange(4, len(framesAnn), 1):
        annParsePrev = [int(float(number)) for number in framesAnn[frameAnnIndx-4].split(',')]
        annParseCurr = [int(float(number)) for number in framesAnn[frameAnnIndx].split(',')]
        framePrev = cv2.imread(frames[frameAnnIndx-4])
        frameCurr = cv2.imread(frames[frameAnnIndx])
        [bbx1Prev, bby1Prev, bbx2Prev, bby2Prev] = extBBvot(annParsePrev)
        [bbx1Curr, bby1Curr, bbx2Curr, bby2Curr] = extBBvot(annParseCurr)

        cxPrev = bbx1Prev + ((bbx2Prev - bbx1Prev)/2)
        cyPrev = bby1Prev + ((bby2Prev - bby1Prev)/2)
        bbPadsPrevH = k1*((bby2Prev - bby1Prev)/2)
        bbPadsPrevW = k1*((bbx2Prev - bbx1Prev)/2)

        startCropPrevY = 0 if cyPrev-bbPadsPrevH < 0 else cyPrev-bbPadsPrevH
        endCropPrevY = framePrev.shape[0]-1 if cyPrev+bbPadsPrevH > framePrev.shape[0]-1 else cyPrev+bbPadsPrevH
        startCropPrevX = 0 if cxPrev-bbPadsPrevW < 0 else cxPrev-bbPadsPrevW
        endCropPrevX = framePrev.shape[1]-1 if cxPrev+bbPadsPrevW > framePrev.shape[1]-1 else cxPrev+bbPadsPrevW

        framePrevCropPads = framePrev[startCropPrevY:endCropPrevY, startCropPrevX:endCropPrevX]
        fileNameTarget = "{}/{}_prev_{}".format(votExtdataTarget, video.split("/")[-1], frames[frameAnnIndx-5].split("/")[-1][:-4])
        cv2.imwrite(fileNameTarget + ".jpg", framePrevCropPads)

        #viewer(framePrev)
        #cv2.rectangle(framePrev, (bbx1Prev,bby1Prev), (bbx2Prev,bby2Prev), (0,255,0), 3)
        #viewer(framePrevCropPads)
        invalidAugCounter = 0
        for augIdx in range(k3):
            [bbx1CurrCrop, bby1CurrCrop, bbx2CurrCrop, bby2CurrCrop, startCropCurrY, endCropCurrY, startCropCurrX, endCropCurrX, valid] = goturnAugmentation(augIdx, cxPrev, cyPrev, bbx1Prev, bby1Prev, bbx2Prev, bby2Prev, frameCurr.shape[:2], bbx1Curr, bby1Curr, bbx2Curr, bby2Curr)
            if not valid:
                invalidAugCounter += 1
                continue
            frameCurrCropPads = frameCurr[startCropCurrY:endCropCurrY, startCropCurrX:endCropCurrX]
            fileNameSearch = "{}/{}_curr_{}_{}".format(votExtdataSearching, video.split("/")[-1], frames[frameAnnIndx].split("/")[-1][:-4], augIdx)
            cv2.imwrite(fileNameSearch + ".jpg", frameCurrCropPads)
            annFile = open(fileNameSearch + ".ann", "w")
            annFile.write("{},{},{},{}".format(bbx1CurrCrop, bby1CurrCrop, bbx2CurrCrop, bby2CurrCrop))
            annFile.close()
            votTestSet.write("{}.jpg,{}.jpg,{},{},{},{}\n".format(fileNameTarget, fileNameSearch, bbx1CurrCrop, bby1CurrCrop, bbx2CurrCrop, bby2CurrCrop))
            cv2.rectangle(frameCurrCropPads, (bbx1CurrCrop,bby1CurrCrop), (bbx2CurrCrop,bby2CurrCrop), (0,255,0), 3)
            #cv2.rectangle(frameCurr, (bbx1Curr,bby1Curr), (bbx2Curr,bby2Curr), (0,255,0), 3)
            #viewer(frameCurrCropPads)

            if invalidAugCounter:
                print "For frame {}_{}, not all augmentation succeeded ({} augmentation)".format(video.split("/")[-1], frames[frameAnnIndx].split("/")[-1][:-4], 10-invalidAugCounter)


votTestSet.close()