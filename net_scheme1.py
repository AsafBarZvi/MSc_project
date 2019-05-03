import tensorflow as tf
import numpy as np
import cv2
from threading import Thread
#import ipdb


#-------------------------------------------------------------------------------
def smooth_l1_loss(x):
    square_loss   = 0.5*x**2
    absolute_loss = tf.abs(x)
    return tf.where(tf.less(absolute_loss, 1.), square_loss, absolute_loss-0.5)

def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        #logging.debug("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            tf.summary.scalar(name, var)
            #mean = tf.reduce_mean(var)
            #tf.summary.scalar(name + '/mean', mean)
            #with tf.name_scope('stddev'):
            #    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            #tf.summary.scalar(name + '/sttdev', stddev)
            #tf.summary.scalar(name + '/max', tf.reduce_max(var))
            #tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)



#-------------------------------------------------------------------------------
class TRACKNET:

    def __init__(self, batch_size, wd = 0.000005, bd = 0.0000005, train = True):

        self.batch_size = batch_size
        self.wreg = tf.contrib.layers.l2_regularizer(wd)
        self.breg = tf.contrib.layers.l2_regularizer(bd)
        self.train = train

        self.build()


    #-----------------------------------------------------------------------
    # Building the net
    #-----------------------------------------------------------------------
    def build(self):

        self.target = tf.placeholder(tf.float32, [self.batch_size, 227, 227, 3])
        self.mid = tf.placeholder(tf.float32, [self.batch_size, 227, 227, 3])
        self.search = tf.placeholder(tf.float32, [self.batch_size, 227, 227, 3])
        self.bboxMid= tf.placeholder(tf.float32, [self.batch_size, 5])
        self.bbox= tf.placeholder(tf.float32, [self.batch_size, 4])

        def resUnit(inData, outChannel, kerSize, layerName):
            info = tf.layers.conv2d(inData, outChannel, kerSize, 1, name=layerName+'_info', reuse=False, activation=tf.nn.elu, padding='same', kernel_regularizer=self.wreg, bias_regularizer=self.breg)
            _activation_summary(info)
            strid = tf.layers.conv2d(info, outChannel*2, kerSize, 2, name=layerName+'_strid', reuse=False, activation=tf.nn.elu, padding='same', kernel_regularizer=self.wreg, bias_regularizer=self.breg)
            _activation_summary(strid)
            redu = tf.layers.conv2d(strid, outChannel, 1, 1, name=layerName+'_redu', reuse=False, activation=tf.nn.elu, padding='same', kernel_regularizer=self.wreg, bias_regularizer=self.breg)
            _activation_summary(redu)
            if 'ResUnit1' in layerName:
                info = tf.image.resize_images(info, redu.shape[1:3])
                outData = tf.concat([redu, info], axis=-1)
            else:
                inData = tf.image.resize_images(inData, redu.shape[1:3])
                inData = tf.layers.conv2d(inData, outChannel, 1, 1, name=layerName+'_skip_redu', reuse=False, activation=tf.nn.elu, padding='same', kernel_regularizer=self.wreg, bias_regularizer=self.breg)
                outData = tf.concat([redu, inData], axis=-1)
            return outData


        ########### for target ###########
        with tf.variable_scope("net_target"):

            x = self.target
            x = resUnit(x, 16, 3, 'targetResUnit1')
            x = resUnit(x, 32, 3, 'targetResUnit2')
            x = resUnit(x, 64, 3, 'targetResUnit3')
            x = resUnit(x,128, 3, 'targetResUnit4')
            self.targetF = x
            self.conv_output_target_up = resUnit(x, 32, 3, 'targetResUnit5_up')
            x = resUnit(x,256, 3, 'targetResUnit5')
            x = resUnit(x, 64, 3, 'targetResUnit6')
            self.conv_output_target = x
            _activation_summary(self.conv_output_target)

        ########### for mid ###########
        with tf.variable_scope("net_mid"):

            x = self.mid
            x = resUnit(x, 16, 3, 'midResUnit1')
            x = resUnit(x, 32, 3, 'midResUnit2')
            x = resUnit(x, 64, 3, 'midResUnit3')
            x = resUnit(x,128, 3, 'midResUnit4')
            self.midF = x
            self.conv_output_mid_up = resUnit(x, 32, 3, 'midResUnit5_up')
            x = resUnit(x,256, 3, 'midResUnit5')
            x = resUnit(x, 64, 3, 'midResUnit6')
            self.conv_output_mid = x
            _activation_summary(self.conv_output_mid)

        ########### for search ###########
        with tf.variable_scope("net_search"):

            x = self.search
            x = resUnit(x, 16, 3, 'searchResUnit1')
            x = resUnit(x, 32, 3, 'searchResUnit2')
            x = resUnit(x, 64, 3, 'searchResUnit3')
            x = resUnit(x,128, 3, 'searchResUnit4')
            self.searchF = x
            self.conv_output_search_up = resUnit(x, 32, 3, 'searchResUnit5_up')
            x = resUnit(x,256, 3, 'searchResUnit5')
            x = resUnit(x, 64, 3, 'searchResUnit6')
            self.conv_output_search = x
            _activation_summary(self.conv_output_search)

        ########### Concatnate all nets ###########
        ########### fully connencted layers ###########
        with tf.variable_scope("fc_nets"):

            # now three features maps, each 3 x 3 x 128 + three upper features maps, each 6 x 6 x 64
            concatLow = tf.concat([self.conv_output_target, self.conv_output_mid, self.conv_output_search], axis = 3)
            concatUp = tf.concat([self.conv_output_target_up, self.conv_output_mid_up, self.conv_output_search_up], axis = 3)

            flatLow = tf.layers.flatten(concatLow)
            flatUp = tf.layers.flatten(concatUp)
            x = tf.concat([flatLow, flatUp], axis = -1)

            x = tf.layers.dense(x, 4096, name='fc1', activation=tf.nn.elu, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.dense(x, 2048, name='fc2', activation=tf.nn.elu, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            self.fc_output = tf.layers.dense(x, 8, name='fc_out', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)

            self.result = {
                    'bbox_mid': self.fc_output[:,:4],
                    'bbox_search': self.fc_output[:,4:]
            }


        #-----------------------------------------------------------------------
        # Compute loss
        #-----------------------------------------------------------------------
        with tf.variable_scope("loss"):

            fc_output_mid = self.fc_output[:,:4]
            fc_output_search = self.fc_output[:,4:]
            bboxGT = self.bbox
            pmMidBB = self.bboxMid

            ## Calculate mid predicted BB distance from search GT BB and penalize if above threshold - 40% the image size
            midBBDistFromGT = tf.subtract(bboxGT, fc_output_mid)
            midBBDistFromGT = tf.abs(midBBDistFromGT)
            midBBDistFromGT = tf.where(tf.greater(midBBDistFromGT, 0.4), midBBDistFromGT, tf.zeros(midBBDistFromGT.shape, dtype=tf.float32))
            midBBDistFromGT = tf.reduce_sum(midBBDistFromGT, axis=1)
            self.midBBoxGTLoss = tf.reduce_mean(midBBDistFromGT, name="midBBoxGTLoss")
            _variable_summaries(self.midBBoxGTLoss)

            ## Bound the predicted box dimensions
            widthBoxMid = tf.abs(fc_output_mid[:,2] - fc_output_mid[:,0])
            heightBoxMid = tf.abs(fc_output_mid[:,3] - fc_output_mid[:,1])
            searchWidth = tf.abs(bboxGT[:,2] - bboxGT[:,0])
            searchHeight = tf.abs(bboxGT[:,3] - bboxGT[:,1])
            widthDiff = tf.abs(widthBoxMid - searchWidth)
            heightDiff = tf.abs(heightBoxMid - searchHeight)
            widthDiffLoss = tf.where(tf.greater(widthDiff, 0.15), widthDiff, tf.zeros(widthDiff.shape, dtype=tf.float32))
            heightDiffLoss = tf.where(tf.greater(heightDiff, 0.15), heightDiff, tf.zeros(heightDiff.shape, dtype=tf.float32))
            diffLoss = tf.add(widthDiffLoss, heightDiffLoss)
            self.diffLoss = tf.reduce_mean(diffLoss, name="diffLoss")
            _variable_summaries(self.diffLoss)

            ## Calculate bounding box regression loss for the mid patch
            # First, filter the invalid bounding boxes
            bbCordSum = tf.reduce_sum(pmMidBB[:,:4], axis=1)
            validMask = tf.tile(tf.reshape(tf.logical_and(tf.greater(bbCordSum, 0.01), tf.greater(pmMidBB[:,4],1.5)), [self.batch_size,1]), [1,4])
            pmMidBB = tf.reshape(tf.boolean_mask(pmMidBB[:,:4], validMask), [-1,4])
            fc_output_mid = tf.reshape(tf.boolean_mask(fc_output_mid, validMask), [-1,4])
            self.checks = pmMidBB
            # Now, calculate the L1 loss
            bboxDistMid = tf.subtract(pmMidBB, fc_output_mid)
            bboxDistMid = tf.abs(bboxDistMid)
            bboxDistMid = tf.reduce_sum(bboxDistMid, axis=1)
            bboxLossMid = tf.reduce_mean(bboxDistMid, name="bboxLossMid")
            self.bboxLossMid = tf.where(tf.is_nan(bboxLossMid), 0., bboxLossMid, name="bboxLossMid")
            _variable_summaries(self.bboxLossMid)

            ## Calculate bounding box regression loss
            bboxDist = tf.subtract(bboxGT, fc_output_search)
            bboxDist = tf.abs(bboxDist)
            bboxDist = tf.reduce_sum(bboxDist, axis=1)
            self.bboxLoss = tf.reduce_mean(bboxDist, name="bboxLoss")
            _variable_summaries(self.bboxLoss)

            self.loss = self.bboxLoss + self.bboxLossMid + 2*self.midBBoxGTLoss + 2*self.diffLoss

            self.losses = {
                    'bboxLoss': self.bboxLoss,
                    'bboxLossMid': self.bboxLossMid,
                    'midBBoxLoss': self.midBBoxGTLoss,
                    'diffLoss': self.diffLoss
            }


    #-------------------------------------------------------------------------------

    #-----------------------------------------------------------------------
    # Python function to compute the best mid bounding box using PM match
    #-----------------------------------------------------------------------
    def pmMatchMidBB(self, inData):

        #def findBestMatch(i, searchBBoxGTScaled, mid, search, targetPM, meanTarget, stdTarget, nccMax, pmMidBB, countBadMidBB, countBadSearchBB):

        #    ## Extract search object based on GT bounding box, for the PM match
        #    searchPM = search[i, int(search.shape[1]*searchBBoxGTScaled[i,1]):int(search.shape[1]*searchBBoxGTScaled[i,3]),
        #                         int(search.shape[2]*searchBBoxGTScaled[i,0]):int(search.shape[2]*searchBBoxGTScaled[i,2]), :]
        #    if searchPM.size == 0:
        #        countBadSearchBB += 1
        #        return

        #    ## Calculate search NCC parameters for the photometric match
        #    meanSearch = np.mean(searchPM)
        #    stdSearch = np.sqrt(np.var(searchPM))

        #    ## Run over the mid patch to find the bounding box which yield the maximum NCC similarity
        #    # Initial guess - the mid BB prediction
        #    yMin = int(mid.shape[1]*searchBBoxGTScaled[i,1])
        #    yMax = int(mid.shape[1]*searchBBoxGTScaled[i,3])
        #    xMin = int(mid.shape[2]*searchBBoxGTScaled[i,0])
        #    xMax = int(mid.shape[2]*searchBBoxGTScaled[i,2])

        #    firstCheck = True
        #    for xShift in range(-10, 10, 2):
        #        for yShift in range(-10, 10, 2):
        #            for xVar in range(-2, 2, 2):
        #                for yVar in range(-2, 2, 2):

        #                    # Extract mid object prediction for the PM loss
        #                    midPM = mid[i, yMin+yShift+yVar:yMax+yShift-yVar, xMin+xShift+xVar:xMax+xShift-xVar, :]
        #                    if midPM.size == 0:
        #                        countBadMidBB += 1
        #                        return
        #                    midTargetPM = cv2.resize(midPM, (targetPM.shape[2], targetPM.shape[1]))
        #                    midSearchPM = cv2.resize(midPM, (searchPM.shape[1], searchPM.shape[0]))

        #                    ## Calculate mid NCC parameters for the photometric match
        #                    meanMidTargetPM = np.mean(midTargetPM)
        #                    stdMidTargetPM = np.sqrt(np.var(midTargetPM))
        #                    meanMidSearchPM = np.mean(midSearchPM)
        #                    stdMidSearchPM = np.sqrt(np.var(midSearchPM))

        #                    try:
        #                        nccTargetMid = np.mean(((targetPM[i]-meanTarget[i])/stdTarget[i])*((midTargetPM-meanMidTargetPM)/stdMidTargetPM))
        #                        nccMidSearch = np.mean(((searchPM-meanSearch)/stdSearch)*((midSearchPM-meanMidSearchPM)/stdMidSearchPM))
        #                    except:
        #                        print "Couldn't calculate NCC..."
        #                        return
        #
        #                    if firstCheck:
        #                        nccMax[i] = nccTargetMid + nccMidSearch
        #                        pmMidBB[i] = [((xMin+xShift+xVar)/float(mid.shape[2]))*(1/1.2), ((yMin+yShift+yVar)/float(mid.shape[1]))*(1/1.2),
        #                                      ((xMax+xShift-xVar)/float(mid.shape[2]))*(1/0.9), ((yMax+yShift-yVar)/float(mid.shape[1]))*(1/0.9)]
        #                    else:
        #                        if nccMax[i] < nccTargetMid + nccMidSearch:
        #                            nccMax[i] = nccTargetMid + nccMidSearch
        #                            pmMidBB[i] = [((xMin+xShift+xVar)/float(mid.shape[2]))*(1/1.2), ((yMin+yShift+yVar)/float(mid.shape[1]))*(1/1.2),
        #                                          ((xMax+xShift-xVar)/float(mid.shape[2]))*(1/0.9), ((yMax+yShift-yVar)/float(mid.shape[1]))*(1/0.9)]

        #-------------------------------------------------------------------------------
        ## Scaling all bounding boxes for the PM loss - target BBox is 1/3 of the patch size --> make it 1/5 of the patch size, thus 1/3 to 2/5 is 1.2, and 2/3 to 3/5 is 0.9
        searchBBoxGT = inData[3]
        searchBBoxGTScaled = searchBBoxGT*[1.2, 1.2, 0.9, 0.9]

        target = inData[0]
        mid = inData[1]
        search = inData[2]

        ## Extract target object for the PM match
        targetPM = target
        targetPM = targetPM[:, targetPM.shape[1]*2/5:targetPM.shape[1]*3/5, targetPM.shape[2]*2/5:targetPM.shape[2]*3/5, :]

        ## Calculate target NCC parameters for the photometric match
        meanTarget = np.mean(targetPM, axis=(1,2,3))
        stdTarget = np.sqrt(np.var(targetPM, axis=(1,2,3)))

        ## Run over the batch and match the most similar mid bounding box to the target and search
        nccMax = -2*np.ones((self.batch_size), dtype=np.float32)
        pmMidBB = np.zeros((self.batch_size, 4), dtype=np.float32)
        countBadSearchBB = 0
        countBadMidBB = 0

        #threads = [None] * self.batch_size

        #for i in range(len(threads)):
        #    threads[i] = Thread(target=findBestMatch, args=(i, searchBBoxGTScaled, mid, search, targetPM, meanTarget, stdTarget, nccMax, pmMidBB, countBadMidBB, countBadSearchBB))
        #    threads[i].start()

        #for i in range(len(threads)):
        #    threads[i].join()

        for i in range(self.batch_size):
            #print "Run on sample number: {}".format(i)

            ## Extract search object based on GT bounding box, for the PM match
            searchPM = search[i, int(search.shape[1]*searchBBoxGTScaled[i,1]):int(search.shape[1]*searchBBoxGTScaled[i,3]),
                                 int(search.shape[2]*searchBBoxGTScaled[i,0]):int(search.shape[2]*searchBBoxGTScaled[i,2]), :]
            if searchPM.size == 0:
                countBadSearchBB += 1
                continue
            #searchPM = cv2.resize(searchPM, (targetPM.shape[2], targetPM.shape[1]))

            ## Calculate search NCC parameters for the photometric match
            meanSearch = np.mean(searchPM)
            stdSearch = np.sqrt(np.var(searchPM))

            ## Run over the mid patch to find the bounding box which yield the maximum NCC similarity
            # Initial guess - the mid BB prediction
            yMin = int(mid.shape[1]*searchBBoxGTScaled[i,1])
            yMax = int(mid.shape[1]*searchBBoxGTScaled[i,3])
            xMin = int(mid.shape[2]*searchBBoxGTScaled[i,0])
            xMax = int(mid.shape[2]*searchBBoxGTScaled[i,2])

            firstCheck = True
            foundGoodMatch = False
            for xShift in range(-30, 30, 6):
                for yShift in range(-30, 30, 6):
                    for xVar in range(-2, 2, 2):
                        for yVar in range(-2, 2, 2):

                            if foundGoodMatch:
                                continue

                            # Extract mid object prediction for the PM loss
                            midPM = mid[i, yMin+yShift+yVar:yMax+yShift-yVar, xMin+xShift+xVar:xMax+xShift-xVar, :]
                            if midPM.size == 0:
                                countBadMidBB += 1
                                continue
                            midTargetPM = cv2.resize(midPM, (targetPM.shape[2], targetPM.shape[1]))
                            midSearchPM = cv2.resize(midPM, (searchPM.shape[1], searchPM.shape[0]))

                            ## Calculate mid NCC parameters for the photometric match
                            meanMidTargetPM = np.mean(midTargetPM)
                            stdMidTargetPM = np.sqrt(np.var(midTargetPM))
                            meanMidSearchPM = np.mean(midSearchPM)
                            stdMidSearchPM = np.sqrt(np.var(midSearchPM))

                            try:
                                nccTargetMid = np.mean(((targetPM[i]-meanTarget[i])/stdTarget[i])*((midTargetPM-meanMidTargetPM)/stdMidTargetPM))
                                nccMidSearch = np.mean(((searchPM-meanSearch)/stdSearch)*((midSearchPM-meanMidSearchPM)/stdMidSearchPM))
                            except:
                                print "Couldn't calculate NCC..."
                                continue

                            if firstCheck:
                                nccMax[i] = nccTargetMid + nccMidSearch
                                pmMidBB[i] = [((xMin+xShift+xVar)/float(mid.shape[2]))*(1/1.2), ((yMin+yShift+yVar)/float(mid.shape[1]))*(1/1.2),
                                              ((xMax+xShift-xVar)/float(mid.shape[2]))*(1/0.9), ((yMax+yShift-yVar)/float(mid.shape[1]))*(1/0.9)]
                            else:
                                if nccMax[i] < nccTargetMid + nccMidSearch:
                                    nccMax[i] = nccTargetMid + nccMidSearch
                                    pmMidBB[i] = [((xMin+xShift+xVar)/float(mid.shape[2]))*(1/1.2), ((yMin+yShift+yVar)/float(mid.shape[1]))*(1/1.2),
                                                  ((xMax+xShift-xVar)/float(mid.shape[2]))*(1/0.9), ((yMax+yShift-yVar)/float(mid.shape[1]))*(1/0.9)]

                            if nccMax[i] > 1.6 or nccTargetMid > 0.9 or nccMidSearch > 0.9:
                                foundGoodMatch = True



        ## return the final results
        return nccMax, pmMidBB, countBadSearchBB, countBadMidBB





