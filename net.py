import tensorflow as tf
import numpy as np
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

        self.target = tf.placeholder(tf.float32, [self.batch_size, 100, 100, 3])
        self.mid = tf.placeholder(tf.float32, [self.batch_size, 400, 400, 3])
        self.search = tf.placeholder(tf.float32, [self.batch_size, 400, 400, 3])
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
            self.conv_output_target_up = resUnit(x, 32, 3, 'targetResUnit3_up')
            x = resUnit(x, 64, 3, 'targetResUnit3')
            self.targetF = x
            x = resUnit(x,128, 3, 'targetResUnit4')
            x = resUnit(x, 64, 3, 'targetResUnit5')
            self.conv_output_target = x
            _activation_summary(self.conv_output_target)

        ########### for mid ###########
        with tf.variable_scope("net_mid"):

            x = self.mid
            x = resUnit(x, 16, 3, 'midResUnit1')
            x = resUnit(x, 32, 3, 'midResUnit2')
            x = resUnit(x, 64, 3, 'midResUnit3')
            self.midF = x
            x = resUnit(x,128, 3, 'midResUnit4')
            self.conv_output_mid_up = resUnit(x, 32, 3, 'midResUnit5_up')
            x = resUnit(x,256, 3, 'midResUnit5')
            x = resUnit(x,128, 3, 'midResUnit6')
            x = resUnit(x, 64, 3, 'midResUnit7')
            self.conv_output_mid = x
            _activation_summary(self.conv_output_mid)

        ########### for search ###########
        with tf.variable_scope("net_search"):

            x = self.search
            x = resUnit(x, 16, 3, 'searchResUnit1')
            x = resUnit(x, 32, 3, 'searchResUnit2')
            x = resUnit(x, 64, 3, 'searchResUnit3')
            self.searchF = x
            x = resUnit(x,128, 3, 'searchResUnit4')
            self.conv_output_search_up = resUnit(x, 32, 3, 'searchResUnit5_up')
            x = resUnit(x,256, 3, 'searchResUnit5')
            x = resUnit(x,128, 3, 'searchResUnit6')
            x = resUnit(x, 64, 3, 'searchResUnit7')
            self.conv_output_search = x
            _activation_summary(self.conv_output_search)

        ########### Concatnate all nets ###########
        ########### fully connencted layers ###########
        with tf.variable_scope("fc_nets"):

            # now three features maps, each 3 x 3 x 128 + three upper features maps, each 13 x 13 x 64
            concatLow = tf.concat([self.conv_output_target, self.conv_output_mid, self.conv_output_search], axis = 3)
            concatUp = tf.concat([self.conv_output_target_up, self.conv_output_mid_up, self.conv_output_search_up], axis = 3)

            flatLow = tf.layers.flatten(concatLow)
            flatUp = tf.layers.flatten(concatUp)
            x = tf.concat([flatLow, flatUp], axis = -1)

            x = tf.layers.dense(x, 1024, name='fc1', activation=tf.nn.elu, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
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

            ## Scaling all bounding boxes for the PM loss - target BBox is 1/3 of the patch size --> make it 1/5 of the patch size, thus 1/3 to 2/5 is 1.2, and 2/3 to 3/5 is 0.9
            #t = tf.ones([self.batch_size, 2], dtype=tf.bool)
            #f = tf.zeros([self.batch_size, 2], dtype=tf.bool)
            #boxScaleMask = tf.concat([t, f], axis=1)
            #fc_output_mid_scale = tf.where(boxScaleMask , fc_output_mid*1.2, fc_output_mid*0.9)
            #fc_output_search_scale = tf.where(boxScaleMask , fc_output_search*1.2, fc_output_search*0.9)
            #bboxGT_scale = tf.where(boxScaleMask , bboxGT*1.2, bboxGT*0.9)
            fc_output_mid_scale = fc_output_mid
            fc_output_search_scale = fc_output_search
            bboxGT_scale = bboxGT

            badBBmid = 0
            badBBsearch = 0
            badBBsearchGT = 0 # Sanity

            for imType in ['image']:#, ' feature']:

                if imType == 'image':
                    target = self.target
                    mid = self.mid
                    search = self.search
                else:
                    target = self.targetF
                    mid = self.midF
                    search = self.searchF

                ## Extract the target object
                targetPM = target

                def f1(batchIdx, img): return tf.abs(img[batchIdx, img.shape[1]*3/8:img.shape[1]*5/8, img.shape[2]*3/8:img.shape[2]*5/8, :]-256)
                def f2(img): return tf.image.resize_images(img, [int(targetPM.shape[1]),int(targetPM.shape[2])])

                # Extract mid-target object prediction for the PM loss
                midF = mid
                midPM_target = midF[0,tf.cast(int(midF.shape[1])*fc_output_mid_scale[0,1], tf.int32):tf.cast(int(midF.shape[1])*fc_output_mid_scale[0,3], tf.int32),
                        tf.cast(int(midF.shape[2])*fc_output_mid_scale[0,0], tf.int32):tf.cast(int(midF.shape[2])*fc_output_mid_scale[0,2], tf.int32), :]
                if imType == 'image':
                    badBBmid = tf.cond(tf.equal(tf.size(midPM_target), 0), lambda: tf.add(badBBmid, 1), lambda: badBBmid, name="badBBmid")
                midPM_target = tf.cond(tf.equal(tf.size(midPM_target), 0), lambda: f1(0, midF), lambda: f2(midPM_target))
                midPM_target = tf.expand_dims(midPM_target, 0)
                for i in range(1, self.batch_size):
                    midF_cropNscale = midF[i,tf.cast(int(midF.shape[1])*fc_output_mid_scale[i,1], tf.int32):tf.cast(int(midF.shape[1])*fc_output_mid_scale[i,3], tf.int32),
                            tf.cast(int(midF.shape[2])*fc_output_mid_scale[i,0], tf.int32):tf.cast(int(midF.shape[2])*fc_output_mid_scale[i,2], tf.int32), :]
                    if imType == 'image':
                        badBBmid = tf.cond(tf.equal(tf.size(midF_cropNscale), 0), lambda: tf.add(badBBmid, 1), lambda: badBBmid, name="badBBmid")
                    midF_cropNscale = tf.cond(tf.equal(tf.size(midF_cropNscale), 0), lambda: f1(i, midF), lambda: f2(midF_cropNscale))
                    midF_cropNscale = tf.expand_dims(midF_cropNscale, 0)
                    midPM_target = tf.concat([midPM_target,midF_cropNscale], axis=0)

                # Extract search-target object prediction for the PM loss
                searchF = search
                searchPM_target = searchF[0,tf.cast(int(searchF.shape[1])*fc_output_search_scale[0,1], tf.int32):tf.cast(int(searchF.shape[1])*fc_output_search_scale[0,3], tf.int32),
                        tf.cast(int(searchF.shape[2])*fc_output_search_scale[0,0], tf.int32):tf.cast(int(searchF.shape[2])*fc_output_search_scale[0,2], tf.int32), :]
                if imType == 'image':
                    badBBsearch = tf.cond(tf.equal(tf.size(searchPM_target), 0), lambda: tf.add(badBBsearch, 1), lambda: badBBsearch, name="badBBsearch")
                searchPM_target = tf.cond(tf.equal(tf.size(searchPM_target), 0), lambda: f1(0, searchF), lambda: f2(searchPM_target))
                searchPM_target = tf.expand_dims(searchPM_target, 0)
                for i in range(1, self.batch_size):
                    searchF_cropNscale = searchF[i,tf.cast(int(searchF.shape[1])*fc_output_search_scale[i,1], tf.int32):tf.cast(int(searchF.shape[1])*fc_output_search_scale[i,3], tf.int32),
                            tf.cast(int(searchF.shape[2])*fc_output_search_scale[i,0], tf.int32):tf.cast(int(searchF.shape[2])*fc_output_search_scale[i,2], tf.int32), :]
                    if imType == 'image':
                        badBBsearch = tf.cond(tf.equal(tf.size(searchF_cropNscale), 0), lambda: tf.add(badBBsearch, 1), lambda: badBBsearch, name="badBBsearch")
                    searchF_cropNscale = tf.cond(tf.equal(tf.size(searchF_cropNscale), 0), lambda: f1(i, searchF), lambda: f2(searchF_cropNscale))
                    searchF_cropNscale = tf.expand_dims(searchF_cropNscale, 0)
                    searchPM_target = tf.concat([searchPM_target, searchF_cropNscale], axis=0)

                # Extract search object GT for the PM loss
                searchPM = searchF[0,tf.cast(int(searchF.shape[1])*bboxGT_scale[0,1], tf.int32):tf.cast(int(searchF.shape[1])*bboxGT_scale[0,3], tf.int32),
                        tf.cast(int(searchF.shape[2])*bboxGT_scale[0,0], tf.int32):tf.cast(int(searchF.shape[2])*bboxGT_scale[0,2], tf.int32), :]
                if imType == 'image':
                    badBBsearchGT = tf.cond(tf.equal(tf.size(searchPM), 0), lambda: tf.add(badBBsearchGT, 1), lambda: badBBsearchGT, name="badBBsearchGT")
                searchPM = tf.cond(tf.equal(tf.size(searchPM), 0), lambda: f1(0, searchF), lambda: f2(searchPM))
                searchPM = tf.expand_dims(searchPM, 0)
                for i in range(1, self.batch_size):
                    searchF_cropNscale = searchF[i,tf.cast(int(searchF.shape[1])*bboxGT_scale[i,1], tf.int32):tf.cast(int(searchF.shape[1])*bboxGT_scale[i,3], tf.int32),
                            tf.cast(int(searchF.shape[2])*bboxGT_scale[i,0], tf.int32):tf.cast(int(searchF.shape[2])*bboxGT_scale[i,2], tf.int32), :]
                    if imType == 'image':
                        badBBsearchGT = tf.cond(tf.equal(tf.size(searchF_cropNscale), 0), lambda: tf.add(badBBsearchGT, 1), lambda: badBBsearchGT, name="badBBsearchGT")
                    searchF_cropNscale = tf.cond(tf.equal(tf.size(searchF_cropNscale), 0), lambda: f1(i, searchF), lambda: f2(searchF_cropNscale))
                    searchF_cropNscale = tf.expand_dims(searchF_cropNscale, 0)
                    searchPM = tf.concat([searchPM, searchF_cropNscale], axis=0)

                # Extract mid-search object prediction for the PM loss
                midPM_search = midF[0,tf.cast(int(midF.shape[1])*fc_output_mid_scale[0,1], tf.int32):tf.cast(int(midF.shape[1])*fc_output_mid_scale[0,3], tf.int32),
                        tf.cast(int(midF.shape[2])*fc_output_mid_scale[0,0], tf.int32):tf.cast(int(midF.shape[2])*fc_output_mid_scale[0,2], tf.int32), :]
                midPM_search = tf.cond(tf.equal(tf.size(midPM_search), 0), lambda: f1(0, midF), lambda: f2(midPM_search))
                midPM_search = tf.expand_dims(midPM_search, 0)
                for i in range(1, self.batch_size):
                    midF_cropNscale = midF[i,tf.cast(int(midF.shape[1])*fc_output_mid_scale[i,1], tf.int32):tf.cast(int(midF.shape[1])*fc_output_mid_scale[i,3], tf.int32),
                            tf.cast(int(midF.shape[2])*fc_output_mid_scale[i,0], tf.int32):tf.cast(int(midF.shape[2])*fc_output_mid_scale[i,2], tf.int32), :]
                    midF_cropNscale = tf.cond(tf.equal(tf.size(midF_cropNscale), 0), lambda: f1(i, midF), lambda: f2(midF_cropNscale))
                    midF_cropNscale = tf.expand_dims(midF_cropNscale, 0)
                    midPM_search = tf.concat([midPM_search,midF_cropNscale], axis=0)

                if imType == 'image':
                    _variable_summaries(badBBmid)
                    _variable_summaries(badBBsearch)
                    _variable_summaries(badBBsearchGT)
                    self.checks = {'badBBmid': badBBmid, 'badBBsearch': badBBsearch, 'badBBsearchGT': badBBsearchGT}

                ## Calculate NCC photometric losses
                meanTarget, varTarget = tf.nn.moments(targetPM, axes=[1,2,3])
                meanTarget = tf.reshape(meanTarget, [-1,1,1,1])
                stdTarget = tf.reshape(tf.sqrt(varTarget), [-1,1,1,1])

                meanMid_target, varMid_target = tf.nn.moments(midPM_target, axes=[1,2,3])
                meanMid_target = tf.reshape(meanMid_target, [-1,1,1,1])
                stdMid_target = tf.reshape(tf.sqrt(varMid_target), [-1,1,1,1])

                meanMid_search, varMid_search = tf.nn.moments(midPM_search, axes=[1,2,3])
                meanMid_search = tf.reshape(meanMid_search, [-1,1,1,1])
                stdMid_search = tf.reshape(tf.sqrt(varMid_search), [-1,1,1,1])

                meanSearch_target, varSearch_target = tf.nn.moments(searchPM_target, axes=[1,2,3])
                meanSearch_target = tf.reshape(meanSearch_target, [-1,1,1,1])
                stdSearch_target = tf.reshape(tf.sqrt(varSearch_target), [-1,1,1,1])

                meanSearch, varSearch = tf.nn.moments(searchPM, axes=[1,2,3])
                meanSearch = tf.reshape(meanSearch, [-1,1,1,1])
                stdSearch = tf.reshape(tf.sqrt(varSearch), [-1,1,1,1])

                if imType == 'image':

                    pmLossTargetSearchBound = (1 - tf.reduce_mean(((targetPM-meanTarget)/stdTarget)*((searchPM-meanSearch)/stdSearch))) / 2
                    self.pmLossTargetSearchBound = tf.where(tf.is_nan(pmLossTargetSearchBound), 0., pmLossTargetSearchBound, name="pmLossTargetSearchBound")
                    _variable_summaries(self.pmLossTargetSearchBound)

                    pmLossTargetMid = ((1 - tf.reduce_mean(((targetPM-meanTarget)/stdTarget)*((midPM_target-meanMid_target)/stdMid_target))) / 2) - pmLossTargetSearchBound
                    pmLossTargetMid = tf.maximum(0., pmLossTargetMid)
                    self.pmLossTargetMid = tf.where(tf.is_nan(pmLossTargetMid), 0., pmLossTargetMid, name="pmNccLossTargetMid")
                    _variable_summaries(self.pmLossTargetMid)

                    pmLossMidSearch = ((1 - tf.reduce_mean(((searchPM-meanSearch)/stdSearch)*((midPM_search-meanMid_search)/stdMid_search))) / 2) - pmLossTargetSearchBound
                    pmLossMidSearch = tf.maximum(0., pmLossMidSearch)
                    self.pmLossMidSearch = tf.where(tf.is_nan(pmLossMidSearch), 0., pmLossMidSearch, name="pmNccLossMidSearch")
                    _variable_summaries(self.pmLossMidSearch)

                    #pmLossTargetSearch = ((1 - tf.reduce_mean(((targetPM-meanTarget)/stdTarget)*((searchPM_target-meanSearch_target)/stdSearch_target))) / 2) - pmLossTargetSearchBound
                    #pmLossTargetSearch = tf.maximum(0., pmLossTargetSearch)
                    #self.pmLossTargetSearch = tf.where(tf.is_nan(pmLossTargetSearch), 0., pmLossTargetSearch, name="pmNccLossTargetSearch")
                    #_variable_summaries(self.pmLossTargetSearch)
                else:
                    pmLossTargetSearchBoundF = (1 - tf.reduce_mean(((targetPM-meanTarget)/stdTarget)*((searchPM-meanSearch)/stdSearch))) / 2
                    self.pmLossTargetSearchBoundF = tf.where(tf.is_nan(pmLossTargetSearchBoundF), 0., pmLossTargetSearchBoundF, name="pmLossTargetSearchBoundF")
                    _variable_summaries(self.pmLossTargetSearchBoundF)

                    pmLossTargetMidF = ((1 - tf.reduce_mean(((targetPM-meanTarget)/stdTarget)*((midPM_target-meanMid_target)/stdMid_target))) / 2)
                    pmLossTargetMidF = tf.maximum(0., pmLossTargetMidF)
                    self.pmLossTargetMidF = tf.where(tf.is_nan(pmLossTargetMidF), 0., pmLossTargetMidF, name="pmNccLossTargetMidF")
                    _variable_summaries(self.pmLossTargetMidF)

                    pmLossMidSearchF = ((1 - tf.reduce_mean(((searchPM-meanSearch)/stdSearch)*((midPM_search-meanMid_search)/stdMid_search))) / 2)
                    pmLossMidSearchF = tf.maximum(0., pmLossMidSearchF)
                    self.pmLossMidSearchF = tf.where(tf.is_nan(pmLossMidSearchF), 0., pmLossMidSearchF, name="pmNccLossMidSearchF")
                    _variable_summaries(self.pmLossMidSearchF)

                    pmLossTargetSearchF = ((1 - tf.reduce_mean(((targetPM-meanTarget)/stdTarget)*((searchPM_target-meanSearch_target)/stdSearch_target))) / 2) - pmLossTargetSearchBoundF
                    pmLossTargetSearchF = tf.maximum(0., pmLossTargetSearchF)
                    self.pmLossTargetSearchF = tf.where(tf.is_nan(pmLossTargetSearchF), 0., pmLossTargetSearchF, name="pmNccLossTargetSearchF")
                    _variable_summaries(self.pmLossTargetSearchF)


            ## Calculate mid predicted BB distance from search GT BB and penalize if above threshold - half the image size
            midBBDistFromGT = tf.subtract(bboxGT, fc_output_mid)
            midBBDistFromGT = tf.abs(midBBDistFromGT)
            midBBDistFromGT = tf.where(tf.greater(midBBDistFromGT, 0.4), midBBDistFromGT, tf.zeros(midBBDistFromGT.shape, dtype=tf.float32))
            midBBDistFromGT = tf.reduce_sum(midBBDistFromGT, axis=1)
            self.midBBoxLoss = tf.reduce_mean(midBBDistFromGT, name="midBBoxLoss")
            _variable_summaries(self.midBBoxLoss)

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

            ## Calculate bounding box regression loss
            bboxDist = tf.subtract(bboxGT, fc_output_search)
            bboxDist = tf.abs(bboxDist)
            bboxDist = tf.reduce_sum(bboxDist, axis=1)
            self.bboxLoss = tf.reduce_mean(bboxDist, name="bboxLoss")
            _variable_summaries(self.bboxLoss)


            self.loss = self.bboxLoss + self.midBBoxLoss + self.diffLoss + self.pmLossTargetMid + self.pmLossMidSearch

            self.losses = {
                    'bboxLoss': self.bboxLoss,
                    'midBBoxLoss': self.midBBoxLoss,
                    'diffLoss': self.diffLoss,
                    'pmLossTargetMid': self.pmLossTargetMid,
                    'pmLossMidSearch': self.pmLossMidSearch
                    #'pmLossTargetSearch': self.pmLossTargetSearch,
                    #'pmLossTargetMidF': self.pmLossTargetMidF,
                    #'pmLossMidSearchF': self.pmLossMidSearchF,
                    #'pmLossTargetSearchF': self.pmLossTargetSearchF
            }




