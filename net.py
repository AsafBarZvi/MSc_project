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

        self.target = tf.placeholder(tf.float32, [self.batch_size, 227, 227, 3])
        self.mid = tf.placeholder(tf.float32, [self.batch_size, 227, 227, 3])
        self.search = tf.placeholder(tf.float32, [self.batch_size, 227, 227, 3])
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
            x = resUnit(x,256, 3, 'targetResUnit5')
            self.conv_output_target = tf.layers.conv2d(x , 64 , 1 , 1 , name='conv6redu_target' , reuse=False, activation=tf.nn.elu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(self.conv_output_target)

        ########### for mid ###########
        with tf.variable_scope("net_mid"):

            x = self.mid
            x = resUnit(x, 16, 3, 'midResUnit1')
            x = resUnit(x, 32, 3, 'midResUnit2')
            x = resUnit(x, 64, 3, 'midResUnit3')
            x = resUnit(x,128, 3, 'midResUnit4')
            self.midF = x
            x = resUnit(x,256, 3, 'midResUnit5')
            self.conv_output_mid = tf.layers.conv2d(x , 64 , 1 , 1 , name='conv6redu_mid' , reuse=False, activation=tf.nn.elu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(self.conv_output_mid)

        ########### for search ###########
        with tf.variable_scope("net_search"):

            x = self.search
            x = resUnit(x, 16, 3, 'searchResUnit1')
            x = resUnit(x, 32, 3, 'searchResUnit2')
            x = resUnit(x, 64, 3, 'searchResUnit3')
            x = resUnit(x,128, 3, 'searchResUnit4')
            self.searchF = x
            x = resUnit(x,256, 3, 'searchResUnit5')
            self.conv_output_search = tf.layers.conv2d(x , 64 , 1 , 1 , name='conv6redu_search' , reuse=False, activation=tf.nn.elu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(self.conv_output_search)

        ########### Concatnate all nets ###########
        ########### fully connencted layers ###########
        with tf.variable_scope("fc_nets"):

            # now three features map, each 6 x 6 x 64
            concat = tf.concat([self.conv_output_target, self.conv_output_mid, self.conv_output_search], axis = 3)

            x = tf.layers.flatten(concat)

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
            badBBmid = 0
            badBBsearch = 0

            ## Extract the tracked object in the input image
            targetPM = self.targetF
            targetPM = targetPM[:, targetPM.shape[1]*1/3:targetPM.shape[1]*2/3, targetPM.shape[2]*1/3:targetPM.shape[2]*2/3, :]

            def f1(batchIdx, img): return tf.abs(img[batchIdx, img.shape[1]*1/3:img.shape[1]*2/3, img.shape[2]*1/3:img.shape[2]*2/3, :]-256)
            def f2(img): return tf.image.resize_images(img, [int(targetPM.shape[1]),int(targetPM.shape[2])])

            midF = self.midF
            midPM = midF[0,tf.cast(int(midF.shape[1])*fc_output_mid[0,1], tf.int32):tf.cast(int(midF.shape[1])*fc_output_mid[0,3], tf.int32),
                    tf.cast(int(midF.shape[2])*fc_output_mid[0,0], tf.int32):tf.cast(int(midF.shape[2])*fc_output_mid[0,2], tf.int32), :]
            badBBmid = tf.cond(tf.equal(tf.size(midPM), 0), lambda: tf.add(badBBmid, 1), lambda: badBBmid, name="badBBmid")
            midPM = tf.cond(tf.equal(tf.size(midPM), 0), lambda: f1(0, midF), lambda: f2(midPM))
            midPM = tf.expand_dims(midPM, 0)
            for i in range(1, self.batch_size):
                midF_cropNscale = midF[i,tf.cast(int(midF.shape[1])*fc_output_mid[i,1], tf.int32):tf.cast(int(midF.shape[1])*fc_output_mid[i,3], tf.int32),
                        tf.cast(int(midF.shape[2])*fc_output_mid[i,0], tf.int32):tf.cast(int(midF.shape[2])*fc_output_mid[i,2], tf.int32), :]
                badBBmid = tf.cond(tf.equal(tf.size(midF_cropNscale), 0), lambda: tf.add(badBBmid, 1), lambda: badBBmid, name="badBBmid")
                midF_cropNscale = tf.cond(tf.equal(tf.size(midF_cropNscale), 0), lambda: f1(i, midF), lambda: f2(midF_cropNscale))
                midF_cropNscale = tf.expand_dims(midF_cropNscale, 0)
                midPM = tf.concat([midPM,midF_cropNscale], axis=0)

            bboxGT = self.bbox
            bboxPredSearch = fc_output_search

            searchF = self.searchF
            searchPM = searchF[0,tf.cast(int(searchF.shape[1])*bboxGT[0,1], tf.int32):tf.cast(int(searchF.shape[1])*bboxGT[0,3], tf.int32),
                    tf.cast(int(searchF.shape[2])*bboxGT[0,0], tf.int32):tf.cast(int(searchF.shape[2])*bboxGT[0,2], tf.int32), :]
            badBBsearch = tf.cond(tf.equal(tf.size(searchPM), 0), lambda: tf.add(badBBsearch, 1), lambda: badBBsearch, name="badBBsearch")
            searchPM = tf.cond(tf.equal(tf.size(searchPM), 0), lambda: f1(0, searchF), lambda: f2(searchPM))
            searchPM = tf.expand_dims(searchPM, 0)
            for i in range(1, self.batch_size):
                searchF_cropNscale = searchF[i,tf.cast(int(searchF.shape[1])*bboxGT[i,1], tf.int32):tf.cast(int(searchF.shape[1])*bboxGT[i,3], tf.int32),
                        tf.cast(int(searchF.shape[2])*bboxGT[i,0], tf.int32):tf.cast(int(searchF.shape[2])*bboxGT[i,2], tf.int32), :]
                badBBsearch = tf.cond(tf.equal(tf.size(searchF_cropNscale), 0), lambda: tf.add(badBBsearch, 1), lambda: badBBsearch, name="badBBsearch")
                searchF_cropNscale = tf.cond(tf.equal(tf.size(searchF_cropNscale), 0), lambda: f1(i, searchF), lambda: f2(searchF_cropNscale))
                searchF_cropNscale = tf.expand_dims(searchF_cropNscale, 0)
                searchPM = tf.concat([searchPM, searchF_cropNscale], axis=0)

            _variable_summaries(badBBmid)
            _variable_summaries(badBBsearch)

            self.checks = {'badBBmid': badBBmid, 'badBBsearch': badBBsearch}

            ## Calculate NCC photometric losses on input images
            #meanTarget, varTarget = tf.nn.moments(targetPM, axes=[1,2,3])
            #meanTarget = tf.reshape(meanTarget, [-1,1,1,1])
            #stdTarget = tf.reshape(tf.sqrt(varTarget), [-1,1,1,1])

            #meanMid, varMid = tf.nn.moments(midPM, axes=[1,2,3])
            #meanMid = tf.reshape(meanMid, [-1,1,1,1])
            #stdMid = tf.reshape(tf.sqrt(varMid), [-1,1,1,1])

            #meanSearch, varSearch = tf.nn.moments(searchPM, axes=[1,2,3])
            #meanSearch = tf.reshape(meanSearch, [-1,1,1,1])
            #stdSearch = tf.reshape(tf.sqrt(varSearch), [-1,1,1,1])

            #pmLossTargetMid = tf.divide(1., tf.abs(tf.reduce_mean(((targetPM-meanTarget)/stdTarget)*((midPM-meanMid)/stdMid))), name="pmNccLossTargetMid")
            #_variable_summaries(pmLossTargetMid)
            #self.pmLossTargetMid = tf.where(tf.is_nan(pmLossTargetMid), 0., pmLossTargetMid)
            #pmLossMidSearch = tf.divide(1., tf.abs(tf.reduce_mean(((searchPM-meanSearch)/stdSearch)*((midPM-meanMid)/stdMid))), name="pmNccLossMidSearch")
            #_variable_summaries(pmLossMidSearch)
            #self.pmLossMidSearch = tf.where(tf.is_nan(pmLossMidSearch), 0., pmLossMidSearch)


            ## Extract the tracked object in the feature maps
            #targetPM = self.targetF
            #targetPM = targetPM[:, targetPM.shape[1]*1/3:targetPM.shape[1]*2/3, targetPM.shape[2]*1/3:targetPM.shape[2]*2/3, :]

            #midF = self.midF
            #midPM = midF[0,tf.cast(int(midF.shape[1])*fc_output_mid[0,1], tf.int32):tf.cast(int(midF.shape[1])*fc_output_mid[0,3], tf.int32),
            #        tf.cast(int(midF.shape[2])*fc_output_mid[0,0], tf.int32):tf.cast(int(midF.shape[2])*fc_output_mid[0,2], tf.int32), :]
            #midPM = tf.cond(tf.equal(tf.size(midPM), 0), lambda: f1(0, midF), lambda: f2(midPM))
            #midPM = tf.expand_dims(midPM, 0)
            #for i in range(1, self.batch_size):
            #    midF_cropNscale = midF[i,tf.cast(int(midF.shape[1])*fc_output_mid[i,1], tf.int32):tf.cast(int(midF.shape[1])*fc_output_mid[i,3], tf.int32),
            #            tf.cast(int(midF.shape[2])*fc_output_mid[i,0], tf.int32):tf.cast(int(midF.shape[2])*fc_output_mid[i,2], tf.int32), :]
            #    midF_cropNscale = tf.cond(tf.equal(tf.size(midF_cropNscale), 0), lambda: f1(i, midF), lambda: f2(midF_cropNscale))
            #    midF_cropNscale = tf.expand_dims(midF_cropNscale, 0)
            #    midPM = tf.concat([midPM,midF_cropNscale], axis=0)

            #searchF = self.searchF
            #searchPM = searchF[0,tf.cast(int(searchF.shape[1])*bboxGT[0,1], tf.int32):tf.cast(int(searchF.shape[1])*bboxGT[0,3], tf.int32),
            #        tf.cast(int(searchF.shape[2])*bboxGT[0,0], tf.int32):tf.cast(int(searchF.shape[2])*bboxGT[0,2], tf.int32), :]
            #searchPM = tf.cond(tf.equal(tf.size(searchPM), 0), lambda: f1(0, searchF), lambda: f2(searchPM))
            #searchPM = tf.expand_dims(searchPM, 0)
            #for i in range(1, self.batch_size):
            #    searchF_cropNscale = searchF[i,tf.cast(int(searchF.shape[1])*bboxGT[i,1], tf.int32):tf.cast(int(searchF.shape[1])*bboxGT[i,3], tf.int32),
            #            tf.cast(int(searchF.shape[2])*bboxGT[i,0], tf.int32):tf.cast(int(searchF.shape[2])*bboxGT[i,2], tf.int32), :]
            #    searchF_cropNscale = tf.cond(tf.equal(tf.size(searchF_cropNscale), 0), lambda: f1(i, searchF), lambda: f2(searchF_cropNscale))
            #    searchF_cropNscale = tf.expand_dims(searchF_cropNscale, 0)
            #    searchPM = tf.concat([searchPM, searchF_cropNscale], axis=0)

            ## Calculate NCC photometric losses on feature maps
            meanTarget, varTarget = tf.nn.moments(targetPM, axes=[1,2,3])
            meanTarget = tf.reshape(meanTarget, [-1,1,1,1])
            stdTarget = tf.reshape(tf.sqrt(varTarget), [-1,1,1,1])

            meanMid, varMid = tf.nn.moments(midPM, axes=[1,2,3])
            meanMid = tf.reshape(meanMid, [-1,1,1,1])
            stdMid = tf.reshape(tf.sqrt(varMid), [-1,1,1,1])

            meanSearch, varSearch = tf.nn.moments(searchPM, axes=[1,2,3])
            meanSearch = tf.reshape(meanSearch, [-1,1,1,1])
            stdSearch = tf.reshape(tf.sqrt(varSearch), [-1,1,1,1])

            pmLossTargetMidF = tf.divide(1., tf.abs(tf.reduce_mean(((targetPM-meanTarget)/stdTarget)*((midPM-meanMid)/stdMid))), name="pmNccLossTargetMidF")
            _variable_summaries(pmLossTargetMidF)
            self.pmLossTargetMidF = tf.where(tf.is_nan(pmLossTargetMidF), 0., pmLossTargetMidF)
            pmLossMidSearchF = tf.divide(1., tf.abs(tf.reduce_mean(((searchPM-meanSearch)/stdSearch)*((midPM-meanMid)/stdMid))), name="pmNccLossMidSearchF")
            _variable_summaries(pmLossMidSearchF)
            self.pmLossMidSearchF = tf.where(tf.is_nan(pmLossMidSearchF), 0., pmLossMidSearchF)


            ## Calculate mid predicted BB distance from search GT BB and penalize if above threshold
            midBBDistFromGT = tf.subtract(bboxGT, fc_output_mid)
            midBBDistFromGT = tf.abs(midBBDistFromGT)
            midBBDistFromGT = tf.where(tf.greater(midBBDistFromGT, 0.5), midBBDistFromGT, tf.zeros(midBBDistFromGT.shape, dtype=tf.float32))
            midBBDistFromGT = tf.reduce_sum(midBBDistFromGT, axis=1)
            self.midBBoxLoss = tf.reduce_mean(midBBDistFromGT, name="midBBoxLoss")
            _variable_summaries(self.midBBoxLoss)

            ## Calculate bounding box regression loss
            bboxDist = tf.subtract(bboxGT, bboxPredSearch)
            bboxDist = tf.abs(bboxDist)
            bboxDist = tf.reduce_sum(bboxDist, axis=1)
            self.bboxLoss = tf.reduce_mean(bboxDist, name="bboxLoss")
            _variable_summaries(self.bboxLoss)

            #self.loss = self.bboxLoss + self.midBBoxLoss + self.pmLossTargetMid + self.pmLossMidSearch + self.pmLossTargetMidF + self.pmLossMidSearchF
            self.loss = self.bboxLoss + self.midBBoxLoss + (self.pmLossTargetMidF-1) + (self.pmLossMidSearchF-1)

            self.losses = {
                    'bboxLoss': self.bboxLoss,
                    'midBBoxLoss': self.midBBoxLoss,
                    'pmLossTargetMidF': self.pmLossTargetMidF-1,
                    'pmLossMidSearchF': self.pmLossMidSearchF-1
            }




