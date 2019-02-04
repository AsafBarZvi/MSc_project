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
        self.mid1 = tf.placeholder(tf.float32, [self.batch_size, 227, 227, 3])
        self.mid2 = tf.placeholder(tf.float32, [self.batch_size, 227, 227, 3])
        self.search = tf.placeholder(tf.float32, [self.batch_size, 227, 227, 3])
        self.bbox= tf.placeholder(tf.float32, [self.batch_size, 4])


        ########### for target ###########
        with tf.variable_scope("net_target"):

            x = tf.layers.conv2d(self.target,  8 , 3 , 2 , name='conv1_target'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 16 , 3 , 1 , name='conv2_target'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 32 , 3 , 2 , name='conv2a_target'    , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 16 , 1 , 1 , name='conv2redu_target' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 32 , 3 , 1 , name='conv3_target'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 64 , 3 , 2 , name='conv3a_target'    , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 32 , 1 , 1 , name='conv3redu_target' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 64 , 3 , 1 , name='conv4_target'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 96 , 3 , 2 , name='conv4a_target'    , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 64 , 1 , 1 , name='conv4redu_target' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 96 , 3 , 1 , name='conv5_target'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            self.targetF = x
            x = tf.layers.conv2d(x          ,128 , 3 , 2 , name='conv5a_target'    , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            self.conv_output_target = tf.layers.conv2d(x , 32 , 1 , 1 , name='conv5redu_target' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(self.conv_output_target)

        ########### for mid1 ###########
        with tf.variable_scope("net_mid1"):

            x = tf.layers.conv2d(self.mid1  ,  8 , 3 , 2 , name='conv1_mid1'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 16 , 3 , 1 , name='conv2_mid1'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 32 , 3 , 2 , name='conv2a_mid1'    , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 16 , 1 , 1 , name='conv2redu_mid1' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 32 , 3 , 1 , name='conv3_mid1'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 64 , 3 , 2 , name='conv3a_mid1'    , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 32 , 1 , 1 , name='conv3redu_mid1' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 64 , 3 , 1 , name='conv4_mid1'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 96 , 3 , 2 , name='conv4a_mid1'    , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 64 , 1 , 1 , name='conv4redu_mid1' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 96 , 3 , 1 , name='conv5_mid1'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            self.mid1F = x
            x = tf.layers.conv2d(x          ,128 , 3 , 2 , name='conv5a_mid1'    , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            self.conv_output_mid1 = tf.layers.conv2d(x , 32 , 1 , 1 , name='conv5redu_mid1' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(self.conv_output_mid1)

        ########### for mid2 ###########
        with tf.variable_scope("net_mid2"):

            x = tf.layers.conv2d(self.mid2  ,  8 , 3 , 2 , name='conv1_mid2'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 16 , 3 , 1 , name='conv2_mid2'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 32 , 3 , 2 , name='conv2a_mid2'    , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 16 , 1 , 1 , name='conv2redu_mid2' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 32 , 3 , 1 , name='conv3_mid2'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 64 , 3 , 2 , name='conv3a_mid2'    , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 32 , 1 , 1 , name='conv3redu_mid2' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 64 , 3 , 1 , name='conv4_mid2'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 96 , 3 , 2 , name='conv4a_mid2'    , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 64 , 1 , 1 , name='conv4redu_mid2' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 96 , 3 , 1 , name='conv5_mid2'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            self.mid2F = x
            x = tf.layers.conv2d(x          ,128 , 3 , 2 , name='conv5a_mid2'    , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            self.conv_output_mid2 = tf.layers.conv2d(x , 32 , 1 , 1 , name='conv5redu_mid2' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(self.conv_output_mid2)


        ########### for search ###########
        with tf.variable_scope("net_search"):

            x = tf.layers.conv2d(self.search  ,  8 , 3 , 2 , name='conv1_search'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 16 , 3 , 1 , name='conv2_search'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 32 , 3 , 2 , name='conv2a_search'    , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 16 , 1 , 1 , name='conv2redu_search' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 32 , 3 , 1 , name='conv3_search'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 64 , 3 , 2 , name='conv3a_search'    , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 32 , 1 , 1 , name='conv3redu_search' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 64 , 3 , 1 , name='conv4_search'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 96 , 3 , 2 , name='conv4a_search'    , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 64 , 1 , 1 , name='conv4redu_search' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 96 , 3 , 1 , name='conv5_search'     , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            self.searchF = x
            x = tf.layers.conv2d(x          ,128 , 3 , 2 , name='conv5a_search'    , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            self.conv_output_search = tf.layers.conv2d(x , 32 , 1 , 1 , name='conv5redu_search' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(self.conv_output_search)

        ########### Concatnate all nets ###########
        ########### fully connencted layers ###########
        with tf.variable_scope("fc_all_nets"):

            # now four features map, each 6 x 6 x 32
            concat1 = tf.concat([self.conv_output_target, self.conv_output_mid1], axis = 3)
            concat2 = tf.concat([self.conv_output_target, self.conv_output_mid1, self.conv_output_mid2], axis = 3)
            concat3 = tf.concat([self.conv_output_target, self.conv_output_mid1, self.conv_output_mid2, self.conv_output_search], axis = 3)

            x1 = tf.layers.flatten(concat1)
            x2 = tf.layers.flatten(concat2)
            x3 = tf.layers.flatten(concat3)

            x1 = tf.layers.dense(x1, 1028, name='mid1_fc1', activation=tf.nn.relu, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x1)
            x1 = tf.layers.dense(x1, 1028, name='mid1_fc2', activation=tf.nn.relu, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x1)
            self.fc_output_mid1 = tf.layers.dense(x1,    4, name='mid1_fc3', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)

            x2 = tf.layers.dense(x2, 1028, name='mid2_fc1', activation=tf.nn.relu, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x2)
            x2 = tf.layers.dense(x2, 1028, name='mid2_fc2', activation=tf.nn.relu, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x2)
            self.fc_output_mid2 = tf.layers.dense(x2,    4, name='mid2_fc3', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)

            x3 = tf.layers.dense(x3, 1028, name='search_fc1', activation=tf.nn.relu, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x3)
            x3 = tf.layers.dense(x3, 1028, name='search_fc2', activation=tf.nn.relu, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x3)
            self.fc_output_search = tf.layers.dense(x3,    4, name='search_fc3', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)

            self.result = {
                    'bbox_mid1': self.fc_output_mid1,
                    'bbox_mid2': self.fc_output_mid2,
                    'bbox_search': self.fc_output_search
            }


        #-----------------------------------------------------------------------
        # Compute loss
        #-----------------------------------------------------------------------
        with tf.variable_scope("loss"):

            fc_output_mid1 = self.fc_output_mid1
            fc_output_mid2 = self.fc_output_mid2
            fc_output_search = self.fc_output_search
            badBBmid1 = 0
            badBBmid2 = 0
            badBBsearch = 0

            targetPM = self.target
            targetPM = targetPM[:, targetPM.shape[1]*1/4:targetPM.shape[1]*3/4, targetPM.shape[2]*1/4:targetPM.shape[2]*3/4, :]

            def f1(batchIdx, img): return img[batchIdx, img.shape[1]*1/4:img.shape[1]*3/4, img.shape[2]*1/4:img.shape[2]*3/4, :]
            def f2(img): return tf.image.resize_images(img, [int(targetPM.shape[1]),int(targetPM.shape[2])])

            mid1F = self.mid1
            mid1PM = mid1F[0,tf.cast(int(mid1F.shape[1])*fc_output_mid1[0,1], tf.int32):tf.cast(int(mid1F.shape[1])*fc_output_mid1[0,3], tf.int32),
                    tf.cast(int(mid1F.shape[2])*fc_output_mid1[0,0], tf.int32):tf.cast(int(mid1F.shape[2])*fc_output_mid1[0,2], tf.int32), :]
            badBBmid1 = tf.cond(tf.equal(tf.size(mid1PM), 0), lambda: tf.add(badBBmid1, 1), lambda: badBBmid1, name="badBBmid1")
            mid1PM = tf.cond(tf.equal(tf.size(mid1PM), 0), lambda: f1(0, mid1F), lambda: f2(mid1PM))
            mid1PM = tf.expand_dims(mid1PM, 0)
            for i in range(1, self.batch_size):
                mid1F_cropNscale = mid1F[i,tf.cast(int(mid1F.shape[1])*fc_output_mid1[i,1], tf.int32):tf.cast(int(mid1F.shape[1])*fc_output_mid1[i,3], tf.int32),
                        tf.cast(int(mid1F.shape[2])*fc_output_mid1[i,0], tf.int32):tf.cast(int(mid1F.shape[2])*fc_output_mid1[i,2], tf.int32), :]
                badBBmid1 = tf.cond(tf.equal(tf.size(mid1F_cropNscale), 0), lambda: tf.add(badBBmid1, 1), lambda: badBBmid1, name="badBBmid1")
                mid1F_cropNscale = tf.cond(tf.equal(tf.size(mid1F_cropNscale), 0), lambda: f1(i, mid1F), lambda: f2(mid1F_cropNscale))
                mid1F_cropNscale = tf.expand_dims(mid1F_cropNscale, 0)
                mid1PM = tf.concat([mid1PM,mid1F_cropNscale], axis=0)

            mid2F = self.mid2
            mid2PM = mid2F[0,tf.cast(int(mid2F.shape[1])*fc_output_mid2[0,1], tf.int32):tf.cast(int(mid2F.shape[1])*fc_output_mid2[0,3], tf.int32),
                    tf.cast(int(mid2F.shape[2])*fc_output_mid2[0,0], tf.int32):tf.cast(int(mid2F.shape[2])*fc_output_mid2[0,2], tf.int32), :]
            badBBmid2 = tf.cond(tf.equal(tf.size(mid2PM), 0), lambda: tf.add(badBBmid2, 1), lambda: badBBmid2, name="badBBmid2")
            mid2PM = tf.cond(tf.equal(tf.size(mid2PM), 0), lambda: f1(0, mid2F), lambda: f2(mid2PM))
            mid2PM = tf.expand_dims(mid2PM, 0)
            for i in range(1, self.batch_size):
                mid2F_cropNscale = mid2F[i,tf.cast(int(mid2F.shape[1])*fc_output_mid2[i,1], tf.int32):tf.cast(int(mid2F.shape[1])*fc_output_mid2[i,3], tf.int32),
                        tf.cast(int(mid2F.shape[2])*fc_output_mid2[i,0], tf.int32):tf.cast(int(mid2F.shape[2])*fc_output_mid2[i,2], tf.int32), :]
                badBBmid2 = tf.cond(tf.equal(tf.size(mid2F_cropNscale), 0), lambda: tf.add(badBBmid2, 1), lambda: badBBmid2, name="badBBmid2")
                mid2F_cropNscale = tf.cond(tf.equal(tf.size(mid2F_cropNscale), 0), lambda: f1(i, mid2F), lambda: f2(mid2F_cropNscale))
                mid2F_cropNscale = tf.expand_dims(mid2F_cropNscale, 0)
                mid2PM = tf.concat([mid2PM,mid2F_cropNscale], axis=0)

            #searchF = self.search
            #searchPM = searchF[0,tf.cast(int(searchF.shape[1])*fc_output_search[0,1], tf.int32):tf.cast(int(searchF.shape[1])*fc_output_search[0,3], tf.int32),
            #        tf.cast(int(searchF.shape[2])*fc_output_search[0,0], tf.int32):tf.cast(int(searchF.shape[2])*fc_output_search[0,2], tf.int32), :]
            #badBBsearch = tf.cond(tf.equal(tf.size(searchPM), 0), lambda: tf.add(badBBsearch, 1), lambda: badBBsearch, name="badBBsearch")
            #searchPM = tf.cond(tf.equal(tf.size(searchPM), 0), lambda: f1(0, searchF), lambda: f2(searchPM))
            #searchPM = tf.expand_dims(searchPM, 0)
            #for i in range(1, self.batch_size):
            #    searchF_cropNscale = searchF[i,tf.cast(int(searchF.shape[1])*fc_output_search[i,1], tf.int32):tf.cast(int(searchF.shape[1])*fc_output_search[i,3], tf.int32),
            #            tf.cast(int(searchF.shape[2])*fc_output_search[i,0], tf.int32):tf.cast(int(searchF.shape[2])*fc_output_search[i,2], tf.int32), :]
            #    badBBsearch = tf.cond(tf.equal(tf.size(searchF_cropNscale), 0), lambda: tf.add(badBBsearch, 1), lambda: badBBsearch, name="badBBsearch")
            #    searchF_cropNscale = tf.cond(tf.equal(tf.size(searchF_cropNscale), 0), lambda: f1(i, searchF), lambda: f2(searchF_cropNscale))
            #    searchF_cropNscale = tf.expand_dims(searchF_cropNscale, 0)
            #    searchPM = tf.concat([searchPM,searchF_cropNscale], axis=0)

            bboxGT = self.bbox
            bboxPredSearch = self.fc_output_search

            searchF = self.search
            searchPM = searchF[0, bboxGT[0,1]:bboxGT[0,3], bboxGT[0,0]:bboxGT[0,2], :]
            badBBsearch = tf.cond(tf.equal(tf.size(searchPM), 0), lambda: tf.add(badBBsearch, 1), lambda: badBBsearch, name="badBBsearch")
            searchPM = tf.cond(tf.equal(tf.size(searchPM), 0), lambda: f1(0, searchF), lambda: f2(searchPM))
            searchPM = tf.expand_dims(searchPM, 0)
            for i in range(1, self.batch_size):
                searchF_cropNscale = searchF[i, bboxGT[i,1]:bboxGT[i,3], bboxGT[i,0]:bboxGT[i,2], :]
                badBBsearch = tf.cond(tf.equal(tf.size(searchF_cropNscale), 0), lambda: tf.add(badBBsearch, 1), lambda: badBBsearch, name="badBBsearch")
                searchF_cropNscale = tf.cond(tf.equal(tf.size(searchF_cropNscale), 0), lambda: f1(i, searchF), lambda: f2(searchF_cropNscale))
                searchF_cropNscale = tf.expand_dims(searchF_cropNscale, 0)
                searchPM = tf.concat([searchPM, searchF_cropNscale], axis=0)


            _variable_summaries(badBBmid1)
            _variable_summaries(badBBmid2)
            _variable_summaries(badBBsearch)

            self.checks = {'badBBmid1': badBBmid1, 'badBBmid2': badBBmid2, 'badBBsearch': badBBsearch}

            ## Calculate photometric losses
            self.pmLossTargetMid1 = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(targetPM/256-mid1PM/256)), name="pmLossTargetMid1"))
            _variable_summaries(self.pmLossTargetMid1)
            #self.pmLossMid1Mid2 = tf.reduce_mean(tf.reduce_sum(tf.square(mid1PM/256-mid2PM/256)), name="pmLossMid1Mid2")
            #_variable_summaries(self.pmLossMid1Mid2)
            self.pmLossMid2Search = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(mid2PM/256-searchPM/256)), name="pmLossMid2Search"))
            _variable_summaries(self.pmLossMid2Search)

            ## Calculate bounding box regression loss
            bboxDist = tf.subtract(bboxGT, bboxPredSearch)
            bboxDist = tf.abs(bboxDist)
            bboxDist = tf.reduce_sum(bboxDist, axis=1)
            self.bboxLoss = tf.reduce_mean(bboxDist, name="bboxLoss")
            _variable_summaries(self.bboxLoss)

            #self.regLoss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='regLoss')
            #self.regLoss = tf.losses.get_regularization_loss()
            #_variable_summaries(self.regLoss)
            #self.loss = self.bboxLoss + self.pmLossTargetMid1 + self.pmLossMid1Mid2 + self.pmLossMid2Search
            self.loss = self.bboxLoss + self.pmLossTargetMid1 + self.pmLossMid2Search

            self.losses = {
                    'bboxLoss': self.bboxLoss,
                    'pmLossTargetMid1': self.pmLossTargetMid1,
                    #'pmLossMid1Mid2': self.pmLossMid1Mid2,
                    'pmLossMid2Search': self.pmLossMid2Search
            }


            #self.checks = {'bboxDist': bboxDist, 'motionDist': motionDist}


