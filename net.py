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
        self.image = tf.placeholder(tf.float32, [self.batch_size, 227, 227, 3])
        self.bbox_motion = tf.placeholder(tf.float32, [self.batch_size, 6])

        ########### for target ###########
        with tf.variable_scope("net_target"):

            x = tf.layers.conv2d(self.target,  8 , 3 , 2 , name='conv1_target' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 16 , 3 , 1 , name='conv2_target' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 16 , 3 , 1 , name='conv3_target' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 32 , 3 , 2 , name='conv4_target' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 32 , 3 , 1 , name='conv5_target' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 64 , 3 , 2 , name='conv6_target' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 64 , 3 , 1 , name='conv7_target' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          ,128 , 3 , 2 , name='conv8_target' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          ,128 , 3 , 1 , name='conv9_target' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          ,256 , 3 , 2 , name='conv10_target', reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            self.conv_output_target = tf.layers.conv2d(x ,256 , 3 , 1 , name='conv11_target' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(self.conv_output_target)

        ########### for target ###########
        with tf.variable_scope("net_search"):

            x = tf.layers.conv2d(self.image ,  8 , 3 , 2 , name='conv1_search' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 16 , 3 , 1 , name='conv2_search' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 16 , 3 , 1 , name='conv3_search' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 32 , 3 , 2 , name='conv4_search' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 32 , 3 , 1 , name='conv5_search' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 64 , 3 , 2 , name='conv6_search' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          , 64 , 3 , 1 , name='conv7_search' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          ,128 , 3 , 2 , name='conv8_search' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          ,128 , 3 , 1 , name='conv9_search' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.conv2d(x          ,256 , 3 , 2 , name='conv10_search', reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            self.conv_output_search = tf.layers.conv2d(x ,256 , 3 , 1 , name='conv11_search' , reuse=False, activation=tf.nn.relu , padding='same' , kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(self.conv_output_search)

        ########### Concatnate two nets ###########
        ########### fully connencted layers ###########
        with tf.variable_scope("fc_two_nets"):

            # now two features map, each 6 x 6 x 256
            concat = tf.concat([self.conv_output_target, self.conv_output_search], axis = 3)

            x = tf.contrib.layers.flatten(concat)
            x = tf.layers.dense(x, 4096, name='fc1', activation=tf.nn.relu, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.dense(x, 4096, name='fc2', activation=tf.nn.relu, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.dense(x, 2096, name='fc3', activation=tf.nn.relu, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            _activation_summary(x)
            x = tf.layers.dense(x,    6, name='fc4', activation=None, kernel_regularizer=self.wreg , bias_regularizer=self.breg)
            self.fc_output = tf.reshape(x, [self.batch_size, 6])
            _activation_summary(self.fc_output)


        with tf.variable_scope("result"):

            self.result = {
                    'bbox': self.fc_output[:,:4],
                    'motion': self.fc_output[:,4:],
            }


        #-----------------------------------------------------------------------
        # Compute loss
        #-----------------------------------------------------------------------
        with tf.variable_scope("loss"):

            bboxGT = self.bbox_motion[:,:4]
            motionGT = self.bbox_motion[:,4:]
            bboxPred = self.fc_output[:,:4]
            motionPred = self.fc_output[:,4:]

            ## Calculate bounding box loss
            bboxDist = tf.subtract(bboxGT, bboxPred)
            bboxDist = tf.abs(bboxDist)
            bboxDist = tf.reduce_sum(bboxDist, axis=1)
            self.bboxLoss = tf.reduce_mean(bboxDist, name="bboxLoss")
            _variable_summaries(self.bboxLoss)

            ## Calculate bounding box loss
            motionDist = tf.subtract(motionGT, motionPred)
            motionDist = tf.abs(motionDist)
            motionDist = tf.reduce_sum(motionDist, axis=1)
            self.motionLoss = tf.reduce_mean(motionDist, axis=0, name="motionLoss")
            _variable_summaries(self.motionLoss)

            #self.checks = {'bboxDist': bboxDist, 'motionDist': motionDist}


        #-----------------------------------------------------------------------
        # Compute total loss
        #-----------------------------------------------------------------------
        with tf.variable_scope('total_loss'):
            self.loss = tf.add(self.bboxLoss, self.motionLoss*2, name='loss')


        #-----------------------------------------------------------------------
        # Store the tensors
        #-----------------------------------------------------------------------
        self.losses = {
            'total': self.loss,
            'bboxLoss': self.bboxLoss,
            'motionLoss': self.motionLoss
        }

