#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   29.08.2017
#-------------------------------------------------------------------------------
# This file is part of SSD-TensorFlow.
#
# SSD-TensorFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SSD-TensorFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SSD-Tensorflow.  If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------

import argparse
import math
import cv2

import tensorflow as tf
import numpy as np

from collections import namedtuple
#import ipdb


#-------------------------------------------------------------------------------
def initialize_uninitialized_variables(sess):
    """
    Only initialize the weights that have not yet been initialized by other
    means, such as importing a metagraph and a checkpoint. It's useful when
    extending an existing model.
    """
    uninit_vars    = []
    uninit_tensors = []
    for var in tf.global_variables():
        uninit_vars.append(var)
        uninit_tensors.append(tf.is_variable_initialized(var))
    uninit_bools = sess.run(uninit_tensors)
    uninit = zip(uninit_bools, uninit_vars)
    for init , var in uninit:
        if not init:
            print "Init to {}".format(var)
    uninit = [var for init, var in uninit if not init]
    sess.run(tf.variables_initializer(uninit))

#-------------------------------------------------------------------------------
def load_data_source(data_source):
    """
    Load a data source given it's name
    """
    source_module = __import__('source_'+data_source)
    get_source    = getattr(source_module, 'get_source')
    return get_source()

#-------------------------------------------------------------------------------
def rgb2bgr(tpl):
    """
    Convert RGB color tuple to BGR
    """
    return (tpl[2], tpl[1], tpl[0])

#-------------------------------------------------------------------------------
Label   = namedtuple('Label',   ['name', 'color'])
Size    = namedtuple('Size',    ['w', 'h'])
Point   = namedtuple('Point',   ['x', 'y'])
Sample  = namedtuple('Sample',  ['filename', 'boxes', 'imgsize','idx_in_clip','tsize'])
Sample.__new__.__defaults__ = (None,None)
Box     = namedtuple('Box',     ['label', 'labelid', 'center', 'size','sig','att','is_diff'])
Box.__new__.__defaults__ = (None,None,None)
Score   = namedtuple('Score',   ['idx', 'score'])
Overlap = namedtuple('Overlap', ['best', 'good' , 'amb'])
Overlap.__new__.__defaults__ = (None)

#-------------------------------------------------------------------------------
def str2bool(v):
    """
    Convert a string to a boolean
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#-------------------------------------------------------------------------------
def abs2prop(xmin, xmax, ymin, ymax, imgsize , scale=True):
    """
    Convert the absolute min-max box bound to proportional center-width bounds
    """
    width   = float(xmax-xmin)
    height  = float(ymax-ymin)
    cx      = float(xmin)+width/2
    cy      = float(ymin)+height/2

    if scale:
        width  /= imgsize.w
        height /= imgsize.h
        cx     /= imgsize.w
        cy     /= imgsize.h

    return Point(cx, cy), Size(width, height)

#-------------------------------------------------------------------------------
def prop2abs(center, size, imgsize):
    """
    Convert proportional center-width bounds to absolute min-max bounds
    """
    width2  = size.w*imgsize.w/2
    height2 = size.h*imgsize.h/2
    cx      = center.x*imgsize.w
    cy      = center.y*imgsize.h
    return int(cx-width2), int(cx+width2), int(cy-height2), int(cy+height2)

#-------------------------------------------------------------------------------
def box_is_valid(box):
    for x in [box.center.x, box.center.y, box.size.w, box.size.h]:
        if math.isnan(x) or math.isinf(x):
            return False
    return True

#-------------------------------------------------------------------------------
def draw_box(img, box, color):

    xmin, ymin, xmax, ymax = box
    cv2.rectangle(img, (xmin,ymin), (xmax,ymax), color, 2)


#-------------------------------------------------------------------------------
class PrecisionSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, labels, restore=False):
        self.session = session
        self.writer = writer
        self.labels = labels

        sess = session
        ph_name = sample_name+'_mAP_ph'
        sum_name = sample_name+'_mAP'

        self.mAP_placeholder = tf.placeholder(tf.float32, name=ph_name)
        self.mAP_summary_op = tf.summary.scalar(sum_name,
                                                self.mAP_placeholder)

        self.placeholders = {}
        self.summary_ops = {}

        for label in labels:
            sum_name = sample_name+'_AP_'+label
            ph_name = sample_name+'_AP_ph_'+label
            placeholder = tf.placeholder(tf.float32, name=ph_name)
            summary_op = tf.summary.scalar(sum_name, placeholder)
            self.placeholders[label] = placeholder
            self.summary_ops[label] = summary_op

    #---------------------------------------------------------------------------
    def push(self, epoch, mAP, APs):
        if not APs: return

        feed = {self.mAP_placeholder: mAP}
        tensors = [self.mAP_summary_op]
        for label in self.labels:
            feed[self.placeholders[label]] = APs[label]
            tensors.append(self.summary_ops[label])

        summaries = self.session.run(tensors, feed_dict=feed)

        for summary in summaries:
            self.writer.add_summary(summary, epoch)

#-------------------------------------------------------------------------------
class ImageSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, restore=False):
        self.session = session
        self.writer = writer

        sess = session
        sum_name = sample_name+'_img'
        ph_name = sample_name+'_img_ph'
        self.img_placeholder = tf.placeholder(tf.float32, name=ph_name, shape=[None, None, None, 3])
        self.img_summary_op = tf.summary.image(sum_name, self.img_placeholder, max_outputs=10)

    #---------------------------------------------------------------------------
    def push(self, epoch, samples):
        imgs = np.zeros((5, 227, 227*3, 3))
        for i, sample in enumerate(samples):
            imgTarget = sample[0]
            imgMid = sample[1]
            imgSearch = sample[2]
            imgMidPred = np.copy(imgMid)
            imgMidGT = np.copy(imgMid)
            draw_box(imgMidPred, sample[3], [255,255,0])
            imgSearchPred = np.copy(imgSearch)
            imgSearchGT = np.copy(imgSearch)
            draw_box(imgSearchPred, sample[4], [255,0,0])
            if len(sample[5]) > 4:
                draw_box(imgMidGT, sample[5][:4], [0,255,0])
                draw_box(imgSearchGT, sample[5][4:], [0,255,0])
            else:
                draw_box(imgMidGT, sample[3], [255,255,0])
                draw_box(imgSearchGT, sample[5], [0,255,0])

            #img[img>255] = 255
            #img[img<0] = 0
            alpha = 0.3
            cv2.addWeighted(imgSearchGT, alpha, imgSearchPred, 1.-alpha, 0, imgSearch)
            cv2.addWeighted(imgMidGT, alpha, imgMidPred, 1.-alpha, 0, imgMid)
            imgTrack = np.concatenate((imgTarget, imgMid, imgSearch), axis=1)

            imgs[i] = imgTrack.astype(np.uint8)

        feed = {self.img_placeholder: imgs}
        summary = self.session.run(self.img_summary_op, feed_dict=feed)
        self.writer.add_summary(summary, epoch)

#-------------------------------------------------------------------------------
class LossSummary:
    #---------------------------------------------------------------------------
    def __init__(self, session, writer, sample_name, num_samples):
        self.session = session
        self.writer = writer
        self.num_samples = num_samples
        self.loss_names = ['L1']
        self.loss_values = {}
        self.placeholders = {}

        sess = session

        summary_ops = []
        for loss in self.loss_names:
            sum_name = sample_name+'_'+loss+'_loss'
            ph_name = sample_name+'_'+loss+'_loss_ph'

            placeholder = tf.placeholder(tf.float32, name=ph_name)
            summary_op = tf.summary.scalar(sum_name, placeholder)

            self.loss_values[loss] = float(0)
            self.placeholders[loss] = placeholder
            summary_ops.append(summary_op)

        self.summary_ops = tf.summary.merge(summary_ops)

    #---------------------------------------------------------------------------
    def add(self, value):
        for loss in self.loss_names:
            self.loss_values[loss] += value

    #---------------------------------------------------------------------------
    def push(self, epoch):
        feed = {}
        for loss in self.loss_names:
            feed[self.placeholders[loss]] = self.loss_values[loss]/self.num_samples

        summary = self.session.run(self.summary_ops, feed_dict=feed)
        self.writer.add_summary(summary, epoch)

        for loss in self.loss_names:
            self.loss_values[loss] = float(0)
