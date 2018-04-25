import math
import sys
import os
import shutil
import tensorflow as tf
import numpy as np

from average_precision import APCalculator
from dataPrep import DataPrep
from utils import *
from tqdm import tqdm
import json
import os
from timer import timer_dict , timerStats
from default import config , printCfg

import goturn_net
#import ipdb

config.__dict__.update()
config.name = sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
print "{} \n {}".format(sys.argv[1],printCfg())


#-------------------------------------------------------------------------------
def compute_lr(lr_values, lr_boundaries):
    with tf.variable_scope('learning_rate'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        lr = tf.train.piecewise_constant(global_step, lr_boundaries, lr_values)
    return lr, global_step

#-------------------------------------------------------------------------------
def main():
    #---------------------------------------------------------------------------
    # Parse the commandline
    #---------------------------------------------------------------------------
    args = config

    snaps_path = os.path.join(config.snapdir , config.name)
    #---------------------------------------------------------------------------
    # Find an existing checkpoint
    #---------------------------------------------------------------------------
    start_epoch = 0
    checkpoint_file = args.checkpoint_file
    if args.continue_training:
        state = tf.train.get_checkpoint_state(snaps_path)
        if state is None:
            print('[!] No network state found in ' + snaps_path)
            return 1

        ckpt_paths = state.all_model_checkpoint_paths
        if not ckpt_paths:
            print('[!] No network state found in ' + snaps_path)
            return 1

        last_epoch = None
        checkpoint_file = None
        for ckpt in ckpt_paths:
            #ckpt_num = os.path.basename(ckpt).split('.')[0][1:]
            try:
                ckpt_num = 0#int(ckpt_num)
            except ValueError:
                continue
            if last_epoch is None or last_epoch < ckpt_num:
                last_epoch = ckpt_num
                checkpoint_file = ckpt

        if checkpoint_file is None:
            print('[!] No checkpoints found, cannot continue!')
            return 1

        metagraph_file = checkpoint_file + '.meta'

        if not os.path.exists(metagraph_file):
            print('[!] Cannot find metagraph'.format(metagraph_file))
            return 1
        start_epoch = last_epoch

    #---------------------------------------------------------------------------
    # Create a project directory
    #---------------------------------------------------------------------------
    else:
        try:
            print('[i] Creating directory {}...'.format(snaps_path))
            os.makedirs(snaps_path)
        except Exception as e:
            print('[!] {}'.format( str(e)))
            #return 1

    print('[i] Starting at epoch: {}'.format( start_epoch+1))

    #---------------------------------------------------------------------------
    # Configure the training data
    #---------------------------------------------------------------------------
    print('[i] Configuring the training data...')
    try:
        dp = DataPrep(args.data_dir)
        dp.run(args.batch_size)
        print('[i] # training samples: {}'.format(dp.num_train))
        print('[i] # validation samples: {}'.format(dp.num_valid))
        print('[i] # batch size train: {}'.format(args.batch_size))
    except (AttributeError, RuntimeError) as e:
        print('[!] Unable to load training data: ' + str(e))
        return 1

    #---------------------------------------------------------------------------
    # Create the network
    #---------------------------------------------------------------------------
    with tf.Session() as sess:
        print('[i] Creating the model...')
        n_train_batches = int(math.ceil(dp.num_train/args.batch_size))
        n_valid_batches = int(math.ceil(dp.num_valid/args.batch_size))

        lr_values = args.lr_values.split(';')
        try:
            lr_values = [float(x) for x in lr_values]
        except ValueError:
            print('[!] Learning rate values must be floats')
            sys.exit(1)

        lr_boundaries = args.lr_boundaries.split(';')
        try:
            lr_boundaries = [int(x) for x in lr_boundaries]
        except ValueError:
            print('[!] Learning rate boundaries must be ints')
            sys.exit(1)

        ret = compute_lr(lr_values, lr_boundaries)
        learning_rate, global_step = ret

        tracknet = goturn_net.TRACKNET(args.batch_size, args.weight_decay)
        tracknet.build()

        with tf.variable_scope('train_step'):
            train_step = tf.train.AdamOptimizer(learning_rate, args.momentum).minimize( \
                    tracknet.loss_wdecay, global_step=global_step, name='train_step')

        init = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        sess.run(init)
        sess.run(init_local)

        coord = tf.train.Coordinator()
        # start the threads
        tf.train.start_queue_runners(sess=sess, coord=coord)


        if (start_epoch != 0) or not checkpoint_file is None:
            try:
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint_file)
            except Exception as E:
                print E

        model_saver = tf.train.Saver(max_to_keep=args.max_snapshots_keep)

        if (not checkpoint_file is None) and not args.continue_training:
            sess.run([tf.assign(global_step,0)])

        initialize_uninitialized_variables(sess)

        #-----------------------------------------------------------------------
        # Create various helpers
        #-----------------------------------------------------------------------
        if not os.path.exists(args.logdir):
            os.mkdir(args.logdir)
        if os.path.exists(os.path.join(args.logdir , config.name)):
            shutil.rmtree(os.path.join(args.logdir , config.name))
            os.mkdir(os.path.join(args.logdir , config.name))
        else:
            os.mkdir(os.path.join(args.logdir , config.name))

        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(args.logdir , config.name), sess.graph)

        #training_ap_calc = APCalculator()
        #validation_ap_calc = APCalculator()

        #-----------------------------------------------------------------------
        # Summaries
        #-----------------------------------------------------------------------
        restore = start_epoch != 0

        #training_ap = PrecisionSummary(sess, summary_writer, 'training', td.lname2id.keys(), restore)
        #validation_ap = PrecisionSummary(sess, summary_writer, 'validation', td.lname2id.keys(), restore)

        training_imgs = ImageSummary(sess, summary_writer, 'training', restore)
        validation_imgs = ImageSummary(sess, summary_writer, 'validation', restore)

        training_loss = LossSummary(sess, summary_writer, 'training', args.summary_interval)
        validation_loss = LossSummary(sess, summary_writer, 'validation', n_valid_batches)

        #-----------------------------------------------------------------------
        # Get the initial snapshot of the network
        #-----------------------------------------------------------------------
        #net_summary_ops = net.build_summaries(restore)
        #if start_epoch == 0:
        #    net_summary = sess.run(net_summary_ops)
        #    summary_writer.add_summary(net_summary, 0)
        #summary_writer.flush()

        #-----------------------------------------------------------------------
        # Cycle through the epoch
        #-----------------------------------------------------------------------
        print('[i] Training...')
        for e in range(start_epoch, args.epochs):
            training_imgs_samples = []
            validation_imgs_samples = []

            #-------------------------------------------------------------------
            # Train
            #-------------------------------------------------------------------
            description = '[i] Train {:>2}/{}'.format(e+1, args.epochs)
            for idx in tqdm(range(n_train_batches), total=n_train_batches, desc=description, unit='batches', leave=False):

                cur_batch = sess.run(dp.batch_queue)

                with timer_dict['train']:
                    [_, loss, res] = sess.run([train_step, tracknet.loss, tracknet.fc4], feed_dict={tracknet.image: cur_batch[0],
                                                                         tracknet.target: cur_batch[1],
                                                                         tracknet.bbox: cur_batch[2]})

                training_loss.add(loss/args.batch_size)

                if idx == 0 or ((idx % args.summary_interval) != 0 and (idx % args.val_interval) != 0):
                    continue

                with timer_dict['summary']:
                    for i in range(5):
                        bbox = np.abs((res[i]/10)*226).astype(np.int)
                        bboxGT = np.abs((cur_batch[2][i]/10)*226).astype(np.int)
                        training_imgs_samples.append((np.copy(cur_batch[0][i]), bbox, bboxGT))

                #timerStats()

                #-------------------------------------------------------------------
                # Write summaries
                #-------------------------------------------------------------------
                training_loss.push((e+1)*idx)

                summary = sess.run(merged_summary,feed_dict={tracknet.image: cur_batch[0],
                                                         tracknet.target: cur_batch[1],
                                                         tracknet.bbox: cur_batch[2]})
                summary_writer.add_summary(summary , (e+1)*idx)

                training_imgs.push((e+1)*idx, training_imgs_samples)
                training_imgs_samples = []

                summary_writer.flush()

                #-------------------------------------------------------------------
                # Validate
                #-------------------------------------------------------------------
                if (idx % args.val_interval) != 0:
                    continue

                description = '[i] Valid {:>2}/{}'.format(e+1, args.epochs)
                for idxTest in tqdm(range(n_valid_batches), total=n_valid_batches, desc=description, unit='batches', leave=False):

                    cur_batch = sess.run(dp.batch_test_queue)

                    [loss, res] = sess.run([tracknet.loss, tracknet.fc4], feed_dict={tracknet.image: cur_batch[0],
                                                                             tracknet.target: cur_batch[1],
                                                                             tracknet.bbox: cur_batch[2]})

                    validation_loss.add(loss/args.batch_size)


                with timer_dict['summary']:
                    for i in range(5):
                        bbox = np.abs((res[i]/10)*226).astype(np.int)
                        bboxGT = np.abs((cur_batch[2][i]/10)*226).astype(np.int)
                        #ipdb.set_trace()
                        validation_imgs_samples.append((np.copy(cur_batch[0][i]), bbox, bboxGT))

                #timerStats()

                #-------------------------------------------------------------------
                # Write summaries
                #-------------------------------------------------------------------
                validation_loss.push((e+1)*idx)

                #net_summary = sess.run(net_summary_ops)
                summary = sess.run(merged_summary,feed_dict={tracknet.image: cur_batch[0],
                                                         tracknet.target: cur_batch[1],
                                                         tracknet.bbox: cur_batch[2]})
                summary_writer.add_summary(summary , (e+1)*idx)

                #training_ap.push(e+1, mAP, APs)
                #validation_ap.push(e+1, mAP, APs)

                #training_ap_calc.clear()
                #validation_ap_calc.clear()

                validation_imgs.push((e+1)*idx, validation_imgs_samples)
                validation_imgs_samples = []

                summary_writer.flush()

                #-------------------------------------------------------------------
                # Save a checktpoint
                #-------------------------------------------------------------------
                checkpoint = '{}/{}_{}.ckpt'.format(snaps_path, e+1, idx)
                model_saver.save(sess, checkpoint)
                #print('[i] Checkpoint saved: ' + checkpoint)

        #-------------------------------------------------------------------
        # Save final checktpoint
        #-------------------------------------------------------------------
        timerStats()
        checkpoint = '{}/final.ckpt'.format(snaps_path)
        model_saver.save(sess, checkpoint)
        print('[i] Checkpoint saved:' + checkpoint)

    return 0

if __name__ == '__main__':
    sys.exit(main())
