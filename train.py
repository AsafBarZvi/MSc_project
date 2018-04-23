import math
import sys
import os

import tensorflow as tf
import numpy as np

from average_precision import APCalculator, APs2mAP , CurveAPCalculator
from dataPrep import DataPrep
from utils import *
from tqdm import tqdm
import json
import os
from timer import timer_dict , timerStats
from default import config , printCfg

import goturn_net


cfg_json = json.load(open(sys.argv[1]))
config.__dict__.update(cfg_json)
config.name = os.path.split(sys.argv[1])[-1]
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
            ckpt_num = os.path.basename(ckpt).split('.')[0][1:]
            try:
                ckpt_num = int(ckpt_num)
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
            print('[!] Cannot find metagraph', metagraph_file)
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
            print('[!]', str(e))
            #return 1

    print('[i] Starting at epoch:    ', start_epoch+1)

    #---------------------------------------------------------------------------
    # Configure the training data
    #---------------------------------------------------------------------------
    print('[i] Configuring the training data...')
    try:
        dp = dataPrep(args.data_dir, args.batch_size)
        print('[i] # training samples:   ', dp.num_train)
        print('[i] # validation samples: ', dp.num_valid)
        print('[i] # batch size train: ', args.batch_size)
    except (AttributeError, RuntimeError) as e:
        print('[!] Unable to load training data:', str(e))
        return 1

    #---------------------------------------------------------------------------
    # Create the network
    #---------------------------------------------------------------------------
    with tf.Session() as sess:
        print('[i] Creating the model...')
        n_train_batches = int(math.ceil(td.num_train/args.batch_size))
        n_valid_batches = int(math.ceil(td.num_valid/args.batch_size))

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

        saver = tf.train.Saver(max_to_keep=config.max_snapshots_to_keep)

        if (start_epoch != 0) or not checkpoint_file is None:
            try:
                saver.restore(sess , checkpoint_file)
            except Exception as E:
                print E

        if (not checkpoint_file is None) and not args.continue_training:
            sess.run([tf.assign(global_step,0)])

        initialize_uninitialized_variables(sess)

        #-----------------------------------------------------------------------
        # Create various helpers
        #-----------------------------------------------------------------------
        if not os.path.exists(os.path.join(args.logdir , config.name)):
            os.mkdir(os.path.join(args.logdir , config.name))

        summary_writer = tf.summary.FileWriter(os.path.join(args.logdir , config.name), sess.graph)

        training_ap_calc = APCalculator()
        validation_ap_calc = APCalculator()

        #-----------------------------------------------------------------------
        # Summaries
        #-----------------------------------------------------------------------
        restore = start_epoch != 0

        #training_ap = PrecisionSummary(sess, summary_writer, 'training', td.lname2id.keys(), restore)
        #validation_ap = PrecisionSummary(sess, summary_writer, 'validation', td.lname2id.keys(), restore)

        training_imgs = ImageSummary(sess, summary_writer, 'training', td.label_colors, restore)
        validation_imgs = ImageSummary(sess, summary_writer, 'validation', td.label_colors, restore)

        training_loss = LossSummary(sess, summary_writer, 'training', td.num_train)
        validation_loss = LossSummary(sess, summary_writer, 'validation', td.num_valid)

        lr_sum  = tf.summary.scalar("lr",learning_rate)
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
        coord = tf.train.Coordinator()
        # start the threads
        tf.train.start_queue_runners(sess=sess, coord=coord)
        print('[i] Training...')
        for e in range(start_epoch, args.epochs):
            training_imgs_samples = []
            validation_imgs_samples = []

            #-------------------------------------------------------------------
            # Train
            #-------------------------------------------------------------------
            description = '[i] Train {:>2}/{}'.format(e+1, args.epochs)
            for idx in tqdm(range(total=n_train_batches), desc=description, unit='batches'):

                cur_batch = sess.run(dp.batch_queue)

                if len(training_imgs_samples) < 3:
                    saved_images = np.copy(curr_batch[1][:3])

                with timer_dict['train']:
                    [res, loss] = sess.run([tracknet.fc4, tracknet.loss, train_step], feed_dict={tracknet.image: cur_batch[0],
                                                                         tracknet.target: cur_batch[1],
                                                                         tracknet.bbox: cur_batch[2]})


                training_loss.add(loss, cur_batch[1].shape[0])

                if e == 0: continue

                with timer_dict['summary']:
                    for i in range(result.shape[0]):
                        if (idx % 100) != 0:
                            continue

                        training_ap_calc.add_detections(cur_batch[2][i],res[i])


                        if len(training_imgs_samples) < 3:
                            bbox = (res[i]/10)*227
                            training_imgs_samples.append((saved_images[i], bbox))


            timerStats()

            #-------------------------------------------------------------------
            # Validate
            #-------------------------------------------------------------------
            description = '[i] Valid {:>2}/{}'.format(e+1, args.epochs)
            for idx in tqdm(range(total=n_valid_batches), desc=description, unit='batches'):

                cur_batch = sess.run(dp.batch_test_queue)

                [res, loss] = sess.run([tracknet.fc4, tracknet.loss], feed_dict={tracknet.image: cur_batch[0],
                                                                         tracknet.target: cur_batch[1],
                                                                         tracknet.bbox: cur_batch[2]})

                validation_loss.add(loss, cur_batch[1].shape[0])

                if e == 0: continue

                for i in range(res.shape[0]):
                    validation_ap_calc.add_detections(cur_batch[2][i],res[i])

                    if len(validation_imgs_samples) < 3:
                        bbox = (res[i]/10)*227
                        validation_imgs_samples.append((np.copy(curr_batch[1][i]), bbox))

            #-------------------------------------------------------------------
            # Write summaries
            #-------------------------------------------------------------------
            training_loss.push(e+1)
            validation_loss.push(e+1)

            #net_summary = sess.run(net_summary_ops)
            summary_writer.add_summary(sess.run([lr_sum])[0], e+1)

            #training_ap.push(e+1, mAP, APs)
            #validation_ap.push(e+1, mAP, APs)


            training_ap_calc.clear()
            validation_ap_calc.clear()

            training_imgs.push(e+1, training_imgs_samples)
            validation_imgs.push(e+1, validation_imgs_samples)

            summary_writer.flush()

            #-------------------------------------------------------------------
            # Save a checktpoint
            #-------------------------------------------------------------------
            if (e+1) % args.checkpoint_interval == 0:
                checkpoint = '{}/e{}.ckpt'.format(snaps_path, e+1)
                saver.save(sess, checkpoint)
                print('[i] Checkpoint saved:', checkpoint)

        checkpoint = '{}/final.ckpt'.format(snaps_path)
        saver.save(sess, checkpoint)
        print('[i] Checkpoint saved:', checkpoint)

    return 0

if __name__ == '__main__':
    sys.exit(main())
