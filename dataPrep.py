import tensorflow as tf
import os


class DataPrep:
    #---------------------------------------------------------------------------
    def __init__(self, data_dir):
        #-----------------------------------------------------------------------
        # Read the dataset info
        #-----------------------------------------------------------------------
        try:
            self.ftrain = open(data_dir + '/trainSet.txt', "r")
            self.ftest = open(data_dir + '/testSet.txt', "r")

        except Exception as e:
            raise RuntimeError(str(e))



    def run(self, batch_size):

        def load_training_set(fset):
            '''
            return train_set
            '''
            trainlines = fset.read().splitlines()
            train_target = []
            train_search = []
            train_box = []
            for line in trainlines:
                line = line.split(",")
                train_target.append(line[0])
                train_search.append(line[1])
                box = [10 * float(line[2]), 10 * float(line[3]), 10 * float(line[4]), 10 * float(line[5])]
                train_box.append(box)
            fset.close()

            return [train_target, train_search, train_box]


        def data_reader(input_queue):
            '''
            this function only read the one pair of images and from the queue
            '''
            search_img = tf.read_file(input_queue[0])
            target_img = tf.read_file(input_queue[1])
            search_tensor = tf.to_float(tf.image.decode_jpeg(search_img, channels=3))
            search_tensor = tf.image.resize_images(search_tensor, [227, 227],
                                                   method=tf.image.ResizeMethod.BILINEAR)
            target_tensor = tf.to_float(tf.image.decode_jpeg(target_img, channels=3))
            target_tensor = tf.image.resize_images(target_tensor, [227, 227],
                                                   method=tf.image.ResizeMethod.BILINEAR)
            box_tensor = input_queue[2]
            return [search_tensor, target_tensor, box_tensor]


        def next_batch(input_queue, batchSize):
            min_queue_examples = 128
            num_threads = 8
            [search_tensor, target_tensor, box_tensor] = data_reader(input_queue)
            [search_batch, target_batch, box_batch] = tf.train.shuffle_batch(
                [search_tensor, target_tensor, box_tensor],
                batch_size=batchSize,
                num_threads=num_threads,
                capacity=min_queue_examples + (num_threads + 2) * batchSize,
                seed=88,
                min_after_dequeue=min_queue_examples)
            return [search_batch, target_batch, box_batch]



        [train_target, train_search, train_box] = load_training_set(self.ftrain)
        [test_target, test_search, test_box] = load_training_set(self.ftest)
        target_tensors = tf.convert_to_tensor(train_target, dtype=tf.string)
        search_tensors = tf.convert_to_tensor(train_search, dtype=tf.string)
        box_tensors = tf.convert_to_tensor(train_box, dtype=tf.float64)
        target_test_tensors = tf.convert_to_tensor(test_target, dtype=tf.string)
        search_test_tensors = tf.convert_to_tensor(test_search, dtype=tf.string)
        box_test_tensors = tf.convert_to_tensor(test_box, dtype=tf.float64)
        input_queue = tf.train.slice_input_producer([search_tensors, target_tensors, box_tensors], shuffle=False)
        input_test_queue = tf.train.slice_input_producer([search_test_tensors, target_test_tensors, box_test_tensors])
        self.batch_queue = next_batch(input_queue, batch_size)
        self.batch_test_queue = next_batch(input_test_queue, batch_size)

        #-----------------------------------------------------------------------
        # Set the attributes up
        #-----------------------------------------------------------------------
        self.num_train       = len(train_box)
        self.num_valid       = len(test_box)




