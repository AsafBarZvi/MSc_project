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

        def load_data_set(fset, test=False):
            '''
            return data set
            '''
            trainlines = fset.read().splitlines()
            target = []
            mid1 = []
            mid2 = []
            search = []
            bbox = []
            for line in trainlines:
                line = line.split(",")
                target.append(line[0])
                mid1.append(line[1])
                mid2.append(line[2])
                search.append(line[3])
                #box = [10 * float(line[2]), 10 * float(line[3]), 10 * float(line[4]), 10 * float(line[5])]
                # Normalized 0-1 bounding box GT (xmin,ymin,xmax,ymax) + Normalized motion GT (avgX in search to center(0.5) in target, avgY in search to center(0.5) in target)
                if test:
                    box = [float(line[4]), float(line[5]), float(line[6]), float(line[7]), float(line[8]), float(line[9]), float(line[10]), float(line[11]),
                        float(line[12]), float(line[13]), float(line[14]), float(line[15])]
                else:
                    box = [float(line[4]), float(line[5]), float(line[6]), float(line[7])]#, ((float(line[2])+float(line[4]))/2.)-0.5, ((float(line[3])+float(line[5]))/2.)-0.5]

                bbox.append(box)

            fset.close()

            return [target, mid1, mid2, search, bbox]


        def data_reader(input_queue):
            '''
            this function only read the one pair of images and from the queue
            '''
            target_img = tf.read_file(input_queue[0])
            mid1_img = tf.read_file(input_queue[1])
            mid2_img = tf.read_file(input_queue[2])
            search_img = tf.read_file(input_queue[3])
            target_tensor = tf.to_float(tf.image.decode_jpeg(target_img, channels=3))
            target_tensor = tf.image.resize_images(target_tensor, [227, 227],
                                                   method=tf.image.ResizeMethod.BILINEAR)
            mid1_tensor = tf.to_float(tf.image.decode_jpeg(mid1_img, channels=3))
            mid1_tensor = tf.image.resize_images(mid1_tensor, [227, 227],
                                                   method=tf.image.ResizeMethod.BILINEAR)
            mid2_tensor = tf.to_float(tf.image.decode_jpeg(mid2_img, channels=3))
            mid2_tensor = tf.image.resize_images(mid2_tensor, [227, 227],
                                                   method=tf.image.ResizeMethod.BILINEAR)
            search_tensor = tf.to_float(tf.image.decode_jpeg(search_img, channels=3))
            search_tensor = tf.image.resize_images(search_tensor, [227, 227],
                                                   method=tf.image.ResizeMethod.BILINEAR)
            box_tensor = input_queue[4]
            return [target_tensor, mid1_tensor, mid2_tensor, search_tensor, box_tensor]


        def next_batch(input_queue, batchSize):
            min_queue_examples = 128
            num_threads = 8
            [target_tensor, mid1_tensor, mid2_tensor, search_tensor, box_tensor] = data_reader(input_queue)
            [target_batch, mid1_batch, mid2_batch, search_batch, box_batch] = tf.train.shuffle_batch(
                [target_tensor, mid1_tensor, mid2_tensor, search_tensor, box_tensor],
                batch_size=batchSize,
                num_threads=num_threads,
                capacity=min_queue_examples + (num_threads + 2) * batchSize,
                seed=88,
                min_after_dequeue=min_queue_examples)
            return [target_batch, mid1_batch, mid2_batch, search_batch, box_batch]



        [train_target, train_mid1, train_mid2, train_search, train_box] = load_data_set(self.ftrain)
        [test_target, test_mid1, test_mid2, test_search, test_box] = load_data_set(self.ftest, True)
        target_tensors = tf.convert_to_tensor(train_target, dtype=tf.string)
        mid1_tensors = tf.convert_to_tensor(train_mid1, dtype=tf.string)
        mid2_tensors = tf.convert_to_tensor(train_mid2, dtype=tf.string)
        search_tensors = tf.convert_to_tensor(train_search, dtype=tf.string)
        box_tensors = tf.convert_to_tensor(train_box, dtype=tf.float32)
        target_test_tensors = tf.convert_to_tensor(test_target, dtype=tf.string)
        mid1_test_tensors = tf.convert_to_tensor(test_mid1, dtype=tf.string)
        mid2_test_tensors = tf.convert_to_tensor(test_mid2, dtype=tf.string)
        search_test_tensors = tf.convert_to_tensor(test_search, dtype=tf.string)
        box_test_tensors = tf.convert_to_tensor(test_box, dtype=tf.float32)
        input_queue = tf.train.slice_input_producer([target_tensors, mid1_tensors, mid2_tensors, search_tensors, box_tensors], shuffle=False)
        input_test_queue = tf.train.slice_input_producer([target_test_tensors, mid1_test_tensors, mid2_test_tensors, search_test_tensors, box_test_tensors])
        self.batch_queue = next_batch(input_queue, batch_size)
        self.batch_test_queue = next_batch(input_test_queue, batch_size)

        #-----------------------------------------------------------------------
        # Set the attributes up
        #-----------------------------------------------------------------------
        self.num_train       = len(train_box)
        self.num_valid       = len(test_box)




