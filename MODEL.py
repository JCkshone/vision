import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
import os
import sys

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CLASS_NAME = [
    'Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari',
    'Ford', 'Google', 'Heineken', 'Background']

TRAIN_DIR = "flickr_logos_27_dataset"
CNN_IN_WIDTH = 64
CNN_IN_HEIGHT = 32
CNN_IN_CH = 3
LEARING_RATE = 0.0001
MAX_STEPS = 20001
BATCH_SIZE = 64
CNN_SHAPE = (CNN_IN_HEIGHT, CNN_IN_WIDTH, CNN_IN_CH)
PATCH_SIZE = 5
NUM_CLASSES = len(CLASS_NAME)
PICKLE_FILENAME = 'deep_logo.pickle'

W = tf.Variable(tf.random.normal([1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')


class Model:

    @classmethod
    def init_bias(cls, shape):
        init_bias_val = tf.constant(0.1, shape=shape)
        return tf.Variable(init_bias_val)

    @classmethod
    def init_weights(cls, str, shape):
        initializer = tf.initializers.GlorotUniform()
        return tf.Variable(initializer(shape=shape))
        # return tf.Variable(str, shape=shape, initializer=tf.compat.v1.keras.initializers.glorot_normal())
        # return tf.Variable(str, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

    @classmethod
    def conv2D(cls, x, W):
        # x --> input Tensor.shape-> [batch,H,W,C]
        # W --> Kernel. [filter_H,filter_W,Channel_in,Channel_out]
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # for SAME = 0

    @classmethod
    def conv_layer(cls, input_x, W, b):
        return tf.nn.relu(cls.conv2D(input_x, W) + b)

    @classmethod
    def max_pool_2by2(cls, x):
        # x --> input Tensor.shape-> [batch,H,W,C]
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    @classmethod
    def params(cls):
        # weights and biases
        params = {}
        params['w_conv1'] = cls.init_weights('w_conv1', shape=[PATCH_SIZE, PATCH_SIZE, CNN_IN_CH, 32])
        params['b_conv1'] = cls.init_bias(shape=[32])

        params['w_conv2'] = cls.init_weights('w_conv2', shape=[PATCH_SIZE, PATCH_SIZE, 32, 64])
        params['b_conv2'] = cls.init_bias(shape=[64])

        params['w_conv3'] = cls.init_weights('w_conv3', shape=[PATCH_SIZE, PATCH_SIZE, 64, 128])
        params['b_conv3'] = cls.init_bias(shape=[128])

        params['w_fc1'] = cls.init_weights('w_fc1', shape=[16 * 4 * 128, 2048])
        params['b_fc1'] = cls.init_bias(shape=[2048])

        params['w_fc2'] = cls.init_weights('w_fc2', shape=[2048, NUM_CLASSES])
        params['b_fc2'] = cls.init_bias(shape=[NUM_CLASSES])

        return params

    @classmethod
    def CNN(cls, data, model_params, keep_prob):
        # First layer
        h_conv1 = cls.conv_layer(data, model_params['w_conv1'], model_params['b_conv1'])
        h_pool1 = cls.max_pool_2by2(h_conv1)

        # Second layer
        h_conv2 = cls.conv_layer(h_pool1, model_params['w_conv2'], model_params['b_conv2'])
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

        # Third layer
        h_conv3 = cls.conv_layer(h_pool2, model_params['w_conv3'], model_params['b_conv3'])
        h_pool3 = cls.max_pool_2by2(h_conv3)

        # Fully connected layer
        conv_layer_flat = tf.reshape(h_pool3, [-1, 16 * 4 * 128])
        h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, model_params['w_fc1']) + model_params['b_fc1'])
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob)
        # Output layer
        out = tf.matmul(h_fc1, model_params['w_fc2']) + model_params['b_fc2']
        return out

    @classmethod
    def accuracy(cls, predictions, labels):
        return (100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

    @classmethod
    def reformat(cls, dataset, labels):
        dataset = dataset.reshape((-1, CNN_IN_HEIGHT, CNN_IN_WIDTH, CNN_IN_CH)).astype(np.float32)
        labels = (np.arange(NUM_CLASSES) == labels[:, None]).astype(np.float32)
        return dataset, labels

    @classmethod
    def read_data(cls):
        with open(PICKLE_FILENAME, 'rb') as f:
            save = pickle.load(f)
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            valid_dataset = save['valid_dataset']
            valid_labels = save['valid_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
            del save
            print('Training set', train_dataset.shape, train_labels.shape)
            print('Valid set', valid_dataset.shape, valid_labels.shape)
            print('Test set', test_dataset.shape, test_labels.shape)

        return [train_dataset, valid_dataset, test_dataset], [train_labels, valid_labels, test_labels]

    @classmethod
    def process(cls):
        if len(sys.argv) > 1:
            f = np.load(sys.argv[1])
            # f.files has unordered keys ['arr_8', 'arr_9', 'arr_6'...]
            # Sorting keys by value of numbers
            initial_weights = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]
        else:
            initial_weights = None

        # read input data
        dataset, labels = cls.read_data()
        train_dataset, train_labels = cls.reformat(dataset[0], labels[0])
        valid_dataset, valid_labels = cls.reformat(dataset[1], labels[1])
        test_dataset, test_labels = cls.reformat(dataset[2], labels[2])
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Valid set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)

        graph = tf.Graph()
        with graph.as_default():
            # Weights and biases
            model_params = cls.params()
            # Initial weights
            if initial_weights is not None:
                assert len(model_params) == len(initial_weights)
                assign_ops = [w.assign(v) for w, v in zip(model_params, initial_weights)]
            # Input data
            tf_train_dataset = tf.Variable(tf.ones(shape=(BATCH_SIZE, CNN_IN_HEIGHT, CNN_IN_WIDTH, CNN_IN_CH)),
                                           dtype=tf.float32)

            # tf_train_dataset = tf.placeholder(tf.float32, shape=(BATCH_SIZE, CNN_IN_HEIGHT, CNN_IN_WIDTH, CNN_IN_CH))
            # tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_CLASSES))
            tf_train_labels = tf.Variable(tf.ones(shape=(BATCH_SIZE, NUM_CLASSES)), dtype=tf.float32)
            tf_valid_dataset = tf.constant(valid_dataset)
            tf_test_dataset = tf.constant(test_dataset)
            # Training computation
            logits = cls.CNN(tf_train_dataset, model_params, keep_prob=0.5)

            with tf.name_scope('loss'):
                loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
                tf.summary.scalar('loss', loss)
            optimizer = tf.optimizers.Adam(LEARING_RATE).minimize(loss, var_list=[W, b])

            # Predictions for the training, validation, and test data
            train_prediction = tf.nn.softmax(logits)
            valid_prediction = tf.nn.softmax(cls.CNN(tf_valid_dataset, model_params, keep_prob=1.0))
            test_prediction = tf.nn.softmax(cls.CNN(tf_test_dataset, model_params, keep_prob=1.0))
            # Merge all summaries
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(TRAIN_DIR + '/train')

            # Add ops to save and restore all the variables
            saver = tf.train.Saver()

        # Do training
        with tf.Session(graph=graph) as session:

            tf.global_variables_initializer().run()

            if initial_weights is not None:
                session.run(assign_ops)
                print('initialized by pre-learned values')
            else:
                print('initialized')

            for step in range(MAX_STEPS):
                offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
                batch_data = train_dataset[offset:(offset + BATCH_SIZE), :, :, :]
                batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

                try:
                    ret1, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
                    if step % 100 == 0:
                        summary, ret2 = session.run([merged, optimizer], feed_dict=feed_dict)
                        train_writer.add_summary(summary, step)
                        print('Minibatch loss at step %d: %f' % (step, l))
                        print('Minibatch accuracy: %.1f%%' % cls.accuracy(predictions, batch_labels))
                        print('Validation accuracy: %.1f%%' % cls.accuracy(valid_prediction.eval(), valid_labels))
                except KeyboardInterrupt:
                    last_weights = [p.eval() for p in model_params]
                    np.savez("weights.npz", *last_weights)
                    return last_weights

            print('TEST ACCURACY =  %.1f%%' % cls.accuracy(test_prediction.eval(), test_labels))

            # Save the variables to disk.
            save_dir = "train_models"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, "deep_logo_model")
            saved = saver.save(session, save_path)
            print("Model saved in file: %s" % saved)
