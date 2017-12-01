import argparse
import numpy as np
import tensorflow as tf
import time
import os
import glob
import pprint

parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_dir', type=str, default=None,
                    help='Path to directory containing the Tensorflow checkpoint files')
parser.add_argument('--output', type=str, default='weights',
                    help='Name for the output folder containing weights in .npy format')

args = parser.parse_args()


model_dir =args.model_dir
output = args.output

start = time.time()

print('extracting weights from tensorflow checkpoint...')


model_dir = os.path.abspath(model_dir)
# get string with path to the .meta file
meta_path = glob.glob(os.path.join(model_dir, '*.meta'))[0]

# get string with the pathname to the checkpoint files without file extension. This is for Saver.restore()
get_strings = os.path.join(model_dir,'model*')
string_list = glob.glob(get_strings)
restore_string = os.path.splitext(string_list[0])[0]

with tf.Session() as session:

    # run initializer
    tf.global_variables_initializer().run()

    saver = tf.train.import_meta_graph(meta_path, clear_devices=True)

    # initialize the loaded graph with pre-trained variables
    saver.restore(session, restore_string)
    print('graph restored in {}'.format(time.time() - start))

    # print out the weight arrays
    tensors = tf.trainable_variables()
    weights_list = list()
    for i in xrange(len(tensors)):
        a=session.run(tensors[i])
        weights_list.append(a)

    print('Print the tensors...')
    pprint.pprint(tensors)

    if not os.path.exists(output):
        os.makedirs(output)

    np.save(output + '/weights', weights_list)


end = time.time()
print('Weights extraction time: ', end - start )
