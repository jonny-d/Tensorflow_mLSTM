import numpy as np
import tensorflow as tf
import pprint
import os
import argparse
import time

parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--meta_path', type=str, default=None,
                    help='Path to .meta file for the saved model')
parser.add_argument('--model_dir', type=str, default=None,
                    help='Name of directory to save models during training')
parser.add_argument('--output', type=str, default='model',
                    help='Name for the output folder containing weights in .npy format')

args = parser.parse_args()

meta_path = args.meta_path
model_path =args.model_dir
output_file = args.output



start = time.time()

#restore a saved model then extract and save the weights in the appropriate format for encoder.py
with tf.Session() as session:

    # run initializer
    tf.global_variables_initializer().run()

    # import the graph
    path = tf.train.get_checkpoint_state(args.model_dir)

    saver = tf.train.import_meta_graph(meta_path,clear_devices=True)
    print('graph imported')

    # initialize the loaded graph with pre-trained variables
    saver.restore(session,path.model_checkpoint_path)
    print('graph restored')

    # print out the weight arrays
    tensors = tf.trainable_variables()
    weights_list = list()
    for i in xrange(len(tensors)):
        a=session.run(tensors[i])
        weights_list.append(a)

    print('Print the tensors...')
    pprint.pprint(tensors)

    # folder to save the weights
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    # this is the embedding matrix W_embedding
    W_embedding = weights_list[0] # W_embedding.shape = (256,64)
    np.save(output_file+'/0.npy', W_embedding)

    # these 4 matrices are (64,4096)
    Wix = weights_list[6]
    Wfx = weights_list[12]
    Wox = weights_list[9]
    Whx = weights_list[3]

    # in encoder.py, these matrices are concatenated and saved as 1.npy
    # wx is the variable name in the encoder.py graph
    wx = np.concatenate((Wix,Wfx,Wox,Whx), axis=1)  # wx.shape = (64,16384)
    np.save(output_file+'/1.npy', wx)

    # These matrices are saved seperately in .npy files, then concatenated from within the encoder.py script
    # before being used for the wh variable in the encoder.py graph. These are all of shape (4096,4096)
    Wim = weights_list[7]
    np.save(output_file+'/2.npy', Wim)

    Wfm = weights_list[13]
    np.save(output_file+'/3.npy', Wfm)

    Wom = weights_list[10]
    np.save(output_file+'/4.npy', Wom)

    Whm = weights_list[4]
    np.save(output_file+'/5.npy', Whm)


    # Wmx is used to calculate the m vector, this is saved as 6.npy
    Wmx = weights_list[1]   # Wmx.shape = (64,4096)
    np.save(output_file+'/6.npy', Wmx)

    # Wmh is used to calculate the m vector, this is saved as 7.npy
    Wmh = weights_list[2]   # Wmh.shape = (4096,4096)
    np.save(output_file+'/7.npy', Wmh)


    # The bias variables are concatenated and saved in the 8.npy file
    # These are 4 (4096,) vectors used to calculate z in encoder.py
    Wib = weights_list[8]
    Wfb = weights_list[14]
    Wob = weights_list[11]
    Whb = weights_list[5]
    b = np.concatenate((Wib,Wfb,Wob,Whb), axis=1)
    # remove singleton dimension
    b = b.squeeze()
    np.save(output_file+'/8.npy',b)

    # Coefficients used for weight normalizationn for the wx matrix, used in the calculation of z
    # the following vectores are conctenated and saved as 9.npy
    gix = weights_list[19]
    gfx = weights_list[23]
    gox = weights_list[21]
    ghx = weights_list[17]
    gx = np.concatenate((gix,gfx,gox,ghx))
    np.save(output_file+'/9.npy',gx)


    # Coefficients used for the  weight normalization for the wh matrix used in the calculation of z,
    # these are concatenated and saved as 10.npy
    gim = weights_list[20]
    gfm = weights_list[24]
    gom = weights_list[22]
    ghm = weights_list[18]
    gh = np.concatenate((gim,gfm,gom,ghm))
    np.save(output_file+'/10.npy',gh)

    # gmx and gmh are the weight normalization coefficients used for wmx and wmh in the calculation of m
    # gmx
    gmx = weights_list[15]
    np.save(output_file+'/11.npy',gmx)

    # gmh
    gmh = weights_list[16]
    np.save(output_file+'/12.npy',gmh)

    # These aren't used for the representation extraction but extract the softmax weights too
    Classifier_w = weights_list[25]
    np.save(output_file+'/13.npy',Classifier_w)
    Classifier_b = weights_list[26]
    np.save(output_file+'/14.npy',Classifier_b)


end = time.time()

print('Weights extraction time: ', end - start )
