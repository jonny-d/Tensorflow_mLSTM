# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import numpy as np
import argparse
import random
import tensorflow as tf
import time
import io
from Utilities import DataLoader
import caffeine


parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default=None,
                    help='Path to training data directory') # not optional
parser.add_argument('--saved_models_dir', type=str, default='saved_models',
                    help='Name of directory to save models during training')
parser.add_argument('--log_dir', type=str, default='tensorboard_logs',
                    help='Name of directory for storing tensorboard logs')
parser.add_argument('--rnn_size', type=int, default=128,
                    help='Size of RNN hidden states')
parser.add_argument('--batch_size', type=int, default=32,
                    help='RNN minibatch size')
parser.add_argument('--seq_length', type=int, default=32,
                    help='RNN sequence length')
parser.add_argument('--num_epochs', type=int, default=1,
                    help='Number of training epochs')
parser.add_argument('--init_lr', type=float, default=5*10**-4, # value from paper
                    help='Initial learning rate')
parser.add_argument('--embedding_size', type=int, default=64,
                    help='Character embedding layer size')
parser.add_argument('--wn', type=int, default=1,
                    help='Switch for weight normalisation on the mLSTM parameters. Integer argument of 1 for ON and 0 for OFF')
parser.add_argument('--restore_path', type= str, default=None,
                    help='Path to a directory from which to restore a model from previous session')
parser.add_argument('--summary_frequency', type=int, default=100,
                    help='Save tensorboard data every N steps')
parser.add_argument('--sampling_frequency', type=int, default=100,
                    help='Generate samples from the model during training every N steps ')
parser.add_argument('--num_chars', type=int, default=250,
                    help='Option to specify how many chars to sample from the model if --sampling_frequency is not zero ')
parser.add_argument('--lr_decay', type=int, default=1,
                    help='Switch for learning rate decay. Integer argument of 1 for ON and 0 for OFF, learning rate is decayed to zero over the total number of updates')
parser.add_argument('--prime', type=str, default=None,
                    help='Prime the network with some bytes')

args = parser.parse_args()

rnn_size = args.rnn_size
batch_size = args.batch_size
seq_length = args.seq_length
embedding_size = args.embedding_size

# preprocess the input data and create batches
data_dir =args.data_dir

loader = DataLoader(data_dir, batch_size, seq_length)

vocabulary_size = 256 # byte

graph = tf.Graph()

# initialiser used for all parameters
initializer = tf.glorot_normal_initializer()

with graph.as_default():

    # define all of the model variables
    W_embedding = tf.get_variable('W_embedding', shape=(vocabulary_size, embedding_size), initializer=initializer)

    # mt = (Wmxxt) ⊙ (Wmhht−1) - equation 18
    Wmx = tf.get_variable('Wmx', shape=(embedding_size, rnn_size), initializer=initializer)
    Wmh = tf.get_variable('Wmh', shape=(rnn_size, rnn_size), initializer=initializer )

    # hˆt = Whxxt + Whmmt
    Whx = tf.get_variable('Whx', shape=(embedding_size, rnn_size), initializer=initializer)
    Whm = tf.get_variable('Whm', shape=(rnn_size,rnn_size), initializer=initializer)
    Whb = tf.get_variable('Whb', shape=(1, rnn_size), initializer=initializer)

    # it = σ(Wixxt + Wimmt)
    Wix = tf.get_variable('Wix', shape=(embedding_size, rnn_size), initializer=initializer)
    Wim = tf.get_variable('Wim', shape=(rnn_size, rnn_size), initializer=initializer)
    Wib = tf.get_variable('Wib', shape=(1, rnn_size), initializer=initializer)

    # ot = σ(Woxxt + Wommt)
    Wox = tf.get_variable('Wox', shape=(embedding_size, rnn_size), initializer=initializer)
    Wom = tf.get_variable('Wom', shape=(rnn_size, rnn_size), initializer=initializer)
    Wob = tf.get_variable('Wob', shape=(1, rnn_size), initializer=initializer)

    # ft =σ(Wfxxt +Wfmmt)
    Wfx = tf.get_variable('Wfx', shape=(embedding_size, rnn_size),initializer=initializer)
    Wfm = tf.get_variable('Wfm', shape=(rnn_size, rnn_size), initializer=initializer)
    Wfb = tf.get_variable('Wfb', shape=(1, rnn_size), initializer=initializer)

    # define the g parameters for weight normalization if wn switch is on
    if args.wn == 1:

        gmx = tf.get_variable('gmx', shape=(rnn_size), initializer=initializer)
        gmh = tf.get_variable('gmh', shape=(rnn_size), initializer=initializer)

        ghx = tf.get_variable('ghx', shape=(rnn_size), initializer=initializer)
        ghm = tf.get_variable('ghm', shape=(rnn_size), initializer=initializer)

        gix = tf.get_variable('gix', shape=(rnn_size), initializer=initializer)
        gim = tf.get_variable('gim', shape=(rnn_size), initializer=initializer)

        gox = tf.get_variable('gox', shape=(rnn_size), initializer=initializer)
        gom = tf.get_variable('gom', shape=(rnn_size), initializer=initializer)

        gfx = tf.get_variable('gfx', shape=(rnn_size), initializer=initializer)
        gfm = tf.get_variable('gfm', shape=(rnn_size), initializer=initializer)


        # normalized weights
        Wmx = tf.nn.l2_normalize(Wmx, dim=0)*gmx
        Wmh = tf.nn.l2_normalize(Wmh, dim=0)*gmh

        Whx = tf.nn.l2_normalize(Whx,dim=0)*ghx
        Whm = tf.nn.l2_normalize(Whm,dim=0)*ghm

        Wix = tf.nn.l2_normalize(Wix,dim=0)*gix
        Wim = tf.nn.l2_normalize(Wim,dim=0)*gim

        Wox = tf.nn.l2_normalize(Wox,dim=0)*gox
        Wom = tf.nn.l2_normalize(Wom,dim=0)*gom

        Wfx = tf.nn.l2_normalize(Wfx,dim=0)*gfx
        Wfm = tf.nn.l2_normalize(Wfm,dim=0)*gfm

    # classifier weights and biases.
    w = tf.get_variable('Classifier_w', shape=(rnn_size, vocabulary_size), initializer=initializer)
    b = tf.get_variable('Classifier_b', shape=(vocabulary_size), initializer=initializer)

    # Variables for saving state across unrolled network.
    saved_output = tf.get_variable('saved_output', shape=(batch_size, rnn_size),initializer= tf.zeros_initializer(), trainable=False)
    saved_state = tf.get_variable('saved_state', shape=(batch_size, rnn_size),initializer= tf.zeros_initializer(), trainable=False)

    # placeholder for the inputs and the targets
    inputs = tf.placeholder(tf.int32, shape=[batch_size, seq_length],name='inputs')
    targets = tf.placeholder(tf.int32, shape=[batch_size, seq_length],name='targets') # targets for the lm

    # for the targets
    one_hot_labels = tf.one_hot(targets, vocabulary_size)
    labels_split_ = tf.split(one_hot_labels, seq_length, axis=1)
    list_labels = [tf.squeeze(input_, [1]) for input_ in labels_split_]

    # for the inputs
    embedded_inputs  = tf.nn.embedding_lookup(W_embedding,inputs)
    inputs_split_ = tf.split(embedded_inputs, seq_length, axis=1)
    list_inputs = [tf.squeeze(input_, [1]) for input_ in inputs_split_]

    def mlstm_cell(x, h, c):
        """
        multiplicative LSTM cell. https://arxiv.org/pdf/1609.07959.pdf

        """
        # mt = (Wmxxt) ⊙ (Wmhht−1) - equation 18
        mt = tf.matmul(x,Wmx) * tf.matmul(h,Wmh)
        # hˆt = Whxxt + Whmmt
        ht = tf.tanh(tf.matmul(x,Whx) + tf.matmul(mt,Whm) + Whb)
        # it = σ(Wixxt + Wimmt)
        it = tf.sigmoid(tf.matmul(x,Wix) + tf.matmul(mt,Wim)+ Wib)
        # ot = σ(Woxxt + Wommt)
        ot = tf.sigmoid(tf.matmul(x,Wox) + tf.matmul(mt,Wom)+ Wob)
        # ft =σ(Wfxxt +Wfmmt)
        ft = tf.sigmoid(tf.matmul(x,Wfx) + tf.matmul(mt,Wfm)+ Wfb)

        c_new = (ft * c) + (it * ht)

        h_new = tf.tanh(c_new) * ot

        return h_new, c_new

    # Unrolled LSTM loop.
    outputs = list()
    output = saved_output # these are initially zero
    state = saved_state
    for i in list_inputs:
        output, state = mlstm_cell(i, output, state)
        outputs.append(output)

    with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
        # Classifier.
        logits = tf.nn.xw_plus_b(tf.concat(outputs, 0), w, b)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.concat(list_labels, 0), logits=logits),name='loss')
        perplexity = tf.exp(loss)

    # Optimizer.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    init_lr = args.init_lr
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step, name= 'optimizer_op')

    # Predictions.
    train_prediction = tf.nn.softmax(logits)

    # Sampling code.
    sample_input = tf.placeholder(tf.int32, shape=(1,), name = 'sample_input')
    sample_embedding= tf.nn.embedding_lookup(W_embedding,sample_input)
    saved_sample_output = tf.Variable(tf.zeros([1, rnn_size]),name = 'saved_sample_output')
    saved_sample_state = tf.Variable(tf.zeros([1, rnn_size]),name = 'saved_sample_state')

    reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, rnn_size])), saved_sample_state.assign(tf.zeros([1, rnn_size])),name='reset_sample_state_op')

    sample_output, sample_state = mlstm_cell(sample_embedding, saved_sample_output, saved_sample_state)
    with tf.control_dependencies([saved_sample_output.assign(sample_output),saved_sample_state.assign(sample_state)]):
        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b), name = 'sample_prediction')

# Summaries for tensorboard
tf.summary.scalar('train_loss', loss)
tf.summary.scalar('perplexity', perplexity)
tf.summary.scalar('learning_rate', learning_rate)


with tf.Session(graph=graph) as session:

    saver = tf.train.Saver()

    # initialize variables before restoring from saved model
    tf.global_variables_initializer().run()
    print('Variables Initialized')

    if args.restore_path is not None:

        saver.restore(session, tf.train.latest_checkpoint(args.restore_path))
        print('Restored')

    summaries = tf.summary.merge_all()
    print('Summaries Merged')

    # timestamp for saving the run
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")

    # writer for saving tensorboard logs
    writer = tf.summary.FileWriter(os.path.join(args.log_dir, timestamp))
    writer.add_graph(session.graph)

    for epoch in xrange(args.num_epochs):

        loader.reset_batch_pointer()

        for batch in xrange(loader.num_batches):

            gs = session.run(global_step)
            start = time.time()
            x,y = loader.next_batch()

            if args.lr_decay == 1:

                # linearly decay the learning rate to zero over the number of updates
                lr = init_lr-(((gs)*init_lr)/ (args.num_epochs * loader.num_batches))

                _,l,perp,summary=session.run([optimizer, loss, perplexity, summaries], feed_dict={inputs:x, targets:y,learning_rate:lr})

            else:

                _,l,perp,summary=session.run([optimizer, loss, perplexity, summaries], feed_dict={inputs:x, targets:y,learning_rate:init_lr})

            end = time.time()

            print("Global step: {}, progress: ({}/{}), train_loss = {:.3f}, train_perplexity = {:.3f}, time/batch = {:.3f}, learning_rate = {}"
                .format(gs,epoch * loader.num_batches + batch,args.num_epochs * loader.num_batches,l, perp, end - start, lr))

            # write the summaries to the log_dir for tensorboard
            if args.summary_frequency != 0:
                if (batch % args.summary_frequency == 0):
                    start = time.time()
                    writer.add_summary(summary, gs)
                    end = time.time()
                    print('Writing Summaries...', 'time = ', end - start)

            # sample from the model
            if args.sampling_frequency !=0:

                if batch % args.sampling_frequency == 0:
                    start = time.time()
                    print('Sampling...')

                    sentence = bytearray()

                    if args.prime is not None:

                        # prime the network with a sequence of bytes
                        prime = bytearray(args.prime)
                        sentence += prime
                        for i in prime:
                            feed = np.array(i, ndmin=1)
                            prediction = session.run(sample_prediction, feed_dict = {sample_input: feed})

                    else:

                        # prime with a random byte
                        feed = np.array(random.sample(xrange(vocabulary_size),1), dtype='int32')
                        prediction = session.run(sample_prediction, feed_dict = {sample_input: feed})

                    for _ in xrange(args.num_chars):

                        # sequence is generated here
                        feed = np.expand_dims(np.random.choice(xrange(vocabulary_size), p=prediction.ravel()),axis=0)
                        sentence.append(int(feed))
                        prediction = session.run(sample_prediction, feed_dict = {sample_input: feed})

                    # decode the bytes to get unicode representation
                    sentence = sentence.decode('utf-8', errors='replace')

                    print(sentence)
                    end = time.time()

                    print('='*100)
                    print('Sampling time = ', end - start)

                    #save samples
                    sample_dir = os.path.join('sample_logs',timestamp)

                    if not os.path.exists(sample_dir):
                        os.makedirs(sample_dir)
                        sample_file = os.path.join(sample_dir,'samples')

                    with io.open(sample_file, 'a+', encoding='utf-8') as f:
                        f.write('\n' + 'GLOBAL_STEP: ' + str(gs) + '\n' + sentence)

    # save the fully trained model on the way out
    save_dir = os.path.join(args.saved_models_dir,timestamp)
    os.makedirs(save_dir)
    checkpoint_path = os.path.join(save_dir, 'model')
    saver.save(session, checkpoint_path, global_step=gs)
    print('Model Saved')
