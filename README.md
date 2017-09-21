# Tensorflow_mLSTM
Train a mLSTM language model in Tensorflow

The script train_mLSTM.py can be used to train a Multiplicative LSTM [Multiplicative LSTM for sequence modelling](https://arxiv.org/abs/1609.07959) language model in Tensorflow. The model is a byte-level mLSTM implemented using the techniques used in this paper —> [Generating Reviews and Discovering Sentiment](https://arxiv.org/abs/1704.01444).

# Description:

The script trains an mLSTM model using the selected hyperparameter settings. The fully trained model is saved at the end of the script in a timestamped subfolder in the [saved_models_dir] directory so that the model can later be restored for further use (e.g further training, inference). The model also logs the loss, perplexity and learning rate in a timestamped subfolder in the [tensorboard_logs] directory at specified intervals during training. The model also samples from the model during training, at specified intervals, saving the sampled text also in a timestamped subdirectory in the [sample_logs] directory.

# To Run:

To train a model, run the following line of python code. The path/to/directory must be a folder containing a text file. 

```bash
python train_mLSTM.py --data_dir=path/to/directory
```

To specify the the hyperparameters and other settings for training the model, see the description of python argparse options below.

# Argparse Options:

The following model hyperparameters can be set using argparse:

--[data_dir] - Path to the directory containing the training data. This directory must contain a text file as training data.
--[saved_models_dir] - Name of the directory to save the model during training. A directory is created in the current working directory with this name. After training, the saved model is saved in a timestamped directory inside the [saved_models_dir] folder. 

--[log_dir] - Name of the directory to save the logs during training (loss, learning rate, perplexity). The files saved here can be read by the Tensorboard tool to visualize this data. To use the Tensorboard tool, use the following command specifying the [logdir] argument for Tensorboard:

```bash
tensorboard --logdir=path/to/[log_dir]
```

--[rnn-size] - Argument to specify the hidden state size of the mLSTM. In the paper this is 4096 (very big!). In this script the default value is 128. 

--[seq_length] - Specify the sequence length parameter for training the model. In the paper this is 256. In this script the default value is 256.

--[batch_size] - Mini-batch size. In the paper this is 128. In this script the default value is 32. 

--[num_epochs] - Number of training epochs. Default value is 1.

--[init_lr] - The initial learning rate used for training. The default value here is 5*10^-4 as in the paper. The optimization algorithm used here is ADAM. If the [lr_decay] option is selected the learning rate is decayed linearly to zero as in the paper, otherwise it remains static throughout training. Pass an integer argument of 1 to enable or 0 to disable. Enabled by default.

--[wn] - Switch for weight normalization. In the paper the weight normalization technique detailed here (Weight Normalization: A Simple Reparameterisation to Accelerate Training of Deep Neural Networks) is used. Pass an integer argument of 1 to enable or 0 to disable. Weight normalization is enabled by default. 

--[restore_path] - To restore a saved model from a previous session, specify the path to the saved model. This will be a folder from a previous run inside the [saved_models_dir]. The most recently saved model will be restored. The folder will contain 4 files:

checkpoint file
[your-model].data
[your-model].index
[your-model].meta

--[summary_frequency] - This specifies how often you want to log data for tensorboard. The model will log data every N updates according to this value.

--[sampling_frequency] - This specifies how often you want to generate samples from the model during training. The model will sample after N updates depending on this value, the default value is 1000. The sampling process is primed by generating one random value to feed the model, then sampling the generated output and feeding it back in. The amount of characters generated is specified by the [num_chars] argument which has a default of 250. The generated samples are printed to the terminal and also saved inside a timestamped directory inside the [sample_logs] folder which is created automatically. 

--[num_chars] - This specifies how many characters to generate if the [sampling_frequency] argument is more than zero.

--[lr_decay] - Switch for learning rate decay which is used in the paper. If selected, the learning rate is decayed linearly to zero over the course of training. Pass an integer argument of 1 to enable or 0 to disable. Enabled by default. 

# Requirements:

This code was created using Tensorflow r1.2 and python 2.7
# And...
Thank you, please feel free to leave feedback!


