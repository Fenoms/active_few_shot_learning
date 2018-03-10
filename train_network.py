import tensorflow as tf
from network import *
from experiment_builder import ExperimentBuilder
import tensorflow.contrib.slim as slim
import miniImagenetdata as dataset
import tqdm
import os

# tf.reset_default_graph()

# Experiment Setup
batch_size = 10
ways = 5
shots = 5
query_size = 15
image_shape = [224, 224, 3]
restore = False
total_epochs = 50
total_training_episodes = 1000
total_val_episodes = 250
total_test_episodes = 250
data_format = 'channels_last'

path = os.getcwd()
data_dir = path + '/miniImagenet/'

experiment_name = "few_shot_learning_embedding_{}_{}".format(shots, ways)

# Experiment builder
data = dataset.MiniImagenetData(data_dir = data_dir, image_shape= image_shape, batch_size=batch_size, ways=ways, shots=shots, query_size = query_size)

experiment = ExperimentBuilder(data)

few_shot_miniImagenet, losses, ada_opts, init = experiment.build_experiment(batch_size = batch_size, ways = ways, 
                                                                                shots = shots, query_size = query_size, 
                                                                                    image_shape = image_shape, data_format = data_format)

# define saver object for storing and retrieving checkpoints
saver = tf.train.Saver()
if not os.path.exists('save_dir'):
    os.makedirs('save_dir')

save_path = os.path.join(path, 'save_dir/best_validation') # path for the checkpoint file

if not os.path.exists('logs_dir'):
    os.makedirs('logs_dir')

logs_path = os.path.join(path, 'logs_dir/logs')

# Experiment initialization and running
with tf.Session() as sess:
    if restore:
        try:
            print("tring to restore last checkpoint...")
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir = save_path)
            saver.restore(sess, save_path = last_chk_path)
            print("restored checkpoint from: ", last_chk_path)
        except:
            print("failed to restore checkpoint.")
            sess.run(init)
    else:
        sess.run(init)
    # if restore_flag == True: #load checkpoint if needed
    #     checkpoint = "saved_models/{}_{}.ckpt".format(experiment_name, restore_flag)
    #     variables_to_restore = []
    #     for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #         print(var)
    #         variables_to_restore.append(var)

    #     tf.logging.info('Fine-tuning from %s' % checkpoint)

    #     fine_tune = slim.assign_from_checkpoint_fn(
    #         checkpoint,
    #         variables_to_restore,
    #         ignore_missing_vars=True)
    #     fine_tune(sess)

    best_val = 0.
    with tqdm.tqdm(total=total_epochs) as pbar_e:
        for e in range(0, total_epochs):
            total_c_loss, total_accuracy = experiment.run_training_epoch(total_training_episodes=total_training_episodes,
                                                                                sess=sess)
            print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))

            total_val_c_loss, total_val_accuracy = experiment.run_validation_epoch(total_val_episodes=total_val_episodes,
                                                                                        sess=sess)
            print("Epoch {}: val_loss: {}, val_accuracy: {}".format(e, total_val_c_loss, total_val_accuracy))

            if total_val_accuracy >= best_val: #if new best val accuracy -> produce test statistics
                best_val = total_val_accuracy
                total_test_c_loss, total_test_accuracy = experiment.run_testing_epoch(
                                                                    total_test_episodes=total_test_episodes, sess=sess)
                print("Epoch {}: test_loss: {}, test_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))
            else:
                total_test_c_loss = -1
                total_test_accuracy = -1

            # save_statistics(experiment_name,
            #                 [e, total_c_loss, total_accuracy, total_val_c_loss, total_val_accuracy, total_test_c_loss,
            #                  total_test_accuracy])

            save_path = saver.save(sess, "saved_models/{}_{}.ckpt".format(experiment_name, e))
            pbar_e.update(1)
