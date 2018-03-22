from active_experiment_builder import ExperimentBuilder
import miniImagenet_data as dataset
import tqdm
import os
import tensorflow as tf

tf.reset_default_graph()

tf.logging.set_verbosity(tf.logging.INFO)
# Experiment Setup
batch_size = 5
ways = 5
shots = 5
query_size = 2
image_shape = [224, 224, 3]
restore = False
# active_round = 6
int_total_epochs = 10
total_training_episodes = 250
total_val_episodes = 250
total_test_episodes = 250
data_format = 'channels_last'

path = os.getcwd()
data_dir = path + '/miniImagenet/'

experiment_name = "few_shot_learning_embedding_{}_{}".format(shots, ways)

# Experiment builder
data = dataset.DiabeticData(image_shape= image_shape, batch_size=batch_size,
                                ways=ways, shots=shots, query_size = query_size)

experiment = ExperimentBuilder(data)

few_shot_miniImagenet, losses, ada_opts, init = experiment.build_experiment(batch_size = batch_size, ways = ways,
                                                                            shots = shots, query_size = query_size,
                                                                            image_shape = image_shape, data_format = data_format)
# define saver object for storing and retrieving checkpoints
saver = tf.train.Saver()
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

save_path = os.path.join(path, 'saved_models') # path for the checkpoint file

if not os.path.exists('summary_dir'):
    os.makedirs('summary_dir')

if not os.path.exists('summary_dir/tra_log'):
    os.makedirs('summary_dir/tra_log')


if not os.path.exists('summary_dir/test_log'):
    os.makedirs('summary_dir/test_log')

tra_log_path = os.path.join(path, 'summary_dir/tra_log')
test_log_path = os.path.join(path, 'summary_dir/test_log')


# Experiment initialization and running
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(tra_log_path, sess.graph)
    # val_writer = tf.summary.FileWriter(summary_path + '/val')
    test_writer = tf.summary.FileWriter(test_log_path, sess.graph)
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

    # for r in range(active_round):
    #     print("active round: ", r)
    #     total_epochs = total_training_episodes + 2*r
        best_val = 0.
        total_epochs = int_total_epochs
        with tqdm.tqdm(total=total_epochs) as pbar_e:

            for e in range(0, total_epochs):
                total_c_loss, total_accuracy = experiment.run_training_epoch(total_training_episodes=total_training_episodes, writer = train_writer,
                                                                             sess=sess)
                print('\n')
                print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))

                total_val_c_loss, total_val_accuracy = experiment.run_validation_epoch(total_val_episodes=total_val_episodes,
                                                                                       sess=sess)
                print('\n')
                print("Epoch {}: val_loss: {}, val_accuracy: {}".format(e, total_val_c_loss, total_val_accuracy))

                if total_val_accuracy >= best_val: #if new best val accuracy -> produce test statistics
                    best_val = total_val_accuracy
                    total_test_c_loss, total_test_accuracy = experiment.run_testing_epoch(
                                                                        total_test_episodes=total_test_episodes, sess=sess, writer = test_writer)
                    print('\n')
                    print("Epoch {}: test_loss: {}, test_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))

                    save_path = saver.save(sess, "saved_models/{}_{}.ckpt".format(experiment_name, e))
                else:
                    total_test_c_loss = -1
                    total_test_accuracy = -1

                pbar_e.update(1)

        # most_uncertain_imgs = experiment.run_find_cand_epoch(sess = sess)
        # print(most_uncertain_imgs)
