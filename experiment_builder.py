import tensorflow as tf
import tqdm
from network import FewshotsNet


class ExperimentBuilder:

    def __init__(self, data):
        """
        Initializes an ExperimentBuilder object. The ExperimentBuilder object takes care of setting up our experiment
        and provides helper functions such as run_training_epoch and run_validation_epoch to simplify out training
        and evaluation procedures.
        :param data: A data provider class
        """
        self.data = data

    def build_experiment(self, batch_size, ways, shots, query_size, image_shape, data_format):

        """

        :param batch_size: The experiment batch size
        :param ways: An integer indicating the number of classes per support set
        :param shots: An integer indicating the number of samples per class
        :param channels: The image channels
        :param fce: Whether to use full context embeddings or not
        :return: a few_shot_learning object, along with the losses, the training ops and the init op
        """
        height, width, channels = image_shape
        self.support_set_x = tf.placeholder(tf.float32, shape = [batch_size, ways, shots, height, width,
                                                              channels], name = 'support_set_images')
        self.support_set_y = tf.placeholder(tf.uint8, shape = [batch_size, ways, shots], name = 'support_set_labels')
        self.query_x = tf.placeholder(tf.float32, shape = [batch_size, query_size, height, width, channels], name = 'query_images')
        self.query_y = tf.placeholder(tf.uint8, shape = [batch_size, query_size], name = 'query_labels')
        self.is_training = tf.placeholder(tf.bool, name='training_flag')
        #self.rotate_flag = tf.placeholder(tf.bool, name='rotate-flag')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_prob')
        self.current_learning_rate = 0.1
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.few_shot_miniImagenet = FewshotsNet(batch_size=batch_size, support_set_images=self.support_set_x,
                                            support_set_labels=self.support_set_y,
                                            query_images=self.query_x, query_labels=self.query_y, data_format = data_format,
                                            keep_prob=self.keep_prob,
                                            is_training=self.is_training, learning_rate=self.learning_rate)

        summary, self.losses, self.ada_opts = self.few_shot_miniImagenet.init_train()
        init = tf.global_variables_initializer()
        self.total_train_iter = 0
        return self.few_shot_miniImagenet, self.losses, self.ada_opts, init

    def run_training_epoch(self, total_training_episodes, sess):
        """
        Runs one training epoch
        :param total_train_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_training_categorical_crossentropy_loss and mean_training_accuracy
        """
        total_c_loss = 0.
        total_accuracy = 0.
        with tqdm.tqdm(total=total_training_episodes) as pbar:

            for i in range(total_training_episodes):  # train epoch
                support_set_x, support_set_y, query_x, query_y = self.data.get_batch('tra')
                _, c_loss_value, acc = sess.run(
                    [self.ada_opts, self.losses['loss'], self.losses['accuracy']],
                    feed_dict={self.keep_prob: 0.5, self.support_set_x: support_set_x,
                               self.support_set_y: support_set_y, self.query_x: query_x, self.query_y: query_y,
                               self.is_training: True, self.learning_rate: self.current_learning_rate})

                iter_out = "train_loss: {}, train_accuracy: {}".format(c_loss_value, acc)
                pbar.set_description(iter_out)

                pbar.update(1)
                total_c_loss += c_loss_value
                total_accuracy += acc
                self.total_train_iter += 1
                if self.total_train_iter % 2000 == 0:
                    self.current_learning_rate /= 2
                    print("change learning rate", self.current_learning_rate)

        total_c_loss = total_c_loss / total_training_episodes
        total_accuracy = total_accuracy / total_training_episodes
        return total_c_loss, total_accuracy

    def run_validation_epoch(self, total_val_episodes, sess):
        """
        Runs one validation epoch
        :param total_val_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_validation_categorical_crossentropy_loss and mean_validation_accuracy
        """
        total_val_c_loss = 0.
        total_val_accuracy = 0.

        with tqdm.tqdm(total=total_val_episodes) as pbar:
            for i in range(total_val_episodes):  # validation epoch
                support_set_x, support_set_y, query_x, query_y = self.data.get_batch("val")
                c_loss_value, acc = sess.run(
                    [self.losses['loss'], self.losses['accuracy']],
                    feed_dict={self.keep_prob: 1.0, self.support_set_x: support_set_x,
                               self.support_set_y: support_set_y, self.query_x: query_x, self.query_y: query_y,
                               self.is_training: False})

                iter_out = "val_loss: {}, val_accuracy: {}".format(c_loss_value, acc)
                pbar.set_description(iter_out)
                pbar.update(1)

                total_val_c_loss += c_loss_value
                total_val_accuracy += acc

        total_val_c_loss = total_val_c_loss / total_val_episodes
        total_val_accuracy = total_val_accuracy / total_val_episodes

        return total_val_c_loss, total_val_accuracy

    def run_testing_epoch(self, total_test_episodes, sess):
        """
        Runs one testing epoch
        :param total_test_batches: Number of batches to train on
        :param sess: Session object
        :return: mean_testing_categorical_crossentropy_loss and mean_testing_accuracy
        """
        total_test_c_loss = 0.
        total_test_accuracy = 0.
        with tqdm.tqdm(total=total_test_episodes) as pbar:
            for i in range(total_test_episodes):
                support_set_x, support_set_y, query_x, query_y = self.data.get_batch("test")
                c_loss_value, acc = sess.run(
                    [self.losses['loss'], self.losses['accuracy']],
                    feed_dict={self.keep_prob: 1.0, self.support_set_x: support_set_x,
                               self.support_set_y: support_set_y, self.query_x: query_x,
                               self.query_y: query_y,
                               self.is_training: False})

                iter_out = "test_loss: {}, test_accuracy: {}".format(c_loss_value, acc)
                pbar.set_description(iter_out)
                pbar.update(1)

                total_test_c_loss += c_loss_value
                total_test_accuracy += acc
            total_test_c_loss = total_test_c_loss / total_test_episodes
            total_test_accuracy = total_test_accuracy / total_test_episodes
        return total_test_c_loss, total_test_accuracy
