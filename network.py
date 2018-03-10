import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
_DROPOUT_RATE = 0.5

def batch_norm_relu(inputs, is_training, data_format):
    """
    performs a batch normalization followed by a relu
    """
    print('data_format: ', data_format)
    print('inputs_shape: ', inputs.get_shape())
    inputs = tf.layers.batch_normalization(
        inputs = inputs, axis = 1 if data_format == 'channels_first' else -1,
        momentum = _BATCH_NORM_DECAY, epsilon = _BATCH_NORM_EPSILON, center = True,
        scale = True, training = is_training, fused = True)

    inputs = tf.nn.relu(inputs)
    return inputs

def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
    Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                [pad_beg, pad_end], [0, 0]])
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    #TODO try orthogonal parameter matrix initialization
    return tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(), data_format=data_format)

def building_block(inputs, filters, is_training, projection_shortcut, strides, data_format):
    """standard building block for residual networks with batch normalization before convolutions
    
    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        The output tensor of the block.
    """


    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training, data_format)


    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)

    inputs = tf.layers.dropout(inputs = inputs, rate = _DROPOUT_RATE)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=1,
        data_format=data_format)

    return inputs + shortcut

def bottleneck_block(inputs, filters, is_training, projection_shortcut, strides, data_format):
    """Bottleneck block variant for residual networks with BN before convolutions.
    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first two convolutions. Note that the
      third and final convolution will use 4 times as many filters.
    is_training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').
    Returns:
    The output tensor of the block.
    """
    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training, data_format)

    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=1, strides=1,
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)

    inputs = tf.layers.dropout(inputs = inputs, rate = _DROPOUT_RATE)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=filters, kernel_size=3, strides=strides,
        data_format=data_format)

    inputs = batch_norm_relu(inputs, is_training, data_format)

    inputs = tf.layers.dropout(inputs = inputs, rate = _DROPOUT_RATE)

    inputs = conv2d_fixed_padding(
        inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
        data_format=data_format)

    return inputs + shortcut

def block_layer(inputs, filters, block_fn, blocks, strides, is_training, name, data_format):
    """Creates one layer of blocks for the ResNet model.
    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    is_training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').
    Returns:
    The output tensor of the block layer.
    """
    # Bottleneck blocks end with 4x the number of filters as they start with
    filters_out = 4 * filters if block_fn is bottleneck_block else filters

    def projection_shortcut(inputs):
        return conv2d_fixed_padding(
            inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
            data_format=data_format)

    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, is_training, projection_shortcut, strides,
                        data_format)

    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, is_training, None, 1, data_format)

    return tf.identity(inputs, name)

def embeddingResNet_generator(block_fn, layers, data_format = None):
    """generator for miniImagenet ResNet v2 models

    Args:
        resnet_size: A single integer for the size of the ResNet model.
        num_classes: The number of possible classes for image classification.
        data_format: The input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
        Returns:
        The model function that takes in `inputs` and `is_training` and
        returns the output tensor of the ResNet model.
        Raises:
        ValueError: If `resnet_size` is invalid.
    """


    if data_format is None:
        data_format = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'


    def model(inputs, is_training):
        """constructs the ResNet model given the inputs"""


        if data_format == 'channels_first':
            # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
            # This provides a large performance boost on GPU. See
            # https://www.tensorflow.org/performance/performance_guide#data_formats
            inputs = tf.transpose(inputs, [0, 3, 1, 2])


        #localize network to generate the transformation parameters
        # raw_inputs = inputs

        # inputs = tf.layers.conv2d(inputs = inputs, filters = 32, strides = 2, kernel_size = 5, padding = 'SAME', kernel_initializer=tf.variance_scaling_initializer())

        # print(inputs.shape)
        # inputs = tf.layers.max_pooling2d(inputs = inputs, pool_size = 2, strides = 2, padding = 'VALID')
        # print(inputs.shape)
        # inputs = tf.layers.conv2d(inputs = inputs, filters = 64, strides = 2, kernel_size = 5, padding = 'SAME', kernel_initializer = tf.variance_scaling_initializer())
        # print(inputs.shape)
        # inputs = tf.layers.max_pooling2d(inputs = inputs, pool_size = 2, strides = 2, padding = 'VALID')
        # print(inputs.shape)
        # inputs = tf.layers.dropout(inputs = inputs, rate = _DROPOUT_RATE)

        # inputs = tf.layers.flatten(inputs = inputs)

        # inputs = tf.layers.dense(inputs = inputs, units = 128)
        # print(inputs.shape)
        # trans_parameters = tf.layers.dense(inputs = inputs, units = 6)
        # print(trans_parameters.shape)
        # inputs = stn(input_fmap = raw_inputs, theta = trans_parameters, out_dims = [60, 60])


        print('embedding_init_shape', inputs.shape)
        #embedding network
        inputs = conv2d_fixed_padding(inputs = inputs, filters = 64, kernel_size = 7, strides = 2, data_format = data_format)

        # print('height:', inputs.shape)
        inputs = tf.identity(inputs, 'initial_conv')

        inputs = tf.layers.max_pooling2d(inputs = inputs, pool_size = 3, strides = 2, padding = 'SAME', data_format = data_format)


        inputs = tf.identity(inputs, 'initial_max_pool')

        inputs = block_layer(inputs = inputs, filters = 64, block_fn = block_fn, blocks = layers[0], strides = 1, 
            is_training = is_training, name = 'blcok_layer1', data_format = data_format)


        # #attention module
        # input_fmap = inputs
        # # inputs = tf.reshape(inputs, (-1, 64))
        # inputs = tf.layers.dense(inputs = inputs, units = 32, activation = tf.tanh)

        # inputs = tf.reshape(inputs, [-1, 32])
        # inputs = tf.layers.dense(inputs = inputs, units = 1, activation = tf.sigmoid)

        # attention_para = tf.reshape(inputs, [-1, 21, 21, 1])


        # inputs = tf.multiply(input_fmap, attention_para)

        inputs = block_layer(inputs = inputs, filters = 128, block_fn = block_fn, blocks = layers[1], strides = 2,
            is_training = is_training, name = 'block_layer2', data_format = data_format)

        inputs = block_layer(inputs = inputs, filters = 256, block_fn = block_fn, blocks = layers[2], strides = 2, 
            is_training = is_training, name = 'block_layer3', data_format = data_format)

        inputs = block_layer(inputs = inputs, filters = 512, block_fn = block_fn, blocks = layers[3], strides = 2,
            is_training = is_training, name = 'block_layer4', data_format = data_format)

        print('embedding_final_shape:', inputs.shape)

        embedding = batch_norm_relu(inputs, is_training, data_format)
        

        return embedding

    return model

#define the network structure, return a model
def comparisonNet_generator(block_fn, data_format = None):

    if data_format == None:
        data_format = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'

    def model(inputs, is_training):

        inputs = block_layer(inputs = inputs, filters = 128, block_fn = block_fn,
                                    blocks = 1, strides = 2, is_training = is_training, name = 'compare_1', data_format = data_format)

        inputs = block_layer(inputs = inputs, filters = 64, block_fn = block_fn,
                                    blocks = 1, strides = 2, is_training = is_training, name = 'compare_2', data_format = data_format) 

        inputs = tf.layers.average_pooling2d(inputs = inputs, pool_size = 2, strides = 2, data_format = data_format, name = 'final_pool')


        print("final_pool: ", inputs.shape)
        inputs = tf.layers.dense(inputs, units = 1, name = 'final_dense', activation = tf.sigmoid)

        print("final_activation: ", inputs.shape)
        output = tf.squeeze(inputs)
        print("final_output: ", output.shape)
        return output

    return model

class FewshotsNet:
    """"""
    def __init__(self, support_set_images, support_set_labels, query_images, query_labels, data_format,
                    is_training, keep_prob=0.5, batch_size=16, learning_rate=0.01):


        """
        Builds a comparison network, the training and evaluation ops as well as data augmentation routines.
        :param support_set_images: A tensor containing the support set images [batch_size, sequence_size, 28, 28, 1]
        :param support_set_labels: A tensor containing the support set labels [batch_size, sequence_size, 1]
        :param query_images: A tensor containing the target image (image to produce label for) [batch_size, 28, 28, 1]
        :param query_labels: A tensor containing the target label [batch_size, query_size]
        :param keep_prob: A tf placeholder of type tf.float32 denotes the amount of dropout to be used
        :param batch_size: The batch size for the experiment
        :param num_channels: Number of channels of the images
        :param is_training: Flag indicating whether we are training or evaluating
        :param num_classes_per_set: Integer indicating the number of classes per set
        :param num_samples_per_class: Integer indicating the number of samples per class
        """
        self.batch_size = batch_size
        self.e = embeddingResNet_generator(block_fn = building_block, layers = [2,2,2,2], data_format = data_format)
        self.c = comparisonNet_generator(block_fn = building_block, data_format = data_format)
        self.support_set_images = support_set_images
        self.support_set_labels = support_set_labels
        self.query_images = query_images
        self.query_labels = query_labels
        self.keep_prob = keep_prob
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.data_format = data_format


    def loss(self):
        """
        Builds tf graph for Fewshots Networks, produces losses and summary statistics.
        :return:
        """

        with tf.name_scope('losses'):
            [b, num_classes, num_samples_per_class, h, w, c] = self.support_set_images.get_shape().as_list()
            [b, query_size] = self.query_labels.get_shape().as_list()
            self.support_set_images = tf.reshape(self.support_set_images, shape = [b*num_classes* num_samples_per_class, h, w, c])
            self.support_set_labels = tf.reshape(self.support_set_labels, shape = [b,num_classes* num_samples_per_class])
            # self.support_set_labels = tf.one_hot(self.support_set_labels, num_samples_per_class)
            # self.query_labels = tf.one_hot(self.query_labels, num_samples_per_class)
            # support_images_embeddings = []
            # query_images_embeddings = []
            # for image in tf.unstack(self.support_set_images, axis = 0):
            #     image_embedding = self.e(inputs = image, is_training = self.is_training)
            #     support_images_embeddings.append(image_embedding)
            support_images_embeddings = self.e(inputs = self.support_set_images, is_training=self.is_training)
            support_images_embeddings = tf.reshape(support_images_embeddings, shape=[b, num_classes*num_samples_per_class, 7, 7, 512])
            image_shape_after_embedding = support_images_embeddings[0].shape

            #support_images_embeddings = tf.stack(support_images_embeddings, axis = 1)
            #shape of support_images_embeddings: [batch_size, num_classes* num_samples_per_class, d, d, c]
            support_images_embeddings = tf.expand_dims(support_images_embeddings, axis = 2)
            #shape of support_images_embeddings: [batch_size, num_classes* num_samples_per_class,1, d, d, c]
            support_images_embeddings = tf.tile(support_images_embeddings, [1, 1, query_size, 1, 1, 1])
            #shape of support_images_embeddings: [batch_size, num_classes* num_samples_per_class, query_size, d, d, c]
            self.query_images = tf.reshape(self.query_images, shape=[b*query_size, h, w, c])
            # for image in tf.unstack(self.query_images, axis = 0):
            #     image_embedding = self.e(inputs = image, is_training = self.is_training)
            #     query_images_embeddings.append(image_embedding)
            query_images_embeddings = self.e(inputs = self.query_images, is_training=self.is_training)
            query_images_embeddings = tf.reshape(query_images_embeddings, shape=[b, query_size, 7, 7, 512])
            #query_images_embeddings = tf.stack(query_images_embeddings, axis = 1)
            #shape of query_images_embeddings: [batch_size, query_size, d, d, c]
            query_images_embeddings = tf.expand_dims(query_images_embeddings, axis = 1)
            #shape of query_images_embeddings: [batch_size, 1, query_size, d, d, c]
            query_images_embeddings = tf.tile(query_images_embeddings, [1, num_classes*num_samples_per_class, 1, 1, 1, 1])
            #shape of query_images_embeddings: [batch_size, num_classes* num_samples_per_class, query_size, d, d, c]


            inputs = tf.concat(values = [support_images_embeddings, query_images_embeddings], axis = -1)
            #shape of inputs: [batch_size, num_classes* num_samples_per_class, query_size, d, d, 2*c]


            inputs = tf.reshape(inputs, shape = [self.batch_size* num_classes*num_samples_per_class*query_size, image_shape_after_embedding[1], image_shape_after_embedding[2], -1])
            #shape of inputs: [batch_size*num_classes* nuimage_shape_after_embeddingm_samples_per_class*query_size, d, d, 2*c]
            print("input_shape_for_comparison", inputs.shape)

            scores = self.c(inputs= inputs, is_training= self.is_training)

            scores = tf.reshape(scores, shape = [self.batch_size, num_classes* num_samples_per_class, query_size])

            support_set_labels = tf.expand_dims(self.support_set_labels, axis = 2)

            support_set_labels = tf.tile(support_set_labels, [1, 1, query_size])
            print("support_set_labels shape", support_set_labels.get_shape())

            query_labels = tf.expand_dims(self.query_labels, axis = 1)
            query_labels = tf.tile(query_labels, [1, num_classes*num_samples_per_class, 1])
            print("query_labels shape", query_labels.get_shape())
            #shape of both support_set_labels and query_labels: [batch_size, num_classes* num_samples_per_class, query_size]

            labels = tf.equal(support_set_labels, query_labels, name = 'labels')
            print("final labels shape", labels.get_shape())

            msq_losses = tf.reduce_sum(tf.pow(tf.cast(labels, tf.float16) - tf.cast(scores, tf.float16), 2)) / self.batch_size
            msq_losses = tf.identity(msq_losses, 'loss')
            tf.summary.scalar('loss', msq_losses)

            scores = tf.reshape(scores, shape=[self.batch_size, num_classes, num_samples_per_class, query_size])
            prediction = tf.argmax(tf.reduce_sum(scores, axis = 2), axis = 1)
            correct_prediction = tf.equal(prediction, tf.cast(self.query_labels, tf.int64))
            accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))/ (self.batch_size * query_size)
            accuracy = tf.identity(accuracy, 'accuracy')
            tf.summary.scalar('accuracy', accuracy)

            return {'loss': msq_losses, 'accuracy': accuracy}



    def train(self, losses):
        """
        builds the train op
        :param losses: A dictionary contains the losses
        :param learning_rate: learning rate for sgd
        :return 
        """
        opt = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #Needed for correct batch norm usage

        with tf.control_dependencies(update_ops):
            # train_variables = self.e.variables + self.c.variables
            ada_opt = opt.minimize(losses['loss'])

        return ada_opt


    def init_train(self):
        """
        get all ops and losses
        :return 
        """

        losses = self.loss()
        ada_opt = self.train(losses)
        summary = tf.summary.merge_all()
        return summary, losses, ada_opt
