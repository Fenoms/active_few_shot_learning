import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from PIL import ImageEnhance
import os
import glob
from scipy import misc

path = os.getcwd()

def _read_image_as_array(image, dtype='float32'):
    f = Image.open(image)
    # k = np.random.randint(0, 4)
    # f.rotate(k*90)
    # f = random_brightness(f, [0.8,1])
    try:
        # image = np.asarray(f, dtype=dtype)
        image = tf.keras.preprocessing.image.img_to_array(f)
        # image = augment_image(image)
    finally:
        if hasattr(f, 'close'):
            f.close()
    return image

def augment_image(img):

    img = tf.keras.preprocessing.image.random_rotation(img, 20, row_axis=0, col_axis=1, channel_axis=2)

    img = tf.keras.preprocessing.image.random_shear(img, 0.2, row_axis=0, col_axis=1, channel_axis=2)

    img = tf.keras.preprocessing.image.random_shift(img, 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2)

    img = tf.keras.preprocessing.image.random_zoom(img, [0.9, 0.9], row_axis=0, col_axis=1, channel_axis=2)

    return img


# class can be used to control the brightness of an image.An enhancement factor of 0.0
# gives a black image.A factor of 1.0 gives the original image.
def random_brightness(img, brightness_range):

    imgenhancer_Brightness = ImageEnhance.Brightness(img)

    u = np.random.uniform(brightness_range[0], brightness_range[1])

    img = imgenhancer_Brightness.enhance(u)

    return img


class MiniImagenetData():

    def __init__(self, data_dir, batch_size, image_shape,ways = 5, shots = 5, query_size = 15):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.ways = ways
        self.shots = shots
        self.query_size = query_size
        #image shape info
        self.data_shape = image_shape

    def get_batch(self, mode = 'tra'):
        
        """
        """
        folder = path + '/miniImagenet/' + mode + '_data/'
        csv_folder = path + '/miniImagenet/csv/' + mode + '.csv'
        dataset = pd.read_csv(csv_folder, sep=',')
        labels = dataset.label.unique().tolist()

        support_set_x = np.zeros((self.batch_size, self.ways, self.shots, self.data_shape[0], 
                                    self.data_shape[1], self.data_shape[2]), dtype=np.float32)

        support_set_y = np.zeros((self.batch_size, self.ways, self.shots), dtype=np.float32)

        query_x = np.zeros((self.batch_size, self.query_size, self.data_shape[0], self.data_shape[1], self.data_shape[2]), dtype=np.float32)
        query_y = np.zeros((self.batch_size, self.query_size), dtype=np.float32)


        for i in range(self.batch_size):
            
            # if mode == 'train':
            #     c = np.arange(0, 64)
            # elif mode == 'val':
            #     c = np.arange(64, 80)
            # else:
            #     c = np.arange(80, 100)

            # sampled_classes = np.random.choice(c, self.ways, replace = False)

            # t = np.random.randint(0, self.ways)

            # sampled_classes = _sample_classes(data.shape[0])
            sampled_classes = np.random.choice(labels, self.ways, replace=False)

            for k, class_ in enumerate(sampled_classes):
                #the number of total images, in most case: 600
                # data_dir = path + '/miniImagenet' + '/' + mode + '/' + str(class_) + '/' + mode + '.npy'
                # data = np.load(data_dir)
                # shots_idx = data[class_].shape[0]

                #average num_samples for query image set
                # startidx = np.random.randint(0, 5, size=1)
                img_folder = folder + class_
                files = glob.glob(img_folder + '/*.png')

                q = int(self.query_size / self.ways)
                sample_imgs = np.random.choice(files, self.shots + q, replace=False)
                for j, img in enumerate(sample_imgs):

                    f = _read_image_as_array(img)

                    if j < self.shots:
                        support_set_x[i][k][j] = f
                        support_set_y[i][k] = k
                    else:
                        query_x[i][k*q + j - self.shots] = f
                        query_y[i][k*q + j - self.shots] = k


                # support_set_x = np.reshape(support_set_x,newshape=[self.batch_size, self.ways*self.shots,
                #                                                    self.data_shape[0], self.data_shape[1], self.data_shape[2]])
                # support_set_y = np.reshape(support_set_y, newshape=[self.batch_size, self.ways*self.shots])

                p_s = np.random.permutation(support_set_x.shape[1])
                np.take(support_set_x, p_s, axis=1, out=support_set_x)
                np.take(support_set_y, p_s, axis=1, out=support_set_y)

                p_q = np.random.permutation(query_x.shape[1])
                np.take(query_x, p_q, axis=1, out=query_x)
                np.take(query_y, p_q, axis=1, out=query_y)

                # support_set_x = np.reshape(support_set_x, newshape=[self.batch_size, self.ways , self.shots,
                #                                                     self.data_shape[0], self.data_shape[1], self.data_shape[2]])
                # support_set_y = np.reshape(support_set_y, newshape=[self.batch_size, self.ways, self.shots])

        return support_set_x, support_set_y, query_x, query_y



