import os
import numpy as np
import glob
from PIL import Image

path = os.getcwd()
data_dir = path + '/miniImagenet/'

def _read_image_as_array(image, dtype='float32'):
    f = Image.open(image)
    # k = np.random.randint(0, 4)
    # f.rotate(k*90)
    # f = random_brightness(f, [0.8,1])
    try:
        image = np.asarray(f, dtype=dtype)
        # image = augment_image(image)
    finally:
        if hasattr(f, 'close'):
            f.close()
    return image



class DiabeticData():

    def __init__(self, batch_size, image_shape, ways = 5, shots = 5, query_size = 3):

        self.batch_size = batch_size
        self.data_shape = image_shape
        self.ways = ways
        self.shots = shots
        self.query_size = query_size


    def get_batch(self, mode):

        support_dir = data_dir + 'tra_data_mini/'
        query_dir = data_dir + mode + '_data_mini/'

        class_ = os.listdir(support_dir)

        support_set_x = np.zeros((self.batch_size, self.ways, self.shots, self.data_shape[0],
                                  self.data_shape[1], self.data_shape[2]), dtype=np.float32)

        support_set_y = np.zeros((self.batch_size, self.ways, self.shots), dtype=np.float32)

        query_x = np.zeros((self.batch_size, self.query_size*self.ways, self.data_shape[0], self.data_shape[1], self.data_shape[2]),
                            dtype=np.float32)
        query_y = np.zeros((self.batch_size, self.query_size*self.ways), dtype=np.float32)

        for i in range(self.batch_size):

            #totally five classes
            for j, c in enumerate(class_):
                support_img_dir = support_dir + c
                query_img_dir = query_dir + c

                support_images = glob.glob(support_img_dir + '/*.png')
                query_images = glob.glob(query_img_dir + '/*.png')
                sampled_support_imgs = np.random.choice(support_images, self.shots, replace=False)
                sampled_query_imgs = np.random.choice(query_images, self.query_size, replace=False)

                for m, img in enumerate(sampled_support_imgs):
                    f = _read_image_as_array(img)
                    support_set_x[i][j][m] = f
                    support_set_y[i][j][m] = j
                for n, img in enumerate(sampled_query_imgs):
                    f = _read_image_as_array(img)
                    query_x[i][j*self.query_size + n] = f
                    query_y[i][j*self.query_size + n] = j

            p_s = np.random.permutation(support_set_x.shape[1])
            support_set_x[i] = np.take(support_set_x[i], p_s, axis=0)
            support_set_y[i] = np.take(support_set_y[i], p_s, axis=0)

            p_q = np.random.permutation(query_x.shape[1])
            query_x[i] = np.take(query_x[i], p_q, axis=0)
            query_y[i] = np.take(query_y[i], p_q, axis=0)

        return support_set_x, support_set_y, query_x, query_y

    def get_cand_data(self, b, c, query_img_name):

        support_dir = data_dir + 'tra/'

        support_set_x = np.zeros((b, self.ways, self.shots, self.data_shape[0],
                                  self.data_shape[1], self.data_shape[2]), dtype=np.float32)

        support_set_y = np.zeros((b, self.ways, self.shots), dtype=np.float32)

        query_x = np.zeros((b,1, self.data_shape[0], self.data_shape[1], self.data_shape[2]),
                             dtype=np.float32)
        query_y = np.zeros((b, 1), dtype=np.float32)

        for i in range(b):

            for j in range(5):

                support_img_dir = support_dir + str(i)
                support_images = glob.glob(support_img_dir + '/*.jpeg')
                sampled_support_imgs = np.random.choice(support_images, self.shots, replace=False)

                for m, img in enumerate(sampled_support_imgs):
                    f = _read_image_as_array(img)
                    support_set_x[i][j][m] = f
                    support_set_y[i][j][m] = j

            p_s = np.random.permutation(support_set_x.shape[1])
            support_set_x[i] = np.take(support_set_x[i], p_s, axis=0)
            support_set_y[i] = np.take(support_set_y[i], p_s, axis=0)

        f = _read_image_as_array(query_img_name)
        query_x[:] = f
        query_y[:] = c

        return support_set_x, support_set_y, query_x, query_y