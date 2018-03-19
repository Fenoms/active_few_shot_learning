import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter
from PIL import ImageEnhance
import os
import glob

path = os.getcwd()
csv_dir = path + '/miniImagenet/csv/'


def _read_image_as_array(image, dtype='float32'):
    f = Image.open(image)
    k = np.random.randint(0, 4)
    f.rotate(k*90)
    f = f.filter(ImageFilter.GaussianBlur(1.5))
    f = random_brightness(f, [0.7,1.3])
    try:
        # image = np.asarray(f, dtype=dtype)
        image = tf.keras.preprocessing.image.img_to_array(f)
        image = augment_image(image)
    finally:
        if hasattr(f, 'close'):
            f.close()
    return image

def augment_image(img):

    img = tf.keras.preprocessing.image.random_rotation(img, 30, row_axis=0, col_axis=1, channel_axis=2)

    img = tf.keras.preprocessing.image.random_shear(img, 0.2, row_axis=0, col_axis=1, channel_axis=2)

    img = tf.keras.preprocessing.image.random_shift(img, 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2)

    img = tf.keras.preprocessing.image.random_zoom(img, [0.9, 0.9], row_axis=0, col_axis=1, channel_axis=2)

    return img


def random_brightness(img, brightness_range):

    imgenhancer_Brightness = ImageEnhance.Brightness(img)

    u = np.random.uniform(brightness_range[0], brightness_range[1])

    img = imgenhancer_Brightness.enhance(u)

    return img


def tra_data_augmentation():

    dir = path + '/miniImagenet/tra_data/'

    folders = glob.glob(dir + 'n*')

    for folder in folders:
        print(folder)
        imgs = glob.glob(folder + '/*.png')
        for i in range(3):
            for img in imgs:
                aug_img_name = img[:-4] + '_aug_' + str(i) + '.png'
                aug_img = _read_image_as_array(img)
                aug_img = Image.fromarray(aug_img.astype('uint8'))
                aug_img.save(aug_img_name)



if __name__ == '__main__':

    tra_data_augmentation()