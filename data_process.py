import os
import  pandas as pd
import tarfile
import numpy as np
from PIL import Image

path = os.getcwd()

csv_dir = path + '/miniImagenet/csv/'

data_dir = path + '/miniImagenet/'

if not os.path.exists(data_dir + 'tra_data'):
    os.makedirs(data_dir + 'tra_data')

if not os.path.exists(data_dir + 'val_data'):
    os.makedirs(data_dir + 'val_data')

if not os.path.exists(data_dir + 'test_data'):
    os.makedirs(data_dir + 'test_data')


def pre_process_data(mode):
    data_path = csv_dir + mode + '.csv'
    dataset = pd.read_csv(data_path, sep=',')
    labels = dataset.label.unique().tolist()


    for label in labels:
        folder = data_dir + mode + '_data/' + label
        if not os.path.exists(folder):
            os.makedirs(folder)
        tar = tarfile.open(data_dir + label + '.tar')
        imgs = tar.getmembers()
        c = 0
        for img in imgs:

            f = tar.extractfile(img)

            try:
                f = Image.open(f)
                f = f.resize((224, 224))
                f = np.asarray(f, dtype=np.float32)
                f = np.reshape(f, (224, 224, 3))
                f = Image.fromarray(f.astype('uint8'))
                img_name = folder + '/' + str(c) +'.png'
                f.save(img_name)

                c += 1
            except Exception as e:
                print('skipping image, beacuse ' + str(e))

            if c == 600:
                break

        print(c)




if __name__ == '__main__':

    print('process val data...')

    pre_process_data('val')

    print('process tra data...')

    pre_process_data('tra')

    print('process test data...')

    pre_process_data('test')



