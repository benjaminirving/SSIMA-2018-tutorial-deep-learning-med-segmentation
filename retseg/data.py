import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.util import crop

from skimage.io import imsave, imread

img_cols_orig = 565
img_rows_orig = 584

img_cols = 512
img_rows = 512

crop1 = int((img_rows_orig-img_rows)/2)
crop2 = int((img_cols_orig-img_cols)/2)

data_path = '/home/ben/Code/tutorials/Unet_segmentation_SSIMA/data/DRIVE/training'


def create_train_data():

    image_path = data_path + '/images'
    mask_path  = data_path + '/1st_manual'

    images = os.listdir(image_path)
    total = len(images)

    imgs = np.ndarray((total, img_rows, img_cols, 1), dtype=np.float)
    imgs_mask = np.ndarray((total, img_rows, img_cols, 1), dtype=np.float)

    print('-'*30)
    print('Creating training images...')
    print('-'*30)

    for i, image_name in enumerate(images):

        image_mask_name = image_name.split('_')[0] + '_manual1.gif'
        img = imread(os.path.join(image_path, image_name), as_gray=True)
        img_mask = imread(os.path.join(mask_path, image_mask_name), as_gray=True)

        img = crop(img, ((crop1, crop1), (crop2, crop2+1)))
        img_mask = crop(img_mask, ((crop1, crop1), (crop2, crop2+1)))

        img = np.expand_dims(img, axis=-1)
        img_mask = np.expand_dims(img_mask, axis=-1)

        imgs[i] = img
        imgs_mask[i] = img_mask

    print('Loading done.')

    plt.figure()
    plt.imshow(np.squeeze(imgs[2]))
    plt.contour(np.squeeze(imgs_mask[2]))
    plt.show()

    np.savez('imgs_train.npz', imgs=imgs, imgs_mask=imgs_mask)

    print('Saving to .npy files done.')


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, img_rows, img_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(train_data_path, image_name), as_grey=False)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1

    print('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id


if __name__ == '__main__':
    create_train_data()
    # create_test_data()
