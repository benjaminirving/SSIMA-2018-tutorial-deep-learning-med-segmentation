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

data_path = '/home/ben/Code/tutorials/Unet_segmentation_SSIMA/data/DRIVE/'


def create_data(path, name):

    image_path = path + 'images'
    mask_path =  path + '/1st_manual'

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

    np.savez(name, imgs=imgs, imgs_mask=imgs_mask)

    print('Saving to .npz files done.')


if __name__ == '__main__':

    create_data(data_path + 'training/', data_path + 'imgs_train.npz')
    create_data(data_path + 'test/', data_path + 'imgs_test.npz')

