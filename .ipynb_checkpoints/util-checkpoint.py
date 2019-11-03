import os
import glob
import imageio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

DATASET_FOLDER = 'dataset'
LABELS = [1, 2, 3, 4, 5, 6, 1]
CLASSES = ['buildings', 'street', 'forest',
           'glacier', 'mountain', 'sea', 'city']
IMAGE_SIZE = 150
IMAGE_LEN = 150*150*3


def get_images(image_class, folder="train"):
    """Return a list of images belonging to the specified clazz. By default, the
       images are taken from the training set folder. 

    Args:
        image_class (str): one of the values from CLASSES
        folder (str): A folder inside DATASET_FOLDER
    Returns:
        list of numpy array: Each numpy array is a flatten image. 
    """

    images = []
    path_regex = os.path.join(DATASET_FOLDER, folder, image_class, '*.jpg')
    for path in glob.glob(path_regex):
        image = np.asarray(imageio.imread(path))

        if (image.shape[0]*image.shape[1]*image.shape[2] == IMAGE_LEN):
            images.append(image)

    return images


def get_dataset(folder="train"):
    """Return the images and their labels from the training set folder by default. 

    Args:
        folder (str): A folder inside DATASET_FOLDER
    Returns:
        X (list of numpy arrays): each numpy array is a flatten image
        Y (list of int): image labels
    """

    X = []
    Y = []
    for label, image_class in zip(LABELS, CLASSES):
        images = get_images(image_class, folder)
        X = X + images
        Y = Y + [label]*len(images)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def plot_image(instances, images_per_row=10, **options):
    """Plot images of size IMAGE_SIZExIMAGE_SIZEx3

    Args:
        instances (list of numpy arrays): Each array is an image of size IMAGE_SIZExIMAGE_SIZEx3
        images_per_row (int, optional): Number of images per row. Defaults to 10.

    Reference: code is adopted from 
        Geron A: 2017, "Hands-On Machine Learning with Scikit-Learn and Tensorflow". 
    """

    images_per_row = min(len(instances), images_per_row)
    images = [instance for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((IMAGE_SIZE, IMAGE_SIZE * n_empty)))

    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=mpl.cm.binary, **options)
    plt.axis("off")


def plot_roc_curve(fprs, tprs, legend_labels, legend_colors):
    """Plot the ROC curve

    Args:
        fprs (list of numpy arrays): list of false positive rate
        tprs (list of numpy arrays): list of true positive rate
        legend_labels (list of str): list of labels for legend
        legend_colors (list of str): list of colors
    """

    assert(len(fprs) == len(tprs))
    for fpr, tpr, label, color in zip(fprs, tprs, legend_labels, legend_colors):
        plt.plot(fpr, tpr, color, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
