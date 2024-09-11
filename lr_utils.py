import numpy as np
import h5py

def load_dataset():
    """
    Load the dataset from H5 files and preprocess it.

    Returns:
    --------
    train_set_x_orig : numpy.ndarray
        Training set features, originally loaded from the dataset.
    
    train_set_y_orig : numpy.ndarray
        Training set labels, reshaped to a 2D array.
    
    test_set_x_orig : numpy.ndarray
        Test set features, originally loaded from the dataset.
    
    test_set_y_orig : numpy.ndarray
        Test set labels, reshaped to a 2D array.
    
    classes : numpy.ndarray
        Array containing the list of class labels.
    """
    
    # Load training data
    with h5py.File('datasets/train_catvnoncat.h5', "r") as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # Training set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # Training set labels

    # Load test data
    with h5py.File('datasets/test_catvnoncat.h5', "r") as test_dataset:
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # Test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # Test set labels
        classes = np.array(test_dataset["list_classes"][:])         # Class labels

    # Reshape the labels to be 2D arrays for consistency
    train_set_y_orig = train_set_y_orig.reshape((1, -1))
    test_set_y_orig = test_set_y_orig.reshape((1, -1))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
