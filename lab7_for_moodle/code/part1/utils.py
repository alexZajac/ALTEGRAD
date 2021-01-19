"""
Learning on Sets - ALTEGRAD - Jan 2021
"""

import numpy as np


def create_train_dataset():
    n_train = 100000
    max_train_card = 10

    # Task 1

    ##################
    X_train, y_train = [], []
    for _ in range(n_train):
        num_digits = np.random.randint(1, max_train_card)
        random_integers = np.pad(
            np.random.randint(2, 10, size=num_digits),
            (max_train_card-num_digits, 0)
        )
        X_train.append(random_integers)
        y_train.append(float(np.sum(random_integers)))
    ##################

    return np.array(X_train), np.array(y_train)


def create_test_dataset():

    # Task 2

    ##################
    n_test = 200000
    single_pass_size = 10000
    cardinality = 5
    X_test, y_test = [], []
    for _ in range(0, n_test, single_pass_size):
        current_batch_x = np.empty((single_pass_size, cardinality))
        current_batch_y = np.empty(single_pass_size)
        for i in range(single_pass_size):
            random_integers = np.random.randint(2, 10, size=cardinality)
            current_batch_x[i, :] = random_integers
            current_batch_y[i] = np.sum(random_integers)
        X_test.append(current_batch_x)
        y_test.append(current_batch_y)
        cardinality += 5
    ##################
    return X_test, np.array(y_test)
