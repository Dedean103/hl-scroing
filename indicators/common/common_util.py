import numpy as np


def last_argmax(arr):
    reversed_index = np.argmax(arr[::-1])  # Apply argmax to the reversed array
    return len(arr) - 1 - reversed_index


def last_argmin(arr):
    reversed_index = np.argmin(arr[::-1])  # Apply argmin to the reversed array
    return len(arr) - 1 - reversed_index
