"""Utilities for real-time data augmentation on image data.
"""
import numpy as np

from keras_preprocessing.image.iterator import Iterator


class CustomIterator(Iterator):

    def __init__(self, n, batch_size, shuffle, seed, positive_n):
        super().__init__(n, batch_size, shuffle, seed)
        self.positive_n = positive_n  # number of normal photos in Dataset

    def _set_index_array(self):
        if self.shuffle:
            if self.positive_n:
                negative_n = self.n - self.positive_n  # neg
                positive_indices = np.random.permutation(self.positive_n)
                negative_indices = np.random.permutation(np.arange(self.positive_n, self.n))

                if self.positive_n < negative_n:
                    pos_size = int(self.batch_size * self.positive_n / self.n)  # number of positive photos in each mini_batch
                    neg_size = self.batch_size - pos_size
                else:
                    neg_size = int(self.batch_size * negative_n / self.n)
                    pos_size = self.batch_size - neg_size

                self.index_array = np.array([], dtype="int")
                pos_pointer = neg_pointer = 0

                for i in range(self.n // self.batch_size + 1):

                    if pos_pointer + pos_size < self.positive_n:
                        self.index_array = np.append(self.index_array, positive_indices[pos_pointer: pos_pointer + pos_size])
                        pos_pointer += pos_size
                    else:
                        self.index_array = np.append(self.index_array, positive_indices[pos_pointer:])
                        self.index_array = np.append(self.index_array, negative_indices[neg_pointer:])
                        break
                    if neg_pointer + neg_size < negative_n:
                        self.index_array = np.append(self.index_array, negative_indices[neg_pointer: neg_pointer + neg_size])
                        neg_pointer += neg_size
                    else:
                        self.index_array = np.append(self.index_array, negative_indices[neg_pointer:])
                        self.index_array = np.append(self.index_array, positive_indices[pos_pointer:])
                        break

            else:
                self.index_array = np.random.permutation(self.n)
        else:
            self.index_array = np.arange(self.n)
