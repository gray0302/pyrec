# coding=UTF-8
'''
@author Gray
'''
import pandas as pd
import numpy as np


class DataDao(object):
    def __init__(self, filepath, sep=',', names=('user_id', 'item_id', 'rating')):
        self._data = pd.read_csv(filepath, sep=sep, names=names, index_col=False)
        self.features_id_index, self.features_index_id = [], []
        for i in xrange(0, len(names)):
            if i != 2:
                self.features_id_index.append({})
                self.features_index_id.append({})
                pos = i if i < 2 else i - 1
                for id, index in zip(np.unique(self._data[names[i]]), range(np.unique(self._data[names[i]]).size)):
                    self.features_id_index[pos][id] = index
                    self.features_index_id[pos][index] = id
        for i in xrange(3, len(names)):
            if i != 2:
                pos = i if i < 2 else i - 1
                self._data.loc[:, names[i]] = map(lambda id: self.features_id_index[pos][id], self._data[names[i]])

        self._num_users = len(self.features_id_index[0])
        self._num_items = len(self.features_id_index[1])
        if len(self.features_id_index) > 2:
            self._num_times = len(self.features_id_index[2])
            self._num_tags=len(self.features_id_index[2])

    @property
    def data(self):
        return self._data

    @property
    def num_users(self):
        return self._num_users

    @property
    def num_items(self):
        return self._num_items

    @property
    def num_times(self):
        return self._num_times

    @property
    def num_tags(self):
        return self._num_tags

    @property
    def num_features(self):
        return [len(self.features_id_index[i]) for i in xrange(len(self.features_id_index))]
