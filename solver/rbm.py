# coding=UTF-8
'''
@author Gray
'''
import numpy as np
import copy


class RBM(object):
    def __init__(self, visible, hidden, learning_rate=0.01, momentum=0.8, k=1):
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.1, size=(visible, hidden))
        self.hidden_bias = np.zeros(hidden)
        self.visible_bias = np.ones(visible)
        self.momentum = momentum
        self.k = k

    def iter_passes(self, visible):
        while True:
            hidden = self.logistic(np.dot(visible, self.weights) + self.hidden_bias)
            yield visible, hidden
            hidden_01 = np.random.binomial(n=1, p=hidden)
            visible = self.logistic(np.dot(hidden_01, self.weights.T) + self.visible_bias)

    def fit(self, data, epochs, batch_size):
        data = self.prepare_data(data)
        vb = data.mean(axis=0)
        for i, v in enumerate(vb):
            if (v == 1):
                vb[i] = 0.9999
        vb = 1 / (1 - vb)
        self.visible_bias = np.log(vb)
        (num_examples, data_size) = data.shape
        batches = num_examples / batch_size
        data = data[0:batch_size * batches]
        data = data.reshape((batches, batch_size, data_size))
        velocity = np.zeros(self.weights.shape)
        for epoch in xrange(epochs):
            total_error = 0
            for batch in data:
                gradient, vb, hb, error = self.run_batch(batch)
                total_error += error
                velocity = self.momentum * velocity + gradient
                self.weights += velocity * self.learning_rate
                self.hidden_bias += hb
                self.visible_bias += vb
            total_error /= float(batches)
            print "after epoch {0}, average error: {1}".format(epoch, total_error)

    def run_batch(self, visible_batch):
        passes = self.iter_passes(visible_batch)
        v0, h0 = passes.next()
        for _ in xrange(self.k):
            vk, hk = passes.next()
        weight_gradient = (np.dot(v0.T, h0) - np.dot(vk.T, hk)) / float(visible_batch.shape[0])
        error = np.square(v0 - vk).sum() / float(visible_batch.shape[0])
        vbias_gradient = (v0 - vk).mean(axis=0) * self.learning_rate
        hbias_gradient = (h0 - hk).mean(axis=0) * self.learning_rate
        return weight_gradient, vbias_gradient, hbias_gradient, error

    def transform(self, data, probability=False):
        data = self.prepare_data(data)
        hidden = self.logistic(np.dot(data, self.weights) + self.hidden_bias)
        if not probability:
            hidden = np.random.binomial(n=1, p=hidden)
        return hidden

    def prepare_data(self, data):
        data = np.array(data).astype('float32')
        return data

    def logistic(self, x):
        return 1 / (1 + np.exp(-x))


class RBMCF(RBM):
    def __init__(self, vis_num, hid_num, rate=2, learning_rate=0.02, k=1):
        super(RBMCF, self).__init__(vis_num, hid_num, learning_rate, k=k)
        self.weights = np.random.normal(0, 0.01, size=(vis_num, rate, hid_num))
        self.visible_bias = np.random.normal(0, 0.01, size=(vis_num, rate))
        self.rate = rate
        self.vis_num = vis_num
        self.hid_num = hid_num
        self.learning_rate = learning_rate
        self.k = k

    def fit(self, data, epochs, batch_size=64):
        data = self.prepare_data(data)
        self.data = copy.copy(data)
        (num_examples, item_size, rate) = data.shape
        batches = num_examples / batch_size
        data = data[0:batch_size * batches]
        data = data.reshape((batches, batch_size, item_size, rate))
        for epoch in xrange(epochs):
            total_error = 0
            for batch in data:
                gradient, vb, hb, error = self.run_batch(batch)
                total_error += error
                self.weights += gradient * self.learning_rate
                self.hidden_bias += hb * self.learning_rate
                self.visible_bias += vb * self.learning_rate
            total_error /= float(batches)
            print "after epoch {0}, average error: {1}".format(epoch, total_error)

    def iter_passes(self, visible):
        while True:
            hidden = self.logistic(np.tensordot(visible, self.weights, axes=([1, 2], [0, 1])) + self.hidden_bias)
            yield visible, hidden
            hidden_01 = np.random.binomial(n=1, p=hidden)
            denom = sum(
                [np.exp(np.tensordot(hidden_01, self.weights[:, r, :], axes=([1], [1])) + self.visible_bias[:, r])
                 for r in xrange(self.rate)])
            visible = np.exp((np.tensordot(hidden_01, self.weights, axes=([1], [2])) + self.visible_bias))
            for b in xrange(len(visible)):
                visible[b] = (visible[b].T / denom[b]).T

    def run_batch(self, visible_batch):
        passes = self.iter_passes(visible_batch)
        v0, h0 = passes.next()
        for _ in xrange(self.k):
            vk, hk = passes.next()
        weight_gradient = (np.tensordot(v0, h0, axes=([0], [0])) - np.tensordot(vk, hk, axes=([0], [0]))) / float(
            visible_batch.shape[0])
        error = np.square(v0 - vk).sum() / float(visible_batch.shape[0])
        vbias_gradient = (v0 - vk).mean(axis=0)
        hbias_gradient = (h0 - hk).mean(axis=0)
        return weight_gradient, vbias_gradient, hbias_gradient, error

    def predict(self, uid, iid):
        uvisible = self.data[uid]
        hidden = np.random.binomial(n=1, p=self.logistic(
            np.tensordot(uvisible, self.weights, axes=([0, 1], [0, 1])) + self.hidden_bias))
        denom = sum(
            [np.exp(np.dot(hidden, self.weights[iid, r, :]) + self.visible_bias[iid, r])
             for r in xrange(self.rate)])
        visible = np.exp((np.tensordot(hidden, self.weights[iid], axes=([0], [1])) + self.visible_bias))
        visible /= denom
        score = 0.
        return sum([visible[r] * (r + 1) for r in xrange(self.rate)])
