# coding=UTF-8
'''
@author Gray
'''
import time
import logging.config
import numpy as np
import ConfigParser
from util.util import getConfigValue
from data.splitter import splitRatioByRating

logging.config.fileConfig('../util/logging.conf')
logger = logging.getLogger('root')


class Recommender(object):
    initMean, initStd = 0.0, 0.2

    def __init__(self, rateDao, config_file):
        self.cf = ConfigParser.ConfigParser()
        self.cf.read(config_file)
        self.rateDao = rateDao
        self.trainData, self.testData = splitRatioByRating(0.2, rateDao.data)
        self.save_best = bool(getConfigValue(self.cf, 'algorithm', 'save_best', False))
        self.kFold = int(getConfigValue(self.cf, 'algorithm', 'kFold', 5))
        self.early_stop_measure = getConfigValue(self.cf, 'algorithm', 'early_stop_measure', 'loss')
        self.num_iters = int(getConfigValue(self.cf, 'algorithm', 'num_iters', 100))
        self.optim_measure = getConfigValue(self.cf, 'learn', 'measure', 'sgd')
        self.global_mean = rateDao.data['rating'].values.mean()
        self.min_rate = rateDao.data['rating'].values.min()
        self.max_rate = rateDao.data['rating'].values.max()
        self.columns = rateDao.data.columns

    def execute(self):
        t_start = time.time()
        self.init_model()
        self.build_model()
        t_done = time.time()

    def predict(self, u, i):
        return self.global_mean

    def predict_inbound(self, u, i, bound):
        pred = self.predict(u, i)
        if bound:
            if pred > self.max_rate:
                pred = self.max_rate
            if pred < self.min_rate:
                pred = self.min_rate
        return pred

    def predict_with_time(self, u, i, timestamp):
        return 0

    def init_model(self):
        raise NotImplementedError()

    def build_model(self):
        raise NotImplementedError()

    def feval(self, param, gradient, iter=None):
        raise NotImplementedError()

    def evalRatings(self):
        sum_maes, sum_mses = 0., 0.
        numCnt = len(self.testData)
        for entry in self.testData.iterrows():
            uid, iid, rate = entry[1][0], entry[1][1], entry[1][2]
            pred = self.predict_inbound(uid, iid, True)
            if np.isnan(pred):
                continue
            err = np.abs(rate, pred)
            sum_maes += err
            sum_mses += err ** 2
        mae, mse, rmse = sum_maes / numCnt, sum_mses / numCnt, np.sqrt(sum_mses / numCnt)
        return {'mae': mae, 'mse': mse, 'rmse': rmse}


class Iterative_Recommender(Recommender):
    def __init__(self, rateDao, config_file, initByNorm=False):
        super(Iterative_Recommender, self).__init__(rateDao, config_file)
        self.factors = int(getConfigValue(self.cf, 'algorithm', 'factors', 25))
        self.initByNorm = initByNorm
        self.lrate = float(getConfigValue(self.cf, 'learn', 'learn_rate', 0.01))
        self.isBoldDriver = bool(getConfigValue(self.cf, 'learn', 'bold_driver', False))
        self.decay = float(getConfigValue(self.cf, 'learn', 'decay', -1))
        self.momentum = float(getConfigValue(self.cf, 'learn', 'momentum', 0.9))
        self.eps = 1. / float(getConfigValue(self.cf, 'learn', 'eps', 1e-8))
        self.rho = float(getConfigValue(self.cf, 'learn', 'rho', 0.95))
        self._beta1 = float(getConfigValue(self.cf, 'learn', 'beta1', 0.1))
        self._beta2 = float(getConfigValue(self.cf, 'learn', 'beta2', 0.001))
        self.power_t = float(getConfigValue(self.cf, 'learn', 'power_t', -1))
        self.regU = float(getConfigValue(self.cf, 'reg', 'regU', 0.001))
        self.regI = float(getConfigValue(self.cf, 'reg', 'regI', 0.001))
        self.regB = float(getConfigValue(self.cf, 'reg', 'regB', 0.001))
        self.reg = float(getConfigValue(self.cf, 'reg', 'reg', 0.001))
        self.loss, self.last_loss = 0, 0
        self.measure, self.last_measure = 0, 0

        if self.optim_measure == 'sgd':
            self.mom = 0
        elif self.optim_measure == 'rmsprop':
            self.mean_squared_grad = 0
        elif self.optim_measure == 'adagrad':
            self.sum_squared_grad = 0
        elif self.optim_measure == 'adamax' or self.optim_measure == 'adam':
            self.mom1, self.mom2 = 0, 0
        elif self.optim_measure == 'adadelta':
            self.eg2, self.edx2 = 0, 0

    def isConverged(self, iter):
        if self.early_stop_measure == 'loss':
            self.measure, self.last_measure = self.loss, self.last_loss
        else:
            self.measure = self.evalRatings()[self.early_stop_measure]
        delta_measure = self.last_measure - self.measure
        converged = np.abs(self.loss) < 1e-5 or (delta_measure > 0 and delta_measure < 1e-5)
        if not converged:
            self.updateLRate(iter)
        self.last_loss = self.loss
        self.last_measure = self.measure
        return converged

    def feval(self, param, gradient, iter=None):
        if self.optim_measure == 'sgd':
            if self.momentum != 0:
                self.mom = self.momentum * self.mom + gradient
                direction = self.mom
            else:
                direction = gradient
            return param - self.lrate * direction - self.decay * param
        elif self.optim_measure == 'adagrad':
            self.sum_squared_grad = (1 - self.decay) * self.sum_squared_grad + gradient ** 2
            scale = np.maximum((self.eps, np.sqrt(self.sum_squared_grad)))
            return param - self.lrate / scale * gradient
        elif self.optim_measure == 'adamax':
            self.mom1 = self.mom1 + self._beta1 * (gradient - self.mom1)
            self.mom2 = np.maximum(abs(gradient) + self.eps, (1 - self._beta2) * self.mom2)
            return param - self.lrate * self.mom1 / self.mom2
        elif self.optim_measure == 'rmsprop':
            self.mean_squared_grad = self.decay * self.mean_squared_grad + (1 - self.decay) * gradient ** 2
            sgd_p = self.lrate * gradient / np.maximum(np.sqrt(self.mean_squared_grad), self.eps)
            return param - sgd_p
        elif self.optim_measure == 'adadelta':
            self.eg2 = self.rho * self.eg2 + (1 - self.rho) * gradient ** 2
            sgd_p = gradient * np.sqrt(self.edx2 + self.eps) / np.sqrt(self.eg2 + self.eps)
            self.edx2 = self.rho * self.edx2 + (1 - self.rho) * sgd_p ** 2
            return param - sgd_p
        elif self.optim_measure == 'adam':
            self.mom1 = self._beta1 * self.mom1 + (1 - self._beta1) * gradient
            self.mom2 = self._beta2 * self.mom2 + (1 - self._beta2) * gradient ** 2
            sgd_p = (self.mom1 / (1 - self._beta1 ** iter)) / (
                np.sqrt(self.mom2 / (1 - self._beta2 ** iter)) + self.eps)
            return param - self.lrate * sgd_p

    def updateLRate(self, iter):
        if self.lrate <= 0:
            return
        if self.isBoldDriver and iter > 1:
            self.lrate = self.lrate * 1.05 if np.abs(self.last_loss) > np.abs(self.loss) else self.lrate * 0.5
        elif self.decay < 1 and self.decay > 0:
            self.lrate *= self.decay
        elif self.power_t < 1 and self.power_t > 0:
            self.lrate /= iter ** self.power_t

    def predict(self, u, i):
        return np.dot(self.P[u], self.Q[i])

    def init_model(self):
        if not self.initByNorm:
            self.P = np.random.uniform(-1 / np.sqrt(self.factors), 1 / np.sqrt(self.factors),
                                       (self.rateDao.num_users, self.factors))
            self.Q = np.random.uniform(-1 / np.sqrt(self.factors), 1 / np.sqrt(self.factors),
                                       (self.rateDao.num_items, self.factors))
        else:
            self.P = np.random.normal(Recommender.initMean, Recommender.initStd, (self.rateDao.num_users, self.factors))
            self.Q = np.random.normal(Recommender.initMean, Recommender.initStd, (self.rateDao.num_items, self.factors))


class Tensor_Recommender(Iterative_Recommender):
    def __init__(self, rateDao, config_file):
        super(Tensor_Recommender, self).__init__(rateDao, config_file)

    def evalRatings(self):
        def evalRatings(self):
            sum_maes, sum_mses = 0., 0.
            numCnt = len(self.testData)
            for entry in self.testData.iterrows():
                features = np.hstack((entry[1][:2], entry[1][3:]))
                rate = entry[1][2]
                pred = self.predict_with_keys(features, True)
                if np.isnan(pred):
                    continue
                err = np.abs(rate, pred)
                sum_maes += err
                sum_mses += err ** 2
            mae, mse, rmse = sum_maes / numCnt, sum_mses / numCnt, np.sqrt(sum_mses / numCnt)
            return {'mae': mae, 'mse': mse, 'rmse': rmse}

    def predict_with_keys(self, keys, bound=False):
        return self.predict_inbound(keys[0], keys[1], bound)
