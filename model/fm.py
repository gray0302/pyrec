# coding=UTF-8
'''
@author Gray
'''
import numpy as np
from sklearn.metrics import mean_squared_error, roc_curve, auc
from util.util import logLoss
from collections import defaultdict


class FM(object):
    '''
    implement a simple two-way interaction
    '''

    def __init__(self, num_factors, task, l2_reg_b=0.01, l2_reg_w=0.01, l2_reg_v=0.01, init_lr=0.001,
                 stdev=0.1):
        self.num_factors = num_factors
        self.task = task
        self.init_lr = init_lr
        self.stdev = stdev
        self.reg_b = l2_reg_b
        self.reg_w = l2_reg_w
        self.reg_v = l2_reg_v

    def _init_model(self):
        self.b = 0.
        self.w = np.zeros(self.num_attributes)
        self.v = np.random.normal(scale=self.stdev, size=(self.num_attributes, self.num_factors))

    def predict(self, x):
        interaction = float(np.sum(x.dot(self.v).power(2) - x.power(2).dot(self.v ** 2)) / 2)
        return self.b + x.dot(self.w) + interaction

    def predict_batch(self, X):
        linear = self.X_test.dot(self.w)
        interaction = np.sum(self.X_test.dot(self.v).power(2) - self.X_test.power(2).dot(self.v ** 2), axis=1) / 2.
        return self.b + linear + interaction

    def fit(self, X_train, y_train, X_test, y_test):
        '''
        :param X_train: is a csr matrix
        :param y_train: is a 1d numpy array
        :return:
        '''
        self.num_attributes = X_train.shape[1]
        self.X_test, self.y_test = X_test, y_test
        self.X_train, self.y_train = X_train, y_train
        self._init_model()
        prev = float('inf')
        current = 0
        eps = 1e-3
        while abs(prev - current) > eps:
            prev = current
            for x, y in zip(X_train, y_train):
                current = self.update(x, y)

    def update(self, x, y):
        grad_base = self._loss_derivative(x, y)
        self.b = self.b - self.init_lr * (grad_base + self.reg_b * self.b)
        for attr in xrange(self.num_attributes):
            if not x.getcol(attr).data:
                xi = x.getcol(attr).data[0]
                grad_w = grad_base * xi
                self.w[attr] -= self.init_lr * (grad_w + self.reg_w * self.w[attr])
                grad_v = grad_base * (x.dot(self.v).dot(xi)[0] - self.v[attr] * xi ** 2)
                self.v[attr] -= self.init_lr * (grad_v + self.reg_v * self.v[attr])
        return self._evaluate()

    def _loss_derivative(self, x, y):
        if self.task == 'regression':  # rmse
            return self.predict(x) - y
        elif self.task == 'classification':  # cross entropy
            return 1. / (1 + np.exp(-self.predict(x))) - y

    def _evaluate(self):
        if self.task == 'regression':
            y_pred = self.predict_batch(self.X_test)
            return mean_squared_error(self.y_test, y_pred) ** 0.5
        elif self.task == 'classification':
            y_pred = self.predict_batch(self.X_test)
            fpr, tpr, _ = roc_curve(self.y_test, y_pred, pos_label=1)
            return auc(fpr, tpr)


class HFM_FTRL(object):
    '''
    hashed factorization machine with the follow the regularized leader online learning
    '''

    def __init__(self, num_factors, L1, L2, L1_fm, L2_fm, D, task, alpha, beta, alpha_fm=.1, beta_fm=1., init_lr=0.001,
                 stdev=0.1, dropout_rate=1.):
        self.alpha = alpha
        self.alpha_fm = alpha_fm
        self.num_factors = num_factors
        self.D = D
        self.L1 = L1
        self.L2 = L2
        self.L1_fm = L1_fm
        self.L2_fm = L2_fm
        self.task = task
        self.beta = beta
        self.beta_fm = beta_fm
        self.init_lr = init_lr
        self.stdev = stdev
        self.dropout_rate = dropout_rate

        self.n = np.zeros(D + 1)
        self.z = np.zeros(D + 1)
        self.w = np.zeros(D + 1)
        self.n_fm = {}
        self.z_fm = {}
        self.w_fm = {}

    def init_fm(self, i):
        if i not in self.n_fm:
            self.n_fm[i] = np.zeros(self.num_factors)
            self.z_fm[i] - np.zeros(self.num_factors)
            self.w_fm[i] = np.random.normal(scale=self.stdev, size=self.num_factors)

    def predict_raw(self, x):
        '''
        predict the raw score prior to logit transformation.
        '''
        raw_y = 0
        # w[0]的取值有点迷糊，别的地方配合FTRL的定义都梳理清楚了
        self.w[0] = (-self.z[0]) / ((self.beta + np.sqrt(self.n[0])) / self.alpha)
        raw_y += self.w[0]
        for i in x:
            if np.abs(self.z[i]) <= self.L1:
                self.w[i] = 0
            else:
                self.w[i] = (np.sign(self.z[i]) * self.L1 - self.z[i]) / (
                    (self.beta + np.sqrt(self.n[i])) / self.alpha + self.L2)
            raw_y += self.w[i]

        for i in x:
            self.init_fm(i)
            for k in xrange(self.num_factors):
                sign = np.sign(self.z_fm[i][k])
                if sign * self.z_fm[i][k] <= self.L1_fm:
                    self.w_fm[i][k] = 0
                else:
                    self.w_fm[i][k] = (sign * self.L1_fm - self.z_fm[i][k]) / (
                        (self.beta_fm + np.sqrt(self.n_fm[i][k])) / self.alpha_fm + self.L2_fm)
        len_x = len(x)
        for i in xrange(len_x - 1):
            for j in xrange(i + 1, len_x):
                raw_y += np.dot(self.w_fm[x[i]], self.w_fm[x[j]])
        return raw_y

    def predict(self, x):
        return 1. / (1 + np.exp(self.predict_raw(x)))

    def _dropout(self, x):
        for i, _ in enumerate(x):
            if np.random.random() > self.dropout_rate:
                del x[i]

    def dropoutThenPredict(self, x):
        self._dropout(x)
        return self.predict(x)

    def fit(self, X_train, X_test, y_train, y_test):
        self.X_test, self.y_test = X_test, y_test
        prev = float('inf')
        current = 0
        eps = 1e-3
        while abs(prev - current) > eps:
            prev = current
            for x, y in zip(X_train, y_train):
                current = self.update(x, y)

    def update(self, x, y):
        y_pred = self.predict(x)
        g = y_pred - y
        loss = logLoss(y_pred, y)
        fm_sum = {}
        for i in x + [0]:
            sigma = (np.sqrt(self.n[i] + g ** 2) - np.sqrt(self.n[i])) / self.alpha
            self.z[i] += g - sigma * self.w[i]
            self.n[i] += g ** 2
            fm_sum[i] = np.zeros(self.num_factors)
        len_x = len(x)
        for i in xrange(len_x):
            for j in xrange(len_x):
                if i != j:
                    fm_sum[x[i]] += self.w_fm[x[j]]
        for i in x:
            g_fm = g * fm_sum[i]
            sigma = (np.sqrt(self.n_fm[i] + g_fm ** 2) - np.sqrt(self.n_fm[i])) / self.alpha_fm
            self.z_fm[i] += g_fm - sigma * self.w_fm[i]
            self.n_fm[i] += g_fm ** 2
        return loss


# hash trick
# abs(hash(hash_salt+key+'_'+value))%hash_size+1

class FFM(FM):
    '''
    Field-aware Factorization Machines
    '''

    def __init__(self, num_fields, num_features, num_factors, numerical_features, task, l2_reg_b=0.01, l2_reg_w=0.01,
                 l2_reg_v=0.01,
                 init_lr=0.001, stdev=0.1, normalize=True):
        super(FFM, self).__init__(num_factors, task, l2_reg_b, l2_reg_w, l2_reg_v, init_lr, stdev)
        self.num_fields = num_fields
        self.num_features = num_features
        self.numerical_features = numerical_features
        self.normalize = normalize

    def _init_model(self):
        self.b = 0.
        self.w = np.zeros(self.num_features)
        self.v = np.random.normal(scale=0.1, size=(self.num_features, self.num_fields, self.num_factors))
        if self.normalize:
            self._normalize(self.X_train, self._get_min_max(self.X_train))
            self._normalize(self.X_test, self._get_min_max(self.X_test))

    def _get_min_max(self, X):
        nf_values = defaultdict(list)
        for x in X:
            for f in x:
                if f[0] in self.numerical_features:
                    nf_values[f[0]].append(f[2])
        for k, v in nf_values.iteritems():
            nf_values[k] = (np.min(v), np.max(v))
        return nf_values

    def _normalize(self, X, nf_min_max):
        for x in X:
            for f in x:
                if f[0] in nf_min_max:
                    f[2] = (f[2] - nf_min_max[f[0]][0]) / (nf_min_max[f[0]][1] - nf_min_max[f[0]][0])

    def predict(self, x):
        pred = self.b
        len_x = len(x)
        for i in xrange(len_x):
            feature_i, field_i, value_i = x[i]
            pred += self.w[feature_i] * value_i
            for j in xrange(i + 1, len_x):
                feature_j, field_j, value_j = x[j]
                pred += np.dot(self.v[feature_i][field_j], self.v[feature_j][field_i]) * value_i * value_j
        return pred

    def predict_batch(self, X):
        return np.array([self.predict(x) for x in X])

    def update(self, x, y):
        grad_base = self._loss_derivative(x, y)
        self.b = self.b - self.init_lr * (grad_base + self.reg_b * self.b)
        len_x = len(x)
        for i in xrange(len_x):
            feature_i, field_i, value_i = x[i]
            grad_w = grad_base * value_i
            self.w[feature_i] -= self.init_lr * (grad_w + self.reg_w * self.w[feature_i])
            for j in xrange(i + 1, len_x):
                feature_j, field_j, value_j = x[j]
                tempij, tempji = self.v[feature_j][field_i], self.v[feature_j][field_i]
                self.v[feature_i][field_j] -= self.init_lr * (
                    grad_base * tempji * value_i * value_j + self.reg_v * tempij)
                self.v[feature_j][field_i] -= self.init_lr * (
                    grad_base * tempij * value_i * value_j + self.reg_v * tempji)
        return self._evaluate()
