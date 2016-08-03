# coding=UTF-8
'''
@author Gray
'''
import numpy as np
from numpy.linalg import cholesky
from scipy.stats import chi2


def getConfigValue(cf, section, option, default=0):
    return default if not cf.has_option(section, option) else cf.get(section, option)


def g(x):
    '''
    logistic function g(x)
    :param x:
    :return:
    '''
    return 1. / (1 + np.exp(x))


def gd(x):
    '''
    gradient value of logistic function g(x)
    :param x:
    :return:
    '''
    return g(x) * g(-x)


def wishartrand(nu, phi):
    dim = phi.shape[0]
    chol = cholesky(phi)
    foo = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(i + 1):
            if i == j:
                foo[i, j] = np.sqrt(chi2.rvs(nu - (i + 1) + 1))
            else:
                foo[i, j] = np.random.normal(0, 1)
    return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))


def mv_normalrand(mu, sigma, size):
    lamb = cholesky(sigma)
    return mu + np.dot(lamb, np.random.randn(size))


def logLoss(p, y):
    '''
    calculate the log loss cost
    p: prediction [0, 1]
    y: actual value {0, 1}
    '''
    return - np.log(p) if y == 1. else -np.log(1. - p)


def read_libffm_format(filepath):
    fields_name_index, fields_index_name = {}, {}
    features_name_index, features_index_name = {}, {}
    from collections import defaultdict
    features_fields = defaultdict(set)
    X, y = [], []
    with open(filepath, 'rb') as f:
        for line in f:
            line = line.strip('\n')
            temp = line.split(' ')
            y.append(float(temp[0]))
            x = []
            for i in xrange(1, len(temp)):
                feature, field, value = temp[i].split(':')
                if not fields_name_index.has_key(field):
                    fields_name_index[field] = len(fields_name_index)
                    fields_index_name[len(fields_index_name)] = field
                if not features_name_index.has_key(feature):
                    features_name_index[feature] = len(features_name_index)
                    features_index_name[len(features_index_name)] = feature
                features_fields[features_index_name[feature]].add(fields_index_name[field])
                x.append([features_name_index[feature], fields_name_index[field], float(value)])
            X.append(x)
    numerical_features = map(lambda t: t[0], filter(lambda t: len(t[1]) == 1, features_fields.iteritems()))
    return X, np.array(
        y), fields_name_index, fields_index_name, features_name_index, features_index_name, numerical_features
