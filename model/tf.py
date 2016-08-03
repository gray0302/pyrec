# coding=UTF-8
'''
@author Gray
'''
from base import Recommender, Tensor_Recommender
import numpy as np
from scipy.stats import wishart
from util.util import mv_normalrand, g


class CPTF(Tensor_Recommender):
    '''
    Shao W., Tensor Completion (Section 3)
    '''

    def __init__(self, rateDao, config_file):
        super(CPTF, self).__init__(rateDao, config_file)

    def init_model(self):
        self.M = []
        for dim in self.rateDao.num_features:
            if self.initByNorm:
                temp = np.random.normal(Recommender.initMean, Recommender.initStd,
                                        (dim, self.factors))
            else:
                temp = np.random.uniform(-1. / self.factors, 1. / self.factors,
                                         (dim, self.factors))
            self.M.append(temp)

    def build_model(self):
        for iter in xrange(1, self.num_iters + 1):
            self.loss = 0
            intermediate = []
            for dim in self.rateDao.num_features:
                intermediate.append(np.zeros(dim, self.factors))
            for entry in self.trainData.iterrows():
                features = np.hstack((entry[1][:2], entry[1][3:]))
                rate = entry[1][2]
                pred = self.predict_with_keys(features)
                e = rate - pred
                self.loss += e ** 2
                for d in xrange(len(self.rateDao.num_features)):
                    for f in xrange(self.factors):
                        sgd = 1
                        for dd in xrange(len(self.rateDao.num_features)):
                            if dd == d:
                                continue
                            sgd *= self.M[dd][features[dd], f]
                        intermediate[d][features[d], f] = sgd * e
            for d in xrange(len(self.rateDao.num_features)):
                for r in xrange(self.M[d].shape[0]):
                    for c in xrange(self.M[d].shape[1]):
                        self.M[d][r, c] += self.lrate * (intermediate[d][r, c] - self.reg * self.M[d][r, c])
            if self.isConverged(iter):
                break

    def predict_with_keys(self, keys, bound=False):
        pred = 0
        for f in xrange(self.factors):
            prod = 1
            for d in xrange(len(self.rateDao.num_features)):
                prod *= self.M[d][keys[d], f]
            pred += prod
        return pred


class BPTF(Tensor_Recommender):
    '''
    Temporal Collaborative Filtering with Bayesian Probabilistic Tensor
    Factorization
    '''

    def __init__(self, rateDao, config_file):
        super(BPTF, self).__init__(rateDao, config_file)
        self.gibbs = 2

    def init_model(self):
        super(BPTF, self).init_model()
        if self.initByNorm:
            self.T = np.random.uniform(-1 / np.sqrt(self.factors), 1 / np.sqrt(self.factors),
                                       (self.rateDao.num_times, self.factors))
        else:
            self.T = np.random.normal(Recommender.initMean, Recommender.initStd, (self.rateDao.num_times, self.factors))
        self.alpha = 0
        self.WI_alpha = 1
        self.df_alpha = 1
        self.L = len(self.trainData)

        self.mu_user = np.zeros(self.factors)
        self.beta_user = 2
        self.WI_user = np.eye(self.factors)
        self.df_user = self.factors
        self.lambda_user = np.eye(self.factors)
        self.mu0_u = np.zeros(self.factors)

        self.mu_item = np.zeros(self.factors)
        self.beta_item = 2
        self.WI_item = np.eye(self.factors)
        self.df_item = self.factors
        self.lambda_item = np.eye(self.factors)
        self.mu0_i = np.zeros(self.factors)

        self.mu_time = np.zeros(self.factors)
        self.beta_time = 2
        self.WI_time = np.eye(self.factors)
        self.df_time = self.factors
        self.lambda_time = np.eye(self.factors)
        self.mu0_t = np.ones(self.factors)

    def predict_with_keys(self, keys, bound=False):
        pred = self.global_mean
        for f in xrange(self.factors):
            prod = self.P[keys[0], f] * self.Q[keys[1], f] * self.T[keys[2], f]
            pred += prod
        return pred

    def predict_batch(self, user_ids, item_ids, time_ids):
        return np.array([self.predict_with_keys(key) for key in zip(user_ids, item_ids, time_ids)])

    def _update_alpha(self):
        sum = 0
        for entry in self.trainData.iterrows():
            keys = [entry[1][0], entry[1][1], entry[1][3]]
            rate = entry[1][2]
            sum += (rate - self.predict_with_keys(keys, True)) ** 2
        WI_post = self.WI_alpha + sum
        df_post = self.df_alpha + self.L
        self.alpha = wishart.rvs(df=df_post, scale=1. / WI_post)

    def _update_user_params(self):
        N = self.rateDao.num_users
        X_bar = np.mean(self.P, 0)
        S_bar = np.cov(self.P.T)
        norm_X_bar = self.mu0_u - X_bar
        WI_post = self.WI_user + N * S_bar + np.outer(norm_X_bar, norm_X_bar) * (N * self.beta_user) / (
            self.beta_user + N)
        # ensure the matrix's symmetry
        WI_post = (WI_post + WI_post.T) / 2.
        df_post = self.df_user + N
        self.lambda_user = wishart.rvs(df_post, WI_post)
        # 以下可参考http://blog.pluskid.org/?p=430
        mu_temp = (self.beta_user * self.mu0_u + N * X_bar) / (self.beta_user + N)
        sigma_temp = np.linalg.inv(np.dot(self.beta_user + N, self.lambda_user))
        self.mu_user = mv_normalrand(mu_temp, sigma_temp, self.factors)

    def _update_item_params(self):
        N = self.rateDao.num_items
        X_bar = np.mean(self.Q, 0)
        S_bar = np.cov(self.Q.T)
        norm_X_bar = self.mu0_i - X_bar
        WI_post = self.WI_item + N * S_bar + np.outer(norm_X_bar, norm_X_bar) * (N * self.beta_item) / (
            self.beta_item + N)
        WI_post = (WI_post + WI_post.T) / 2.
        df_post = self.df_item + N
        self.lambda_item = wishart.rvs(df_post, WI_post)
        mu_temp = (self.beta_item * self.mu0_i + N * X_bar) / (self.beta_item + N)
        sigma_temp = np.linalg.inv(np.dot(self.beta_item + N, self.lambda_item))
        self.mu_item = mv_normalrand(mu_temp, sigma_temp, self.factors)

    def _update_time_params(self):
        N = self.rateDao.num_times
        diff_temp = np.diff(self.T, axis=0)
        diff_t0 = self.T[0] - self.mu0_t
        WI_post = self.WI_time + np.dot(diff_temp.T, diff_temp) + self.beta_time / (1 + self.beta_time) * np.outer(
            diff_t0, diff_t0)
        WI_post = (WI_post + WI_post.T) / 2.
        df_post = self.df_item + N
        self.lambda_time = wishart.rvs(df_post, WI_post)
        mu_temp = (self.beta_time * self.mu0_t + self.T[0]) / (self.beta_item + 1)
        sigma_temp = np.linalg.inv(np.dot(self.beta_item + 1, self.lambda_time))
        self.mu_time = mv_normalrand(mu_temp, sigma_temp, self.factors)

    def _update_user_features(self):
        for uid in xrange(self.rateDao.num_users):
            rated = self.trainData[self.trainData[self.columns[0]] == uid]
            if not rated.empty:
                rated_iids = map(lambda entry: entry[1][1], rated.iterrows())
                rated_tids = map(lambda entry: entry[1][3], rated.iterrows())
                # 逐元素相乘，不是点积
                Q = self.Q[rated_iids] * self.T[rated_tids]
                rating = np.array(map(lambda entry: entry[1][2], rated.iterrows())) - self.global_mean
                sigma = np.linalg.inv(self.lambda_user + self.alpha * np.dot(Q.T, Q))
                mu_temp = self.alpha * np.dot(rating, Q) + np.dot(self.lambda_user, self.mu_user)
                mean = np.dot(sigma, mu_temp)
                self.P[uid] = mv_normalrand(mean, sigma, self.factors)

    def _update_item_features(self):
        for iid in xrange(self.rateDao.num_items):
            rated = self.trainData[self.trainData[self.columns[1]] == iid]
            if not rated.empty:
                rated_uids = map(lambda entry: entry[1][0], rated.iterrows())
                rated_tids = map(lambda entry: entry[1][3], rated.iterrows())
                P = self.P[rated_uids] * self.T[rated_tids]
                rating = np.array(map(lambda entry: entry[1][2], rated.iterrows())) - self.global_mean
                sigma = np.linalg.inv(self.lambda_item + self.alpha * np.dot(P.T, P))
                mu_temp = self.alpha * np.dot(rating, P) + np.dot(self.lambda_item, self.mu_item)
                mean = np.dot(sigma, mu_temp)
                self.Q[iid] = mv_normalrand(mean, sigma, self.factors)

    def _update_time_features(self):
        for tid in xrange(self.rateDao.num_times):
            rated = self.trainData[self.trainData[self.columns[3]] == tid]
            if not rated.empty:
                rated_uids = map(lambda entry: entry[1][0], rated.iterrows())
                rated_iids = map(lambda entry: entry[1][1], rated.iterrows())
                X = self.P[rated_uids] * self.Q[rated_iids]
                if tid == 0:
                    mean = (self.T[tid + 1] + self.mu_time) / 2
                    sigma = np.linalg.inv(2 * self.lambda_time + self.alpha * np.dot(X.T, X))
                elif tid < self.rateDao.num_times - 1:
                    rating = np.array(map(lambda entry: entry[1][2], rated.iterrows())) - self.global_mean
                    sigma = np.linalg.inv(2 * self.lambda_time + self.alpha * np.dot(X.T, X))
                    mu_temp = np.dot(self.lambda_time, self.T[tid - 1] + self.T[tid + 1]) + self.alpha * np.dot(rating,
                                                                                                                X)
                    mean = np.dot(sigma, mu_temp)
                else:
                    rating = np.array(map(lambda entry: entry[1][2], rated.iterrows())) - self.global_mean
                    sigma = np.linalg.inv(self.lambda_time + self.alpha * np.dot(X.T, X))
                    mu_temp = np.dot(self.lambda_time, self.T[tid - 1]) + self.alpha * np.dot(rating, X)
                    mean = np.dot(sigma, mu_temp)
                self.T[tid] = mv_normalrand(mean, sigma, self.factors)

    def build_model(self):
        columns = self.columns
        for iter in xrange(1, self.num_iters + 1):
            self._update_user_params()
            self._update_item_params()
            for _ in xrange(self.gibbs):
                self._update_user_features()
                self._update_item_features()
                self._update_time_features()
            rates = self.trainData[columns[2]]
            preds = self.predict_batch(self.trainData[columns[0]], self.trainData[columns[1]],
                                       self.trainData[columns[3]])
            self.loss = (rates - preds) ** 2
            if self.isConverged(iter):
                break


class PITF(Tensor_Recommender):
    '''
    Pairwise Interaction Tensor Factorization
    for Personalized Tag Recommendation
    这个算法在划分训练集和测试集的时候
    '''

    def __init__(self, rateDao, config_file):
        super(PITF, self).__init__(rateDao, config_file)

    def init_model(self):
        self.P = np.random.normal(Recommender.initMean, Recommender.initStd, (self.rateDao.num_users, self.factors))
        self.Q = np.random.normal(Recommender.initMean, Recommender.initStd, (self.rateDao.num_users, self.factors))
        self.TU = np.random.normal(Recommender.initMean, Recommender.initStd, (self.rateDao.num_tags, self.factors))
        self.TI = np.random.normal(Recommender.initMean, Recommender.initStd, (self.rateDao.num_tags, self.factors))

    def predict_with_keys(self, keys, bound=False):
        pred = np.dot(self.P[keys[0]], self.TU[keys[2]]) + np.dot(self.Q[keys[1]], self.TI[keys[2]])
        return pred

    def build_model(self):
        for iter in xrange(self.num_iters * np.max(self.rateDao.num_users, self.rateDao.num_items) + 1):
            pos_sample = self.trainData.sample()
            uid, iid, tagA = pos_sample[self.columns[0]].iat[0], pos_sample[self.columns[1]].iat[0], \
                             pos_sample[self.columns[3]].iat[0]
            neg_sample = self.trainData[
                (self.trainData[self.columns[0]] != uid) & (self.trainData[self.columns[1]] != iid)].sample()
            tagB = neg_sample[self.columns[3]].iat[0]
            y = self.predict_with_keys([uid, iid, tagA]) - self.predict_with_keys([uid, iid, tagB])
            delta = g(-y)
            pu, qi, tua, tub, tia, tib = self.P[uid], self.Q[iid], self.TU[tagA], self.TU[tagB], \
                                         self.TI[tagA], self.TI[tagB]
            self.P[uid] += self.lrate * (delta * (tua - tub) - self.regU * pu)
            self.Q[iid] += self.lrate * (delta * (tia - tib) - self.regI * qi)
            self.TU[tagA] += self.lrate * (delta * pu - self.reg * tua)
            self.TU[tagB] += self.lrate * (-delta * pu - self.reg * tub)
            self.TI[tagA] += self.lrate * (delta * qi - self.reg * tia)
            self.TI[tagB] += self.lrate * (-delta * qi - self.reg * tib)
            if self.isConverged(iter):
                break
