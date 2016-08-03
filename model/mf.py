# coding=UTF-8
'''
@author Gray
'''
from model.base import Iterative_Recommender, Recommender, Tensor_Recommender
from util.util import getConfigValue, g, gd, mv_normalrand
import numpy as np
import numpy.linalg
from scipy.stats import wishart


class NMF(Iterative_Recommender):
    def __init__(self, rateDao, config_file):
        super(NMF, self).__init__(rateDao, config_file)
        self.lrate = -1

    def init_model(self):
        self.P = np.random.uniform(0, 1. / np.sqrt(self.factors), (self.rateDao.num_users, self.factors))
        self.Q = np.random.uniform(0, 1. / np.sqrt(self.factors), (self.rateDao.num_items, self.factors))
        self.V = np.zeros((self.rateDao.num_users, self.rateDao.num_items))
        for entry in self.trainData.iterrows():
            self.V[entry[1][0], entry[1][1]] = entry[1][2]

    # use the square error
    def build_model(self):
        for iter in xrange(1, self.num_iters + 1):
            # update W by fixing H
            VH = np.dot(self.V, self.Q)
            WHTH = np.dot(np.dot(self.P, self.Q.T), self.Q)
            for uid in xrange(self.rateDao.num_users):
                for f in xrange(self.factors):
                    self.P[uid, f] *= VH[uid, f] / WHTH[uid, f]
            # update H by fixing W
            VTW = np.dot(self.V.T, self.W)
            HWTW = np.dot(np.dot(self.Q, self.P.T), self.P)
            for iid in xrange(self.rateDao.num_items):
                for f in xrange(self.factors):
                    self.Q[iid, f] *= VTW[iid, f] / HWTW[iid, f]
            self.loss = 0
            for entry in self.trainData.iterrows():
                uid, iid, rate = entry[1][0], entry[1][1], entry[1][2]
                eui = rate - self.predict(uid, iid)
                self.loss += eui ** 2
            if self.isConverged(iter):
                break

                # # use the KL divergence
                # def build_model(self):
                #     for iter in xrange(1, self.num_iters + 1):
                #         # update W by fixing H
                #         WHT = np.dot(self.P, self.Q.T)
                #         sum_f = {f: np.sum(self.Q[:, f]) for f in xrange(self.factors)}
                #         for uid in xrange(self.rateDao.num_users):
                #             for f in xrange(self.factors):
                #                 nomin = np.sum(
                #                     [self.V[uid, iid] * self.Q[iid, f] / WHT[uid, iid] for iid in xrange(self.rateDao.num_items)])
                #                 self.P[uid, f] *= nomin / sum_f[f]
                #         # update H by fixing W
                #         WHT = np.dot(self.P, self.Q.T)
                #         sum_f = {f: np.sum(self.P[:, f]) for f in xrange(self.factors)}
                #         for iid in xrange(self.rateDao.num_items):
                #             for f in xrange(self.factors):
                #                 nomin = np.sum(
                #                     [self.V[uid, iid] * self.P[uid, f] / WHT[uid, iid] for uid in xrange(self.rateDao.num_users)])
                #                 self.Q[iid, f] *= nomin / sum_f[f]
                #         self.loss = 0
                #         for entry in self.trainData.iterrows():
                #             uid, iid, rate = entry[1][0], entry[1][1], entry[1][2]
                #             eui = rate - self.predict(uid, iid)
                #             self.loss += eui ** 2
                #         if self.isConverged(iter):
                #             break


class BiasedMF(Iterative_Recommender):
    def __init__(self, rateDao, config_file):
        super(BiasedMF, self).__init__(rateDao, config_file)

    def init_model(self):
        super(BiasedMF, self).init_model()
        if self.initByNorm:
            self.user_bias = np.random.normal(Recommender.initMean, Recommender.initStd, size=self.rateDao.num_users)
            self.item_bias = np.random.normal(Recommender.initMean, Recommender.initStd, size=self.rateDao.num_items)
        else:
            self.user_bias = np.random.uniform(-1 / np.sqrt(self.factors), 1 / np.sqrt(self.factors),
                                               size=self.rateDao.num_users)
            self.item_bias = np.random.uniform(-1 / np.sqrt(self.factors), 1 / np.sqrt(self.factors),
                                               size=self.rateDao.num_items)

    def predict(self, u, i):
        return self.global_mean + self.user_bias[u] + self.item_bias[i] + super(BiasedMF, self).predict(u, i)

    def build_model(self):
        for iter in xrange(1, self.num_iters + 1):
            self.loss = 0
            for entry in self.trainData.iterrows():
                uid, iid, rate = entry[1][0], entry[1][1], entry[1][2]
                pred = self.predict(uid, iid)
                eui = rate - pred
                self.loss += eui * eui
                sgd = eui - self.regB * self.user_bias[uid]
                self.user_bias[uid] = self.feval(self.user_bias[uid], sgd)
                sgd = eui - self.regB * self.item_bias[iid]
                self.item_bias[iid] = self.feval(self.item_bias[iid], sgd)
                for f in xrange(self.factors):
                    delta_u = eui * self.Q[iid, f] - self.regU * self.P[uid, f]
                    delta_i = eui * self.P[uid, f] - self.regI * self.Q[iid, f]
                    self.P[uid, f] += self.feval(self.P[uid, f], delta_u)
                    self.Q[iid, f] += self.feval(self.Q[iid, f], delta_i)
            if self.isConverged(iter):
                break


class PMF(Iterative_Recommender):
    def __init__(self, rateDao, config_file):
        super(PMF, self).__init__(rateDao, config_file)

    def build_model(self):
        for iter in xrange(1, self.num_iters + 1):
            self.loss = 0
            for entry in self.trainData.iterrows():
                uid, iid, rate = entry[1][0], entry[1][1], entry[1][2]
                pred = self.predict(uid, iid)
                eui = rate - pred
                self.loss += eui ** 2
                delta_u = eui * self.Q[iid] - self.regU * self.P[uid]
                delta_i = eui * self.P[uid] - self.regI * self.Q[iid]
                self.P[uid] += self.lrate * delta_u
                self.Q[iid] += self.lrate * delta_i
            if self.isConverged(iter):
                break


class SVDPP(BiasedMF):
    def __init__(self, rateDao, config_file):
        super(BiasedMF, self).__init__(rateDao, config_file)

    def init_model(self):
        super(SVDPP, self).init_model()
        if self.initByNorm:
            self.Y = np.random.normal(Recommender.initMean, Recommender.initStd,
                                      size=(self.rateDao.num_items, self.factors))
        else:
            self.Y = np.random.uniform(-1 / np.sqrt(self.factors), 1 / np.sqrt(self.factors),
                                       size=(self.rateDao.num_items, self.factors))

    def predict(self, u, i):
        pred = super(SVDPP, self).predict(u, i)
        ratedItems = self.trainData[self.trainData[self.trainData.columns[0]] == u]
        nu = np.sqrt(len(ratedItems))
        for entry in ratedItems.iterrows():
            pred += np.dot(self.Y[entry[1][1]], self.Q[i]) / nu
        return pred

    def predict_with_neighbor(self, u, i, items):
        pred = super(SVDPP, self).predict(u, i)
        nu = np.sqrt(len(items))
        for entry in items.iterrows():
            pred += np.dot(self.Y[entry[1][1]], self.Q[i]) / nu
        return pred

    def build_model(self):
        columns = self.trainData.columns
        for iter in xrange(1, self.num_iters + 1):
            self.loss = 0
            for uid in np.unique(self.trainData[columns[0]].values):
                items = self.trainData[self.trainData[columns[0]] == uid]
                zu = np.zeros(self.factors)
                nu = np.sqrt(len(items))
                for entry in items.iterrows():
                    zu += self.Y[entry[1][1]] / nu
                sum = np.zeros(self.factors)
                for entry in items.iterrows():
                    iid, rate = entry[1][1], entry[1][2]
                    pred = self.predict_with_neighbor(uid, iid, items)
                    eui = rate - pred
                    self.loss += eui ** 2
                    sgd = eui - self.regB * self.user_bias[uid]
                    self.user_bias[uid] += self.lrate * sgd
                    sgd = eui - self.regB * self.item_bias[iid]
                    self.item_bias[iid] += self.lrate * sgd
                    sum += self.Q[iid] * eui / nu
                    sgd_u = eui * self.Q[iid] - self.regU * self.P[uid]
                    sgd_i = eui * (self.P[uid] + zu) - self.regI * self.Q[iid]
                    self.P[uid] += self.lrate * sgd_u
                    self.Q[iid] += self.lrate * sgd_i
                for entry in items.iterrows():
                    self.Y[entry[1][1]] += self.lrate * (sum - self.regI * self.Y[entry[1][1]])

            # for entry in self.trainData.iterrows():
            #     uid, iid, rate = entry[1][0], entry[1][1], entry[1][2]
            #     pred = self.predict(uid, iid)
            #     eui = rate - pred
            #     self.loss += eui ** 2
            #     ratedItems = self.trainData[self.trainData[self.trainData.columns[0]] == uid]
            #     nu = np.sqrt(len(ratedItems))
            #     sgd = eui - self.regB * self.user_bias[uid]
            #     self.user_bias[uid] += self.lrate * sgd
            #     sgd = eui - self.regB * self.item_bias[iid]
            #     self.item_bias[iid] += self.lrate * sgd
            #     sum_ys = np.zeros(self.factors)
            #     for f in xrange(self.factors):
            #         sum_f = np.sum([self.Y[entry[1][1], f] for entry in ratedItems.iterrows()])
            #         sum_ys[f] = sum_f / nu if nu > 0 else sum_f
            #     for f in xrange(self.factors):
            #         puf, qif = self.P[uid, f], self.Q[iid, f]
            #         sgd_u = eui * qif - self.regU * puf
            #         sgd_i = eui * (puf + sum_ys[f]) - self.regI * qif
            #         self.P[uid, f] += self.lrate * sgd_u
            #         self.Q[iid, f] += self.lrate * sgd_i
            #         for neigh in ratedItems:
            #             sgd_y = eui * qif / nu - self.regI * self.Y[neigh[1][1], f]
            #             self.Y[neigh[1][1], f] += self.lrate * sgd_y
            if self.isConverged(iter):
                break


class RankALS(Iterative_Recommender):
    '''
    Alternating Least Squares for Personalized Ranking
    '''

    def __init__(self, rateDao, config_file):
        super(RankALS, self).__init__(rateDao, config_file)
        self.isSupportWeight = bool(getConfigValue(self.cf, 'algorithm', 'sw', False))
        self.initByNorm = False

    def init_model(self):
        super(RankALS, self).init_model()
        self.s = np.zeros(self.rateDao.num_items)
        for iid in xrange(self.rateDao.num_items):
            si = len(
                self.trainData[self.trainData[self.trainData.columns[1]] == iid]) if self.isSupportWeight else 1
            self.s[iid] = si

    def build_model(self):
        columns = self.trainData.columns
        sum_s = np.sum(self.s)
        for iter in xrange(1, self.num_iters + 1):
            q_wavy = np.dot(self.Q.T, self.s)
            A_wavy = np.dot(np.dot(self.Q.T, np.diag(self.s)), self.Q)
            user_ids = np.unique(self.trainData[columns[0]].values)
            for uid in user_ids:
                r_wavy, r_bar = 0, 0
                q_bar, b_bar, b_wavy = np.zeros(self.factors), np.zeros(self.factors), np.zeros(self.factors)
                A_bar = np.zeros((self.factors, self.factors))
                items = self.trainData[self.trainData[columns[0]] == uid]
                sum_c = len(items)
                for entry in items.iterrows():
                    iid, rate = entry[1][1], entry[1][2]
                    A_bar += np.outer(self.Q[iid], self.Q[iid])
                    r_wavy += self.s[iid] * rate
                    r_bar += rate
                    q_bar += self.Q[iid]
                    b_bar += self.Q[iid] * rate
                    b_wavy += self.Q[iid] * self.s[iid] * rate
                M = A_bar * sum_s - np.outer(q_bar, q_wavy) - np.outer(q_wavy, q_bar) + A_wavy * sum_c
                y = b_bar * sum_s - q_bar * r_wavy - q_wavy * r_bar + b_wavy * sum_c
                self.P[uid] = np.dot(np.linalg.inv(M), y)
            sum_uc = np.zeros(len(user_ids))
            sum_ucq = np.zeros((len(user_ids), self.factors))
            sum_ucr = np.zeros(len(user_ids))
            sum_usr = np.zeros(len(user_ids))
            for uid in user_ids:
                items = self.trainData[self.trainData[columns[0]] == uid]
                sum_uc[uid] += len(items)
                for entry in items.iterrows():
                    iid, rate = entry[1][1], entry[1][2]
                    sum_ucq[uid] += self.Q[iid]
                    sum_ucr[uid] += rate
                    sum_usr[uid] += self.s[iid] * rate
            A_bar_bar = np.dot(np.dot(self.P.T, np.diag(sum_uc)), self.P)
            for iid in xrange(self.rateDao.num_items):
                users = self.trainData[self.trainData[columns[1]] == iid]
                A_bar = np.zeros((self.factors, self.factors))
                b_bar, b_bar_bar = np.zeros(self.factors), np.zeros(self.factors)
                p1_bar_bar, p2_bar_bar, p3_bar_bar = np.zeros(self.factors), np.zeros(self.factors), np.zeros(
                    self.factors)
                for entry in self.trainData.iterrows():
                    uid, rate = entry[1][0], entry[1][2]
                    if iid == entry[1][1]:
                        p1_bar_bar += self.P[uid] * sum_usr[uid]
                        b_bar += self.P[uid] * rate
                        A_bar += np.outer(self.P[uid], self.P[uid])
                        b_bar_bar += self.P[uid] * rate * sum_uc[uid]
                    p2_bar_bar += np.dot(np.outer(self.P[uid], self.P[uid]), sum_ucq[uid])
                    p3_bar_bar += self.P[uid] * sum_ucr[uid]
                M = A_bar * sum_s + A_bar_bar * self.s[iid]
                y = np.dot(A_bar, q_wavy) + b_bar * sum_s - p1_bar_bar + p2_bar_bar - p3_bar_bar * self.s[
                    iid] + b_bar_bar * self.s[iid]
                self.Q[iid] = np.dot(np.linalg.inv(M), y)


class PairwiseSGD(BiasedMF):
    '''
    方法与SVDPP很类似，关键在于这里对物品项的成对更新，当然也可以对用户项做同样操作
    '''

    def __init__(self, rateDao, config_file):
        super(PairwiseSGD, self).__init__(rateDao, config_file)

    def init_model(self):
        super(PairwiseSGD, self).init_model()
        if self.initByNorm:
            self.Y = np.random.normal(Recommender.initMean, Recommender.initStd,
                                      size=(self.rateDao.num_items, self.factors))
        else:
            self.Y = np.random.uniform(-1 / np.sqrt(self.factors), 1 / np.sqrt(self.factors),
                                       size=(self.rateDao.num_items, self.factors))

    def predict(self, u, i):
        pred = super(PairwiseSGD, self).predict(u, i)
        ratedItems = self.trainData[self.trainData[self.trainData.columns[0]] == u]
        nu = np.sqrt(len(ratedItems))
        for entry in ratedItems.iterrows():
            pred += np.dot(self.Y[entry[1][1]], self.Q[i]) / nu
        return pred

    def predict_with_neighbor(self, u, i, items):
        pred = super(PairwiseSGD, self).predict(u, i)
        nu = np.sqrt(len(items))
        for entry in items.iterrows():
            pred += np.dot(self.Y[entry[1][1]], self.Q[i]) / nu
        return pred

    def _getItemByProb(self, user):
        columns = self.trainData.columns
        return np.random.choice(self.trainData[self.trainData[columns[0]] != user][columns[1]].values)

    def build_model(self):
        for iter in xrange(1, self.num_iters + 1):
            self.loss = 0
            for entry in self.trainData.iterrows():
                uid, iid, rate = entry[1][0], entry[1][1], entry[1][2]
                item0 = self._getItemByProb(uid)
                ratedItems = self.trainData[self.trainData[self.trainData.columns[0]] == uid]
                nu = np.sqrt(len(ratedItems))
                pred0, pred1 = self.predict_with_neighbor(uid, iid, ratedItems), self.predict_with_neighbor(uid, item0,
                                                                                                            ratedItems)
                eui = (rate - pred0) - (rate - pred1)
                self.loss += eui ** 2
                sgd = eui - self.regB * self.item_bias[iid]
                self.item_bias[iid] += self.lrate * sgd
                sgd = eui + self.regB * self.item_bias[item0]
                self.item_bias[item0] -= self.lrate * sgd
                sum_ys = np.zeros(self.factors)
                for f in xrange(self.factors):
                    sum_f = np.sum([self.Y[entry[1][1], f] for entry in ratedItems.iterrows()])
                    sum_ys[f] = sum_f / nu if nu > 0 else sum_f
                for f in xrange(self.factors):
                    puf, qif, qi0f = self.P[uid, f], self.Q[iid, f], self.Q[item0, f]
                    sgd_u = eui * (qif - qi0f) - self.regU * puf
                    sgd_i = eui * (puf + sum_ys[f]) - self.regI * qif
                    sgd_i0 = eui * (puf + sum_ys[f]) + self.regI * qi0f
                    self.P[uid, f] += self.lrate * sgd_u
                    self.Q[iid, f] += self.lrate * sgd_i
                    self.Q[item0, f] -= self.lrate * sgd_i0
                    for neigh in ratedItems.iterrows():
                        sgd_y = eui * (qif - qi0f) / nu - self.regI * self.Y[neigh[1][1], f]
                        self.Y[neigh[1][1], f] += self.lrate * sgd_y
            if self.isConverged(iter):
                break


class LRMF(Iterative_Recommender):
    '''
    ListRank MF is a ranking method
    '''

    def __init__(self, rateDao, config_file):
        super(LRMF, self).__init__(rateDao, config_file)
        self.initByNorm = False

    def init_model(self):
        super(LRMF, self).init_model()
        self.userExp = np.zeros(self.rateDao.num_users)
        for entry in self.trainData.iterrows():
            uid, rate = entry[1][0], entry[1][2]
            self.userExp[uid] += np.exp(rate)

    def build_model(self):
        for iter in xrange(self.num_iters + 1):
            self.loss = 0
            for entry in self.trainData.iterrows():
                uid, iid, rate = entry[1][0], entry[1][1], entry[1][2]
                pred = self.predict(uid, iid)
                items = self.trainData[self.trainData[self.trainData.columns[0]] == uid]
                ugexp = 0
                for rated in items.iterrows():
                    ugexp += np.exp(g(self.predict(uid, rated[1][1])))
                self.loss -= np.exp(rate) / self.userExp[uid] * np.log(np.exp(g(pred)) / ugexp)
                for f in xrange(self.factors):
                    delta_u = (np.exp(rate) / self.userExp[uid] - np.exp(g(pred)) / ugexp) * gd(pred) * self.Q[
                        iid, f] - self.regU * self.P[uid, f]
                    delta_i = (np.exp(rate) / self.userExp[uid] - np.exp(g(pred)) / ugexp) * gd(pred) * self.P[
                        uid, f] - self.regI * self.Q[iid, f]
                    self.P[uid, f] += self.lrate * delta_u
                    self.Q[iid, f] += self.lrate * delta_i
            if self.isConverged(iter):
                break


class BPMF(Iterative_Recommender):
    '''
    Bayesian Probabilistic Matrix Factorization using Markov Chain Monte Carlo
    '''

    def __init__(self, rateDao, config_file):
        super(BPMF, self).__init__(rateDao, config_file)
        self.gibbs = 2

    def init_model(self):
        super(BPMF, self).init_model()
        self.alpha = 2
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

        self.V = np.zeros((self.rateDao.num_users, self.rateDao.num_items))
        for entry in self.trainData.iterrows():
            self.V[entry[1][0], entry[1][1]] = entry[1][2]

    def predict(self, u, i):
        return self.global_mean + super(BPMF, self).predict_inbound(u, i, True)

    def predict_batch(self, user_ids, item_ids):
        return np.array([self.predict(uid, iid) for uid, iid in zip(user_ids, item_ids)])

    def _update_user_params(self):
        N = self.rateDao.num_users
        X_bar = np.mean(self.P, 0)
        S_bar = np.cov(self.P.T)
        norm_X_bar = self.mu0_u - X_bar
        WI_post = self.WI_user + N * S_bar + np.outer(norm_X_bar) * (N * self.beta_user) / (self.beta_user + N)
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
        WI_post = self.WI_item + N * S_bar + np.outer(norm_X_bar) * (N * self.beta_item) / (self.beta_item + N)
        WI_post = (WI_post + WI_post.T) / 2.
        df_post = self.df_item + N
        self.lambda_item = wishart.rvs(df_post, WI_post)
        mu_temp = (self.beta_item * self.mu0_i + N * X_bar) / (self.beta_item + N)
        sigma_temp = np.linalg.inv(np.dot(self.beta_item + N, self.lambda_item))
        self.mu_item = mv_normalrand(mu_temp, sigma_temp, self.factors)

    def _update_user_features(self):
        for uid in xrange(self.rateDao.num_users):
            vec = self.V[uid, :] > 0
            features = self.Q[vec]
            rating = self.V[uid, vec] - self.global_mean
            sigma = np.linalg.inv(self.lambda_user + self.alpha * np.dot(features.T, features))
            mu_temp = self.alpha * np.dot(features.T, rating) + np.dot(self.lambda_user, self.mu_user)
            mean = np.dot(sigma, mu_temp)
            self.P[uid] = mv_normalrand(mean, sigma, self.factors)

    def _update_item_features(self):
        for iid in xrange(self.rateDao.num_items):
            vec = self.V[:, iid] > 0
            features = self.P[vec]
            rating = self.V[vec, iid] - self.global_mean
            sigma = np.linalg.inv(self.lambda_item + self.alpha * np.dot(features.T, features))
            mu_temp = self.alpha * np.dot(features.T, rating) + np.dot(self.lambda_item, self.mu_item)
            mean = np.dot(sigma, mu_temp)
            self.Q[iid] = mv_normalrand(mean, sigma, self.factors)

    def build_model(self):
        columns = self.columns
        for iter in xrange(1, self.num_iters + 1):
            self._update_user_params()
            self._update_item_params()
            for _ in xrange(self.gibbs):
                self._update_user_features()
                self._update_item_features()
            rates = self.trainData[columns[2]]
            preds = self.predict_batch(self.trainData[columns[0]], self.trainData[columns[1]])
            self.loss = (rates - preds) ** 2
            if self.isConverged(iter):
                break


class WBPR(Iterative_Recommender):
    '''
    Bayesian Personalized Ranking for Non-Uniformly Sampled Items
    '''

    def __init__(self, rateDao, config_file):
        super(WBPR, self).__init__(rateDao, config_file)

    def init_model(self):
        super(WBPR, self).init_model()
        if self.initByNorm:
            self.item_bias = np.random.normal(Recommender.initMean, Recommender.initStd, size=self.rateDao.num_items)
        else:
            self.item_bias = np.random.uniform(-1 / np.sqrt(self.factors), 1 / np.sqrt(self.factors),
                                               size=self.rateDao.num_items)
        columns = self.rateDao.columns
        sortedItemProbs = sorted({iid: len(self.trainData[self.trainData[columns[1]] == iid]) for iid in
                                  xrange(self.rateDao.num_items)}.iteritems(), key=lambda pair: pair[1],
                                 reverse=True)
        self.invUserProbs = {}
        self.userItemProbs = {}
        for uid in xrange(self.rateDao.num_users):
            ratedItems = self.trainData[self.trainData[columns[0]] == uid]
            sum = 0.0
            self.invUserProbs[uid] = 1. / len(ratedItems)
            itemProbs = []
            for iid, cnt in sortedItemProbs:
                if iid in ratedItems[columns[1]]:
                    itemProbs.append((iid, cnt))
                    sum += cnt
            itemProbs = map(lambda pair: pair[1] / sum, itemProbs)
            self.userItemProbs[uid] = np.array(itemProbs)
        sum = np.sum(self.invUserProbs.values())
        for uid in self.invUserProbs.keys():
            self.invUserProbs[uid] /= sum

    def predict(self, u, i):
        return self.item_bias[i] + super(WBPR, self).predict(u, i)

    def build_model(self):
        for iter in xrange(1, self.num_iters + 1):
            self.loss = 0
            for _ in xrange(self.rateDao.num_users * 10):
                u, i, j = 0, 0, 0
                while True:
                    # 根据用户评价商品数量的倒数概率来采样用户
                    u = np.random.choice(self.invUserProbs.keys(), 1, p=self.invUserProbs.values())
                    # u = np.random.randint(0, self.rateDao.num_users)
                    if len(self.userItemProbs[u]) == 0:
                        continue
                    i = np.random.choice(self.userItemProbs[u][:, 0], 1)[0]
                    j = np.random.choice(self.userItemProbs[u][:, 0], 1, p=self.userItemProbs[u][:, 1])[0]
                    break
                xui = self.predict(u, i)
                xuj = self.predict(u, j)
                xuij = xui - xuj
                self.loss += np.log(g(xuij))
                cmg = g(-xuij)
                bi, bj = self.item_bias[i], self.item_bias[j]
                self.item_bias[i] += -self.lrate * (cmg - self.regB * bi)
                self.item_bias[j] += self.lrate * (cmg + self.regB * bj)
                pu, qi, qj = np.copy(self.P[u]), np.copy(self.Q[i]), np.copy(self.Q[j])
                self.P[u] += -self.lrate * (cmg * (qi - qj) - self.regU * pu)
                self.Q[i] += -self.lrate * (cmg * pu - self.regI * qi)
                self.Q[j] += self.lrate * (cmg * pu + self.regI * qj)
            if self.isConverged(iter):
                break


class FISM(Iterative_Recommender):
    '''
    Factored Item Similarity Models for Top-N Recommender Systems
    '''

    def __init__(self, rateDao, config_file):
        super(FISM, self).__init__(rateDao, config_file)
        self.p = float(getConfigValue(self.cf, 'learn', 'p', 3))
        self.alpha = float(getConfigValue(self.cf, 'learn', 'alpha', 0.1))

    def init_model(self):
        self.P = np.random.uniform(-1. / self.factors, 1. / self.factors,
                                   (self.rateDao.num_items, self.factors))
        self.Q = np.random.uniform(-1. / self.factors, 1. / self.factors,
                                   (self.rateDao.num_items, self.factors))
        self.user_bias = np.random.uniform(-1. / self.factors, 1. / self.factors,
                                           size=self.rateDao.num_users)
        self.item_bias = np.random.uniform(-1. / self.factors, 1. / self.factors,
                                           size=self.rateDao.num_items)

    def predict(self, u, i):
        pred = self.user_bias[u] + self.item_bias[i]
        sum, count = 0, 0
        for entry in self.trainData[self.trainData[self.columns[0]] == u]:
            if entry[1][1] != i:
                sum += np.dot(self.P[u], self.Q[entry[1][1]])
                count += 1
        wu = 0 if count == 0 else count ** (-self.alpha)
        return pred + wu * sum

    def build_model(self):
        sample_size = self.p * len(self.trainData)
        for iter in xrange(1, self.num_iters + 1):
            self.loss = 0
            train_copy = self.trainData.copy(True)
            for uid, iid in zip(np.random.shuffle(range(self.rateDao.num_users)),
                                np.random.shuffle(range(self.rateDao.num_items))):
                if self.trainData[
                            (self.trainData[self.columns[0]] == uid) & (self.trainData[self.columns[1]] == iid)].empty:
                    train_copy.loc[len(train_copy)] = [uid, iid, 0]
                if len(self.trainData) == sample_size:
                    break
            for entry in train_copy.iterrows():
                uid, iid, rate = entry[1][0], entry[1][1], entry[1][2]
                bu, bi = self.user_bias[uid], self.item_bias[iid]
                ratedItems = train_copy[train_copy[self.columns[0]] == uid]
                sum, cnt = 0, 0
                x = np.zeros(self.factors)
                for item in ratedItems.iterrows():
                    if item[1][1] != iid:
                        sum += np.dot(self.P[uid], self.Q[item[1][1]])
                        cnt += 1
                        x += self.P[item[1][1]]
                wu = 0 if cnt == 0 else cnt ** (-self.alpha)
                pred = bu + bi + wu * sum
                eui = pred - rate
                self.loss += eui ** 2
                self.user_bias[uid] += -self.lrate * (eui + self.regB * self.user_bias[uid])
                self.item_bias[iid] += -self.lrate * (eui + self.regB * self.item_bias[iid])
                self.Q[iid] += -self.lrate * (eui * wu * x + self.regI * self.Q[iid])
                for item in ratedItems.iterrows():
                    if item[1][1] != iid:
                        self.P[item[1][1]] += -self.lrate * (eui * wu * self.Q[iid] + self.regI * self.P[item[1][1]])
            if self.isConverged(iter):
                break


class CLiMF(Iterative_Recommender):
    '''
    learning to maximize reciprocal rank with collaborative less-is-more filtering
    the prediction is a binary classification's probability
    '''

    def __init__(self, rateDao, config_file):
        super(CLiMF, self).__init__(rateDao, config_file)

    def build_model(self):
        for iter in xrange(1, self.num_iters + 1):
            self.loss = 0
            for uid in xrange(self.rateDao.num_users):
                ratedItems = self.trainData[self.trainData[self.columns[0]] == uid]
                sgd_u = -self.regU * self.P[uid]
                for item in ratedItems.iterrows():
                    j = item[1][1]
                    fuj = self.predict(uid, j)
                    sgd_u += g(-fuj) * self.Q[j]
                    sgd_i = g(-fuj)
                    for neigh in ratedItems.iterrows():
                        k = neigh[1][1]
                        if k == j:
                            continue
                        fuk = self.predict(uid, k)
                        x = fuk - fuj
                        sgd_u += gd(x) * (self.Q[j] - self.Q[k]) / (1 - g(x))
                        sgd_i += gd(-x) * (1. / (1 - g(x)) - 1. / (1 - g(-x)))
                    sgd_i = sgd_i * self.P[uid] - self.regI * self.Q[j]
                    self.Q[j] += self.lrate * sgd_i
                self.P[uid] += self.lrate * sgd_u
                for item in ratedItems.iteritems():
                    j = item[1][1]
                    fuj = self.predict(uid, j)
                    self.loss += np.log(g(fuj))
                    for neigh in ratedItems.iterrows():
                        k = neigh[1][1]
                        if j == k:
                            continue
                        fuk = self.predict(uid, k)
                        self.loss += np.log(1 - g(fuk - fuj))
            if self.isConverged(iter):
                break
