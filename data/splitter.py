# coding=UTF-8
'''
@author Gray
'''
import copy
import pandas as pd
import numpy as np


def splitRatioByRating(ratio, rateData):
    assert ratio > 0 and ratio < 1
    trainData = pd.DataFrame(columns=rateData.columns)
    testData = pd.DataFrame(columns=rateData.columns)
    for entry in rateData.iterrows():
        rdm = np.random.random()
        if rdm < ratio:
            testData.loc[len(testData)] = entry[1]
        else:
            trainData.loc[len(trainData)] = entry[1]
    return trainData, testData


def splitRatioByUser(ratio, rateDao):
    assert ratio > 0 and ratio < 1
    trainData = pd.DataFrame(columns=rateDao.data.columns)
    testData = pd.DataFrame(columns=rateDao.data.columns)
    uid_name = rateDao.data.columns[0]
    for uid in np.unique(rateDao.data[uid_name].values):
        for entry in rateDao.data[rateDao.data[uid_name] == uid].iterrows():
            rdm = np.random.random()
            if rdm < ratio:
                testData.loc[len(testData)] = entry[1]
            else:
                trainData.loc[len(trainData)] = entry[1]
    return trainData, testData


def splitRatioByItem(ratio, rateDao):
    assert ratio > 0 and ratio < 1
    trainData = pd.DataFrame(columns=rateDao.data.columns)
    testData = pd.DataFrame(columns=rateDao.data.columns)
    iid_name = rateDao.data.columns[1]
    for iid in np.unique(rateDao.data[iid_name].values):
        for entry in rateDao.data[rateDao.data[iid_name] == iid].iterrows():
            rdm = np.random.random()
            if rdm < ratio:
                testData.loc[len(testData)] = entry[1]
            else:
                trainData.loc[len(trainData)] = entry[1]
    return trainData, testData
