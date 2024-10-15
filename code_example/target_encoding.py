import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin

class TargetEncode(BaseEstimator,TransformerMixin):
    def __init__(self, categories='auto',k=1,f=1,noise_level=0,random_state=None):
        if type(categories) == str and categories != 'auto':
            self.categories = [categories]
        else:
            self.categories = categories
        self.k = k
        self.f = f
        self.noise_level = noise_level
        self.encodings = dict()
        self.prior = None
        self.random_state = random_state

    def add_noise(self, series, noise_level):
        return series*(1+noise_level)
    
    def fit(self, X, y=None):
        if type(self.categories) == 'auto':
            self.categories = np.where(X.dtypes == type(object()))[0]

        temp = X.loc[:, self.categories].copy()
        temp['target'] = y
        self.prior = np.mean(y)
        for variable in self.categories:
            avg = (temp.groupby(by=variable))['target'].agg(['mean','count'])
            smoothing = ( 1/ (1+np.exp( -(avg['count'] - self.k)/self.f ) ) )
            self.encodings[variable] = dict(self.prior * (1-smoothing) + avg['mean'] * smoothing)

        return self
    
    def transform(self, X):
        Xt = X.copy()
        for variable in self.categories:
            Xt[variable].replace(self.encodings[variable], inplace=True)
            unkown_value = {value:self.piror for value in X[variable].unique() if value not in self.encodings.keys()}

            if len(unkown_value) > 0:
                Xt[variable].replace(unkown_value,inplace=True)
            Xt[variable] = Xt[variable].astype(float)
            if self.noise_level > 0:
                if self.random_state is not None:
                    np.random.seed(self.random_state)
                Xt[variable] = self.add_noise(Xt[variable], self.noise_level)

        return Xt
    
    def fit_transform(self, X, y=None):
        self.fit(X,y)
        return self.transform(X)