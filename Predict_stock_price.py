import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
import scipy, scipy.stats
import matplotlib.pyplot as plt

print('hello world')


def print_hello(message):
    path = r'aapl.csv'
    df = pd.read_csv(path)
    data = np.matrix(df)
    cln = LinearRegression()
    org = LinearRegression()
    # X is Volume and Y is Price
    X, Y = data[:, 5], data[:, 4]
    cln.fit(X[:-1], Y[:-1])
    org.fit(X, Y)

    clean_score = '{0:.3f}'.format(cln.score(X[:-1], Y[:-1]))
    original_score = '{0:.3f}'.format(org.score(X, Y))
    #Prdicting stock price when volume is
    print(org.predict(26495312))
    print(original_score)


print_hello('hello')
