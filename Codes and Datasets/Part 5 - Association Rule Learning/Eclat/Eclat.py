# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:01:41 2020

@author: Subhasmita
"""
"""Importing the libraries"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""Importing the dataset & Data Preprocessing"""
df = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(len(df)):
  transactions.append([str(df.values[i,j]) for j in range(len(df.iloc[i]))])

"""Training the Apriori Model"""
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

"""Displaying the first results coming directly from the output of the apriori function"""
results = list(rules)
print(results)

"""Putting the results well organised into a Pandas DataFrame"""
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support'])

"""Displaying the results non sorted"""
print(resultsinDataFrame)

"""Displaying the results in sorted"""
print(resultsinDataFrame.nlargest(n=10,columns='Support'))