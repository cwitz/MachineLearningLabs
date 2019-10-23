import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
import sys

def euc(d1, d2, A, result):
  dist = 0
  for a in range(len(A) - 1):
    if(isinstance(d1[a], str) or isinstance(d2[a], str)):
      continue
    dist += math.pow(float(d1[a]) - float(d2[a]), 2)
  return (math.sqrt(dist), d2.loc[result])

def main():
  args = sys.argv
  k = int(args[2])
  D = pd.read_csv(args[1])
  #name of Y feature
  result = D.loc[1, D.columns[0]]
  #cut out formatting stuff from data
  #print(D.loc[0])
  arity = D.loc[0]
  isCont = []
  for i in D.loc[0]:
    isCont.append(0);
  for i in range(len(arity)):
    if(int(arity[i]) < 0):
      del D[D.columns[i]]
      i -= 1
    elif(int(arity[i]) == 0):
      isCont[i] = 1

  D = D.loc[2: ]
  #every possible Y
  C = D.loc[:, result]
  C = list(dict.fromkeys(D.loc[:, result]))
  #names of every feature
  A = D.columns
  A = A.drop(result)
  #format data
  D = D.reset_index()
  label_encoder = LabelEncoder()
  newD = pd.DataFrame();
  for i in range(len(A) - 1):
    if(not(isCont[i])):
      integer_encoded = label_encoder.fit_transform(D.loc[ : , A[i]])
      newD[A[i]] = integer_encoded
    else:
      newD[A[i]] = D.loc[ : , A[i]]
  newD[result] = D.loc[ : , result]

  #KNN
  guesses = []
  #print(newD)
  #loop through D to classify each point
  for j in range(len(D)):
    #list of tuples containing distance and result for point d (below)
    dist = []
    #loop through D to find distance of each point(d)
    for d in range(len(D)):
      if d == j:
        continue
      d1 = newD.loc[j, : ]
      d2 = newD.loc[d, : ]
      #find distance of d, add it to dist
      dist.append(euc(d1, d2, A, result))
    dist.sort()
    #print(dist)
    results = []
    for i in range(len(C)):
      results.append(0)
    #find the plurality of the results of the k nearest neighbors
    for i in range(k):
      currD = dist[i]       #currD[1] refers to result part of dist tuple
      #print(currD[1])
      results[C.index(currD[1])] += 1
    guesses.append(C[results.index(max(results))])

  right = 0
  for i in range(len(D)):
    if guesses[i] == newD.loc[i, result]:
      right += 1
  for i in range(len(guesses)):
    print("row ", i, ": ", guesses[i])
  print("accuracy: ", right / len(D))
main()
