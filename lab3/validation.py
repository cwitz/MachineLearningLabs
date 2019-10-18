import pandas as pd
import numpy as np
import math, sys


class Node:
   def __init__(self, label = None):
      self.children = []
      self.edgelabels = []
      self.label = label

   def getLabel(self):
      return self.label

   def setLabel(self, label):
      self.label = label

   def addChild(self, node, edgelabel):
      self.children.append(node)
      self.edgelabels.append(edgelabel)

   def printNode(self, indentLevel = 0):
      print("  " * indentLevel + f"Node label: {self.label}")
      if len(self.children) > 0:
         print("  " * indentLevel + f"Children of {self.label}:\n")
         for i in range(len(self.children)):
            print("  " * (indentLevel + 1) + f"Edge: {self.edgelabels[i]}")
            self.children[i].printNode(indentLevel + 1)
         print("  " * indentLevel + f"End children of {self.label}")
      print()

class Attribute:
   def __init__(self, attrName, domain, index):
      self.attrName = attrName
      self.domain = domain
      self.index = index

   def __eq__(self, other):
      return self.attrName == other.attrName


#def c45(D, A, threshold, classAttr, Ddataframe):
def c45(D, A, threshold, classAttr):
   r = Node()

   # Check purity
   """
   pure = True
   classAttrIndex = classAttr.index
   curClass = D[0][classAttrIndex]
   for datapoint in D:
      if datapoint[classAttrIndex] != curClass:
         pure = False
         break
   """
   pure = len(set(D.loc[:, classAttr.attrName].tolist())) == 1
   if pure:
      curClass = D.loc[:, classAttr.attrName].iloc[0]
      r.setLabel(curClass)

   # Check if no attributes to split on
   elif len(A) == 0:
      c = find_most_frequent_label(D, classAttr)
      r.setLabel(c)

   # Select splitting attribute
   else:
      attrNames = [att.attrName for att in A]
      #Ag = selectSplittingAttribute(attrNames, Ddataframe, classAttr.domain,
      #                              classAttr.attrName, threshold)
      Ag = selectSplittingAttribute(attrNames, D, classAttr.domain,
                                    classAttr.attrName, threshold)
      if Ag is None:  # No attributes remaining to split on
         c = find_most_frequent_label(D, classAttr)
         r.setLabel(c)
      else:   # Construct and return tree
         r.setLabel(Ag)
         for attr in A:
            if attr.attrName == Ag:
               Ag = attr
               break
         for v in Ag.domain:
            #Dv = []
            #for t in D:
            #   if t[Ag.index] == v:
            #      Dv.append(t)
            #Ddataframev = Ddataframe[Ddataframe[Ag.attrName] == v]
            Dv = D[D[Ag.attrName] == v]
            if len(Dv) > 0:
               remainingAttrs = A.copy()
               for i in range(len(remainingAttrs)):
                  if remainingAttrs[i] == Ag:
                     del remainingAttrs[i]
                     break
               #Tv = c45(Dv, remainingAttrs, threshold, classAttr,
               #         Ddataframev)
               Tv = c45(Dv, remainingAttrs, threshold, classAttr)
               r.addChild(Tv, v)

   return r


def find_most_frequent_label(data, classAttr):
   #classIndex = classAttr.index
   #classes = [dp[classIndex] for dp in data]
   classes = data.loc[:, classAttr.attrName].tolist()
   return max(set(classes), key = classes.count)


def selectSplittingAttribute(A, D, C, result, threshold):
  p = []
  gain = []
  gain.append(0)
  p.append(entropyAll(D, C, result))
  for i in range(len(A)):
    a = list(dict.fromkeys(D.loc[:, A[i]]))
    p.append(entropy(D, a, C, A[i], result))
    gain.append(p[0] - p[len(p) - 1])
  best = max(gain)
  if(best > threshold):
    return A[gain.index(best) - 1]
  return None


def entropyAll(D, C, result):
   sum = [0] * len(C)
   for d in range(len(D)):
      for c in range(len(C)):
         if(D.iloc[d].loc[result] == C[c]):
            sum[c] += 1

   entropy = 0
   for i in range(len(C)):
      part = (sum[i] / float(len(D)))
      if part < .00001:
         entropy += 0.0
      else:
         entropy += (-1 * part * math.log(part, 2))
   return entropy


#aName is the string that is the attribute name of A
#A is specifc attribute, list containing each category of attribute
def entropy(D, A, C, aName,  result):
  sums = []
  for i in range(len(A)):
    sums.append([0] * len(C))
  for d in range(len(D)):
    if(isinstance(A, list)):
      for i in range(len(A)):
        row = 0
        row = D.iloc[d]
        if(row.loc[aName] == A[i]):
          sumI = sums[i]
          for c in range(len(C)):
            if(row.loc[result] == C[c]):
              sumI[c] += 1
              break
          break
    else:
      return 0#implementation if not a list here

  pieces = []
  for s in sums:
    size = sum(s)
    sub = []
    for i in range(len(C)):
      part = (s[i] / size)
      if(part != 0):
        sub.append(-1 * part * math.log(part, 2))

    pieces.append((sum(sub)) * (size / len(D)))
  return sum(pieces)


def treeToJson(tree):
  jStr = ""
  if(len(tree.children) != 0):
    jStr += "\"node\": {\n  \"var\": "
    jStr += '"' +  tree.label + "\","
    jStr += "\n\"edges\": [ \n"
    for i in range(len(tree.edgelabels)):
      jStr += "{\"value\":"
      jStr += '"' + tree.edgelabels[i] + '"' + ","
      jStr += treeToJson(tree.children[i])
      jStr += "}"
      if(i < len(tree.edgelabels) - 1):
        jStr += ","
    jStr += "]"
    jStr += "}"
  else:
    jStr += "\"leaf\":{\"decision\": "
    jStr += '"' + tree.label + "\""
    jStr += "}"
  return jStr


def traverseTree(tree, row):
  #if its a leaf
  if (len(tree.children) == 0):
    return tree.label
  for i in range(len(tree.children)):
    if(tree.edgelabels[i] == row.loc[tree.label]):
      return traverseTree(tree.children[i], row)
  print("ERROR TRAVERSE TREE")
  print(f"Row:\n{row}")
  print(f"Tree label: {tree.label}")
  print(f"Edges: {tree.edgelabels}\nChildren: {tree.children}")
  exit()


#################################################
# Main program
def main():
   args = sys.argv
   threshold = .25

   ### Print usage if args have wrong arity
   if len(args) < 3 or len(args) > 4:
      print("Usage: python3 validation.py <TrainingSetFile.csv> " +
            "<n-folds> [<restrictionsFile>]")
      exit() 

   ### Parse restrictions
   restrictions = None
   try:
      if len(args) == 4:
         restrline = open(args[3], 'r').readlines()[0]
         restrictions = restrline.strip().split(",")
         restrictions = [r.strip() for r in restrictions]
   except:
      print("Error opening restrictionsFile")
      exit()

   ### Parse input csv
   datasetcsv = datasetfile = 0
   try:
      datasetcsv = pd.read_csv(args[1])
      datasetfile = open(args[1], 'r')
   except:
      print(f"Error opening input file: {args[1]}")
      exit()
   Ddataframe = datasetcsv.loc[2:]
   dataset = [line.strip().split(',') for line in datasetfile]
   
   ### Parse n-folds
   nFolds = -2
   try:
      nFolds = int(args[2])
   except:
      print("n-folds must be an integer")
      exit()
   if nFolds < -1:
      print("n-folds must be -1 (leave-one-out), 0 (no cross validation), "+
            "or >= 1")
      exit()

   ### Get list of attributes and their domains
   attrNames = dataset[0]
   attrTypes = [int(n) for n in dataset[1]]
   classAttrName = dataset[2][0]

   classDomain = set([])
   attributelist = []
   for colnum in range(len(attrTypes)):
      attrType = attrTypes[colnum]
      if attrType < 1:
         # if -1, not to be counted in classification
         # if 0, numeric attribute; To Be Implemented
         pass
      else:
         attrName = attrNames[colnum]
         if attrName == classAttrName: # Class attribute
            for dp in dataset[3:]:
               classDomain.add(dp[colnum])
            if len(classDomain) > attrType: # Domain exception
               print(f"Size of domain conflict for attribute: {attrName}")
               exit()

         else: # Qualitative attribute
            attrDomain = set([])
            for dp in dataset[3:]:
               attrDomain.add(dp[colnum])
            if len(attrDomain) > attrType: # Domain exception
               print(f"Size of domain conflict for attribute: {attrName}")
               print(f"Extracted domain: {attrDomain}")
               exit()
            attributelist.append(Attribute(attrName, list(attrDomain), colnum))

   classAttrIndex = attrNames.index(classAttrName)
   classAttr = Attribute(classAttrName, list(classDomain), classAttrIndex)
   dataset = dataset[3:]

   ### Abide by restrictions
   if restrictions is not None:
      for col in range(len(attributelist) - 1, -1, -1):
         if restrictions[col] == "1": # Consider attribute in decision tree
            pass
         elif restrictions[col] == "0": # Omit attribute from decision tree
            del attributelist[col]
         else: 
            print("Invalid format for restrictionFile.")
            exit()

   ### Cross validation
   classDom = list(classDomain)

   if nFolds == -1: # all-but-one validation
      folds = np.array_split(Ddataframe, len(Ddataframe))
      right = wrong = 0.0
      avgAcc = [0.0] * len(folds)
      avgErr = [0.0] * len(folds)
      confMatrix=[[0] * len(classDomain) for n in range(len(classDomain))]
      for f in range(len(folds)):
        testSet = folds[f]
        trainSets = folds[:]
        del trainSets[f]

        curRight = curWrong = 0.0
        for fold in trainSets:
          dTree = c45(fold, attributelist, threshold, classAttr)
          row = testSet.iloc[0]
          guess = traverseTree(dTree, row)

          x = classDom.index(guess)
          y = classDom.index(row[classAttrName])
          confMatrix[x][y] += 1
          if guess == row[classAttrName]:
             right += 1
             curRight += 1
          else:
             wrong += 1
             curWrong += 1
        
        avgAcc[f] = curRight / len(trainSets)
        avgErr[f] = curWrong / len(trainSets)

      print(f"Confusion matrix:\n{classDom}")
      for row in range(len(confMatrix)):
         print(f"{classDom[row]}: {confMatrix[row]}")
      
      print(f"Overall accuracy: {right / (len(avgAcc)*(len(folds)-1))}")
      print(f"Average accuracy: {sum(avgAcc) / len(avgAcc)}")
      print(f"Overall error rate: {wrong / (len(avgErr)*(len(folds)-1))}")
      print(f"Average error rate: {sum(avgErr) / len(avgErr)}")
   elif nFolds == 0 or nFolds == 1: # 1-fold validation
      dTree = c45(Ddataframe, attributelist, threshold, classAttr)
      
      confMatrix=[[0] * len(classDomain) for n in range(len(classDomain))]
      
      right = wrong = 0.0
      for d in range(len(Ddataframe)):
         row = Ddataframe.iloc[d]
         guess = traverseTree(dTree, row)

         x = classDom.index(guess)
         y = classDom.index(row[classAttrName])
         confMatrix[x][y] += 1
         if guess == row[classAttrName]:
            right += 1
         else:
            wrong += 1

      print(f"Confusion matrix:\n{classDom}")
      for i in range(len(confMatrix)):
         print(f"{classDom[i]}: {confMatrix[i]}")
      
      print(f"Overall/Average accuracy: {right / len(Ddataframe)}")
      print(f"Overall/Average error rate: {wrong / len(Ddataframe)}")

   else: # n-fold validation
      Dshuffled = Ddataframe.sample(frac = 1)
      folds = np.array_split(Dshuffled, nFolds) 
      right = wrong = 0.0
      avgAcc = [0.0] * len(folds)
      avgErr = [0.0] * len(folds)
      confMatrix=[[0] * len(classDomain) for n in range(len(classDomain))]
      
      for f in range(len(folds)):
        testSet = folds[f]
        trainSets = folds[:]
        del trainSets[f]

        curRight = curWrong = 0.0
        for fold in trainSets:
          dTree = c45(fold, attributelist, threshold, classAttr)

          for test in range(len(testSet)):
             row = testSet.iloc[test]
             guess = traverseTree(dTree, row)
          
             x = classDom.index(guess)
             y = classDom.index(row[classAttrName])
             confMatrix[x][y] += 1
             if guess == row[classAttrName]:
                right += 1
                curRight += 1
             else:
                wrong += 1
                curWrong += 1
        
        avgAcc[f] = curRight / (len(trainSets) * len(testSet))
        avgErr[f] = curWrong / (len(trainSets) * len(testSet))

      print(f"Confusion matrix:\n{classDom}")
      for i in range(len(confMatrix)):
         print(f"{classDom[i]}: {confMatrix[i]}")
      
      print(f"Overall accuracy: {right / (right + wrong)}")
      print(f"Average accuracy: {sum(avgAcc) / len(avgAcc)}")
      print(f"Overall error rate: {wrong / (right + wrong)}")
      print(f"Average error rate: {sum(avgErr) / len(avgErr)}")

     
main()
