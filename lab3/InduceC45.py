import pandas as pd
import numpy as np
import math, sys
import random as rd


class Node:
   def __init__(self, label = None, isCont = False):
      self.children = []
      self.edgelabels = []
      self.label = label
      self.isCont = isCont

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
   def __init__(self, attrName, domain, index, isCont = False):
      self.attrName = attrName
      self.domain = domain
      self.index = index
      self.isCont = isCont

   def __eq__(self, other):
      return self.attrName==other.attrName and self.isCont==other.isCont


def c45(D, A, threshold, classAttr):
   #print("Entering c45 with attributes:")
   #for a in A:
   #   print(a.attrName)
   r = Node()

   # Check purity
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
      attrNames = []
      isConts = []
      for att in A:
         attrNames.append(att.attrName)
         isConts.append(att.isCont)
      attrNames = [att.attrName for att in A]
      #print("Splitting Attr...")
      Ag = selectSplittingAttribute(attrNames, D, classAttr.domain,
                                    classAttr.attrName, threshold, isConts)
      #print(f"Done splitting! Attribute is {Ag[0]}\nBest binary split " +
      #      f"is: {Ag[1]}")
      Agatt = Ag[0]
      if Agatt is None:  # No attributes remaining to split on
         c = find_most_frequent_label(D, classAttr)
         r.setLabel(c)
      else:   # Construct and return tree
         r.setLabel(Agatt)
         for attr in A:
            if attr.attrName == Agatt:
               Agatt = attr
               break
         if Agatt is None:
            print(f"Error building tree: split attribute {Agatt} still" +
                  " none after looking through attributes:")
            for attr in A:
               print(attr.attrName)
               print(attr.domain)
            exit()
         if Agatt.isCont:
            r.isCont = True
            bestBinSplit = float(Ag[1])
            #print(f"Dataset:\n{D}")
            
            #print("Splitting D along best split...")
            #print(f"\n\n\n\nBReaking:\n{D[Agatt.attrName]}")
            Dless = D[pd.to_numeric(D[Agatt.attrName]) <= bestBinSplit]
            Dgreater = D[pd.to_numeric(D[Agatt.attrName]) > bestBinSplit]
            #print("Success")
            #print(f"Dless:\n{Dless}")
            #print(f"Dmore:\n{Dgreater}")
            #print("Done splitting best split!")
      
            if len(Dless) < 0 or len(Dgreater) < 0:
               print("ERROR: Bad numeric attribute split")
               
            trLess = c45(Dless, A, threshold, classAttr)
            r.addChild(trLess, f"<= {bestBinSplit}")
            trGreater = c45(Dgreater, A, threshold, classAttr)
            r.addChild(trGreater, f"> {bestBinSplit}")
         else:
            for v in Agatt.domain:
               Dv = D[D[Agatt.attrName] == v]
               if len(Dv) > 0:
                  remainingAttrs = A.copy()
                  for i in range(len(remainingAttrs)):
                     if remainingAttrs[i] == Agatt:
                        del remainingAttrs[i]
                        break
               
                  Tv = c45(Dv, remainingAttrs, threshold, classAttr)
                  r.addChild(Tv, v)
         
   return r


def find_most_frequent_label(data, classAttr):
   classes = data.loc[:, classAttr.attrName].tolist()
   return max(set(classes), key = classes.count)


def selectSplittingAttribute(A, D, C, result, threshold, isCont):  
  p = []
  gain = []
  gain.append(0)
  #print("Entropy 1")
  bestBinSplit = [None] * len(A)
  p.append(entropyAll(D, C, result))
  for i in range(len(A)):
    #print(f"Analyzing {A[i]} for att split...")
    a = list(dict.fromkeys(D.loc[:, A[i]]))
    if(not(isCont[i])):
      #print("not cont")
      #print("Entropy 2")
      p.append(entropy(D, a, C, A[i], result))
    else:
      #print("Continuous")
      #print("Finding best split...")
      best = findBestSplit(A[i], D, C, i, result)
      bestBinSplit[i] = best[1]
      #print(f"best: {best}")
      #print("Best split found!")
      p.append(best[0])
    #print("Gain append")
    #print(f"p[0]: {p[0]}")
    #print(f"Gain: {p[0] - p[len(p) - 1]}")
    gain.append(p[0] - p[len(p) - 1])
  best = max(gain)
  if(best > threshold):
    bestIndex = gain.index(best) - 1
    return (A[bestIndex], bestBinSplit[bestIndex])
  return (None, None)


#a is name of continuous attribute
def findBestSplit(a, D, C, i, result):
  sortedD = D.copy()
  sortedD = sortedD.sort_values(by=a)
  results = sortedD.loc[ : , result].tolist()
  aData = sortedD.loc[ : , a]
  #extract data and sort, then do this
  counts = []
  for i in C:
    counts.append(dict())
  gain = []
  #print("Calcing all entropy...")
  p = entropyAll(D, C, result)
  #print("Calced all entropy!")
  #print("Entopy all: ", p)
  for d in range(len(D)):
    #print(aData.loc[d + 2])
    for j in range(len(C)):
      count = counts[j]
      #print("results[d]: ", results[d], " | C[j]: ", C[j])
      if(results[d] == C[j]):
        #print("Another loc...")
        key = str(aData.iloc[d])
        #print("After loc")
        if key in count:
          count[key] = int(count[key]) + 1
        else:
          count[key] = str(1)
        #print("after: ", count)
        break
  splitVals = []
  for m in counts:
    for x in m:
      gain.append(p - entropyCon(D, a, x, counts, C, result))
      splitVals.append(x)
      #print(gain[len(gain) - 1])
  best = max(gain)
  #print(best)
  return (-(best - p), splitVals[gain.index(best)])


#x: value splitting by
#counts contains a dictionary per a result in C with
def entropyCon(D, a, x, counts, C, result):
  #create results array
  results = []
  oneRes = []
  for i in C:
    oneRes.append(0)
  results.append(oneRes)
  results.append(oneRes.copy())
  #logic
  #print("Counts: ", counts)
  #print("x: ", x)
  for i in range(len(counts)):
    for c in counts[i]:
      count = counts[i]
      if float(c) <= float(x):
        curr = results[0]
        curr[i] += int(count[c])
      else:
        curr = results[1]
        curr[i] += int(count[c])
  #print("Results: ", results)
  pieces = []
  for s in results:
    size = sum(s)
    if(size == 0):
      pieces.append(0) #might not be right
      continue
    sub = []
    for i in range(len(C)):
      part = (s[i] / size)
      if(part != 0):
        sub.append(-1 * part * math.log(part, 2))
    pieces.append((sum(sub)) * (size / len(D)))
  return sum(pieces)


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


def createForest(attributelist, numTrees, numAttributes, numDatapoints,
                 threshold, classAttr, Ddataframe):
   forest = []
   attrNames = [att.attrName for att in attributelist]
   attrSet = set(attrNames)
   for n in range(numTrees):
      attSubset = rd.sample(attributelist, numAttributes)
      attRemove = (attrSet -  set([a.attrName for a in attSubset]))

      dataSubset = selectDataSubset(Ddataframe, attRemove,
                                    numAttributes, numDatapoints)
      #print(f"Att Subset: {attSubset}\ndataSubset:\n{dataSubset}")
      dtree = c45(dataSubset, attSubset, threshold, classAttr)

      forest.append(dtree)

   return forest


def selectDataSubset(dataset, attrRemove, numAtt, numDp):
   dataShuffled = dataset.sample(n = numDp)
   returnData = dataShuffled
   #attSubset = attrList - set(rd.sample(attrList, numAtt))
   for att in attrRemove:
      returnData = returnData.drop(columns = att)
   return returnData


def traverseTree(tree, row, classDomain):
   #if its a leaf
   if (len(tree.children) == 0):
      return tree.label
   if tree.isCont:
      # Assuming left child is <= and right child is >
      parsedEdge = tree.edgelabels[0].split(" ")
      #print(row)
      #print(f"Node label: {tree.label}")
      if float(row.loc[tree.label]) <= float(parsedEdge[1]):
         return traverseTree(tree.children[0], row, classDomain)
      else: # greater than split variable
         return traverseTree(tree.children[1], row, classDomain)
   else:
      for i in range(len(tree.children)):
         if(tree.edgelabels[i] == row.loc[tree.label]):
            return traverseTree(tree.children[i], row, classDomain)
  
   # No edge for the attribute. Choose random guess.
   return rd.choice(classDomain)
 

#################################################
# Main program
def main():
   args = sys.argv
   threshold = .1

   ### Print usage if args have wrong arity
   if len(args) != 3:
      print("Usage: python3 randomForest.py <datasetFile.csv> " +
            "<threshold>")
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
   dataset=[line.strip().split(',') for line in datasetfile if len(line)>0]
   

   ### Parse numeric input parameters
   """
   numAttributes = numDatapoints = numTrees = 0
   try:
      numAttributes = int(args[2])
      numDatapoints = int(args[3])
      numTrees = int(args[4])
      
      if numAttributes < -1 or numDatapoints < -1 or numTrees < -1:
         raise ValueError
   except:
      print("NumAttributes, NumDataPoints, and NumTrees must be integers" +
            " >0")
      exit()
   """
   try:
      threshold = float(args[2])

      if threshold >= 1 or threshold < 0:
         raise ValueError
   except:
      print("threshold must be 0 <= threshold < 1")
      exit()

   #print(Ddataframe)

   ### Get list of attributes and their domains
   attrNames = dataset[0]
   attrTypes = [int(n) for n in dataset[1]]
   classAttrName = dataset[2][0]

   classDomain = set([])
   attributelist = []
   for colnum in range(len(attrTypes)):
      attrType = attrTypes[colnum]
      if attrType == -1: # if -1, not to be counted in classification
         pass
      
      else:
         attrName = attrNames[colnum]
         if attrName == classAttrName: # Class attribute
            classDomain = set(Ddataframe[attrName].tolist())
            #for ind, dp in Ddataframe.iterrows():
            #   print("class Datapoint")
            #   print(dp)
            #   classDomain.add(dp[classAttrName])               
            
            #if len(classDomain) > attrType: # Domain exception
            #   print(f"Size of domain conflict for attribute: {attrName}")
            #   exit()

         else: # Attribute to be considered
            attrDomain = set([])
            isCont = attrType == 0
            if not isCont:
               attrDomain = set(Ddataframe[attrName].tolist())
               #for ind, dp in Ddataframe.iterrows():
               #   print("normal Datapoint")
               #   print(dp)
               #   attrDomain.add(dp[colnum])
            
            #if len(attrDomain) > attrType: # Domain exception
            #   print(f"Size of domain conflict for attribute: {attrName}")
            #   print(f"Extracted domain: {attrDomain}")
            #   exit()
            attributelist.append(Attribute(attrName, list(attrDomain), 
                                           colnum, isCont))

   classAttrIndex = attrNames.index(classAttrName)
   classAttr = Attribute(classAttrName, list(classDomain), classAttrIndex)

   """
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
   """

   ### Random Forest
   attrNames = [att.attrName for att in attributelist]
   
   #if numDatapoints > len(dataset):
      #print("Invalid argument for NumDataPoints. Cannot be larger than " +
      #      "size of dataset.")
      #exit()
   
   #if numAttributes > len(attrNames):
      #print("Invalid argument for NumAttributes. Cannot be larger than " +
      #      "number of attributes in dataset.")
      #exit()

   #selectDataSubset(dataset, attrList, classAttr, numAtt, numDp):
   #numDatapoints = min(numDatapoints, len(Ddataframe))
   #numAttributes = min(numAttributes, len(attrNames))
   
   #def createForest(attributelist, numTrees, numAttributes, numDatapoints,
   #              threshold, classAttr, Ddataframe):
   
   #forest = []
   #attrNames = [att.attrName for att in attributelist]
   #attrSet = set(attrNames)
   #for n in range(numTrees):
   #   attSubset = rd.sample(attributelist, numAttributes)
   #   attRemove = (attrSet -  set([a.attrName for a in attSubset]))

   #   dataSubset = selectDataSubset(Ddataframe, attRemove,
   #                                 numAttributes, numDatapoints)
      #print(f"Att Subset: {attSubset}\ndataSubset:\n{dataSubset}")
   #   dtree = c45(dataSubset, attSubset, threshold, classAttr)

   #   forest.append(dtree)

   

   #forest = createForest(attributelist, numTrees, numAttributes, 
   #                      numDatapoints, threshold, classAttr, Ddataframe)
   #for tr in forest:
   #   print("######## TREE##########")
   #   tr.printNode()

   ### Cross validation
   classDom = list(classDomain)

   """
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
          guess = traverseTree(dTree, row, classDom)

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
         guess = traverseTree(dTree, row, classDom)

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
   """
   
   if True: # 10-fold validation
      Dshuffled = Ddataframe.sample(frac = 1)
      folds = np.array_split(Dshuffled, 10)
      right = wrong = 0.0
      avgAcc = [0.0] * len(folds)
      avgErr = [0.0] * len(folds)
      confMatrix=[[0] * len(classDomain) for n in range(len(classDomain))]
      
      for f in range(len(folds)):
        testSet = folds[f]
        trainSets = folds[:]
        del trainSets[f]
        #trainSet = pd.concat(trainSets)

        #nDps = min(numDatapoints, len(trainSet))
        #print("Creating forest...")
        #forest = createForest(attributelist, numTrees, numAttributes, 
        #                      nDps, threshold, classAttr, trainSet)
        #dTree = c45(D, attributelist, threshold, classAttr)
        #print("Finished forest!")
        curRight = curWrong = 0.0
        for fold in trainSets:
          dTree = c45(fold, attributelist, threshold, classAttr)

          for test in range(len(testSet)):
             row = testSet.iloc[test]
           
             #for tr in forest:
             guess = traverseTree(dTree, row, classDom)
          
             x = classDom.index(guess)
             y = classDom.index(row[classAttrName])
             confMatrix[x][y] += 1
             if guess == row[classAttrName]:
                right += 1
                curRight += 1
             else:
                wrong += 1
                curWrong += 1
        
        #avgAcc[f] = curRight / (len(trainSets) * len(testSet))
        #avgErr[f] = curWrong / (len(trainSets) * len(testSet))

      print(f"Confusion matrix:\n{classDom}")
      for i in range(len(confMatrix)):
         print(f"{classDom[i]}: {confMatrix[i]}")
      
      print(f"Overall accuracy: {right / (right + wrong)}")
      #print(f"Average accuracy: {sum(avgAcc) / len(avgAcc)}")
      #print(f"Overall error rate: {wrong / (right + wrong)}")
      #print(f"Average error rate: {sum(avgErr) / len(avgErr)}")

     
main()
