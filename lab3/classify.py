# -*- coding: utf-8 -*-
import json
import pandas as pd
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

def dicToTree(js):
  #print("JS: ", js)
  node = Node(js['var'])
  for e in js['edges']:
    node.edgelabels.append(e['value'])
    for k in e.keys():
      if(k == "node"):
        node.children.append(dicToTree(e['node']))
      elif(k == "leaf"):
        leaf = e['leaf']
        node.children.append(Node(leaf['decision']))
  return node

def traverseTree(tree, row):
  #if its a leaf
  if (len(tree.children) == 0):
    return tree.label
  for i in range(len(tree.children)):
    if(tree.edgelabels[i] == row.loc[tree.label]):
      return traverseTree(tree.children[i], row)

def main():
  args = sys.argv

  ### Print usage if args have wrong arity
  if len(args) != 3:
      print("Usage: python3 <programName> <CSVFile> <JSONFile>")
      exit()
  #print(args[2])
  with open(args[2], 'r') as json_file:
    js = json.loads(json_file.read())
    js = js['node']

  D = pd.read_csv(args[1])

  #name of Y feature
  result = D.loc[1, D.columns[0]]
  #cut out formatting stuff from data
  arity = D.loc[0]
  for i in range(len(arity)):
    if(int(arity[i]) < 0):
      del D[D.columns[i]]
      i -= 1
  D = D.loc[2: ]
  #list of every possible Y
  C = D.loc[:, result]
  C = list(dict.fromkeys(D.loc[:, result]))
  #list of the names of every feature
  A = D.columns

  tree = dicToTree(js)
  #number of right guesses
  right = 0.0
  #number of wrong guesses
  wrong = 0.0
  #confusion matrix
  mat = []
  for i in range(len(C)):
    mat.append([])
    for c in range(len(C)):
      mat[i].append(0)
  #having the tree guess the result, then adding guess to the confusion matrix, and finally incrementing whether the guess was right or wrong
  for d in range(len(D)):
    row = D.loc[d + 2]
    guess = traverseTree(tree, row)
    x = C.index(guess)
    y = C.index(row[result])
    mat[x][y] += 1
    if(guess == row[result]):
      right += 1
    else:
      wrong += 1

  print("Total number of items classified: ", len(D))
  print("Total number correctly classified: ", right)
  print("Total number incorrectly classified: ", wrong)
  print("Overall accuracy: ", right / len(D))
  print("Error rate: ", wrong / len(D))
  print(C)
  for m in mat:
    print(m)

main()
