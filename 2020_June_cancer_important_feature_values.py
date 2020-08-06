# Uses accuracy of a ML classifier as a distance to plot pairwise distances of cancers
import itertools
import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,roc_curve,roc_auc_score,auc
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
plt.switch_backend('agg') 
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os,csv
import random
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import json
import sys
from sklearn.feature_selection import SelectFromModel
import math
chromosomes = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22"]
colors = ['turquoise', 'red', 'green', 'blue', 'cyan','yellow','orange','brown','magenta','purple','maroon']
df = pd.read_csv('June_2020_Important_Features_all.csv')
data = df.to_numpy()
features = set()
for d in data:
    for i in range(1,len(d)):
        if math.isnan(d[i]) == False:
            features.add(int(d[i]))
feature_indices = list(features)
ACss = {}
ACacc = {}
plt.rcParams.update({'font.size': 14})
def combineChrom(dict1, dict2):
    result = dict()
    for k in dict1:
        if k in dict2:
            result[k] = dict1[k] + dict2[k]
    return result

class Cancer:
    def __init__(self, folder, cancer_name, cancerType, chroms):
        self.cancerType = cancerType
        self.folders = ["{}/chr{}{}".format(folder, chrom, cancer_name) for chrom in chroms]
        self.chromosomes = [chrom for chrom in chroms]
        self.X, self.y, self.fn = self.getXy()

    def getData(self, value):
        y = [value for i in self.y]
        return self.X, y, self.fn

    def getXy(self, value=0):
        X = list()
        y = list()
        first = True
        cumulativeVects = dict()
        ccttrr = 0
        ssuumm = 0
        for folder in self.folders:
            newVect = dict()
            chrom = self.chromosomes[ccttrr]
            index_vec = []
            g = open('feature_index_'+chrom+'_alt.txt','r')
            for line in g:
                index_vec.append(int(line))
            g.close()
            ssuumm+=2*len(index_vec)
            ccttrr+=1
            for filename in os.listdir(os.getcwd() + "/" + folder):
                if not filename.endswith("txt"):
                    continue
                f = open(folder + "/" + filename, 'r')
                newVect[filename] = list()
                
                # Construct Vector
                
                counter = 0
                cctr = 0
                for line in f.readlines():
                    if len(line) < 2:
                        continue
                    midi = line.split(" ")
                    mi = int(midi[0].strip())
                    di = int(midi[1].strip())
                    if counter < len(index_vec) and index_vec[counter] == cctr:
                        newVect[filename].append(mi)
                        newVect[filename].append(di)
                        counter+=1
                    cctr+=1
                f.close()
            print("{} has {} files".format(folder, len(newVect.keys())))
            if first:
                cumulativeVects = newVect
                first = False
            else:
                cumulativeVects = combineChrom(newVect, cumulativeVects)
        X = list()
        fn = list()
        y = list()
        for filename in cumulativeVects:
            fn.append(filename.split(".")[0].split("_")[-1])
            X.append(cumulativeVects[filename])
            y.append(value)
        self.data_points = len(y)
        print("{} data points of value {} in cancer {}".format(len(y), value, self.cancerType))
        print("Total features:"+str(ssuumm))
        return X, y, fn



def possiblePermutations(establishedOrder, newCancers):
    if not newCancers:
        return [establishedOrder]
    result = []
    for pperm in possiblePermutations(establishedOrder, newCancers[1:]):
        for splitIndex in range(len(pperm) + 1):
            result.append(pperm[:splitIndex] + [newCancers[0]] + pperm[splitIndex:])
    return result
        

def pairwise_plot(oldCancers, newCancers, distanceFunction, distanceSTD, title, file):
    # Solve the Traveling Salesman Problem
    minDist = float('inf')
    best = None
    print("\n\nCALCULATING TSP")
    for cancerList in possiblePermutations(oldCancers, newCancers):
        cancerList = [c.cancerType for c in cancerList]
        distances = [distanceFunction[c1, c2] for c1, c2 in zip(cancerList, cancerList[1:])]
        dist = sum(distances)
        #print("{}: {}".format(cancerList, dist))
        if dist < minDist:
           best = cancerList
           minDist = dist
    
    # Now use best to make the plot
    distanceMatrix = np.array([[distanceFunction[c1, c2] for c1 in best] for c2 in best])
    #print(distanceMatrix)
    df = pd.DataFrame(distanceMatrix, columns=best)
    print(df)
    ax = plt.axes()
    sns.heatmap(df, annot=True, annot_kws={"size": 10}, yticklabels=best,cmap = sns.cm.rocket_r)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig("{}.png".format(file))
    plt.close()

def pairs(l):
    result = []
    for i in range(len(l)):
        for j in range(len(l)):
            result.append((l[i], l[j]))
    return result

GBM = Cancer("brain_D","brain","GBM",chromosomes)
#THCA = Cancer("thyroid_D","thyroid","THCA",chromosomes)
#LUAD = Cancer("lung_D","lung","LUAD",chromosomes)
#LUSC = Cancer("lung_sq_D","lung_sq","LUSC",chromosomes)
#SKCM = Cancer("skin_D","skin","SKCM",chromosomes)
#PAAD = Cancer("pancreas_D","pancreas","PAAD",chromosomes)
#HNSC = Cancer("head_neck_D","head_neck","HNSC",chromosomes)
#PRAD = Cancer("prostate_m/new_fo","prostate","PRAD",chromosomes)
#LGG = Cancer("brain_lgg_D","brain_lgg","LGG",chromosomes)
#BLCA = Cancer("bladder_D","bladder","BLCA",chromosomes)
#STAD = Cancer("stomach_D","stomach","STAD",chromosomes)
#cancers = [GBM,THCA,LUAD,LUSC,SKCM,PAAD,HNSC,PRAD,LGG,BLCA,STAD]
#cancers = [GBM,THCA,LUAD]
cancers = [GBM]
lf = len(feature_indices)
f = open('2020_June_feature_values.csv','w')
wr = csv.writer(f, quoting=csv.QUOTE_ALL)
wr.writerow(feature_indices+['cancer'])
for i,cancer in enumerate(cancers):
    X, _, _ = cancers[i].getData(i)
    X = np.array(X)
    values = np.zeros((X.shape[0],lf))
    for j,v in enumerate(X):
        v = np.array(v)
        values[j] = v[feature_indices]
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(list(values[j])+[cancer.cancerType])
f.close()
        












    
