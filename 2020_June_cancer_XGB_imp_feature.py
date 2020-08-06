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
#chromosomes = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22"]
chromosomes = ["22"]
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
#Example with only 2 cancer types GBM and THCA, you may add #more cancer types in the similar fomat and change the cancers #and newCancers variable accordingly
GBM = Cancer("brain_D","brain","GBM",chromosomes)
THCA = Cancer("thyroid_D","thyroid","THCA",chromosomes)
#LUAD = Cancer("lung_D","lung","LUAD",chromosomes)
#LUSC = Cancer("lung_sq_D","lung_sq","LUSC",chromosomes)
#SKCM = Cancer("skin_D","skin","SKCM",chromosomes)
#PAAD = Cancer("pancreas_D","pancreas","PAAD",chromosomes)
#HNSC = Cancer("head_neck_D","head_neck","HNSC",chromosomes)
#PRAD = Cancer("prostate_m/new_fo","prostate","PRAD",chromosomes)
#LGG = Cancer("brain_lgg_D","brain_lgg","LGG",chromosomes)
#BLCA = Cancer("bladder_D","bladder","BLCA",chromosomes)
#STAD = Cancer("stomach_D","stomach","STAD",chromosomes)
#BRCA = Cancer("breast","breast","BRCA",chromosomes)
#cancers = [GBM,THCA,LUAD,LUSC,SKCM,PAAD,HNSC,PRAD,LGG,BLCA,STAD]
cancers = [GBM,THCA]
newCancers = cancers
cancerAccuracies = ACacc
cancerSense = ACss

accuracy_std = dict()
sense_std = dict()
best_feature_set = dict()
# Iterate through all pairs of cancers and determine
model_top_coefs = dict()
#f = open('Important_Features_all.csv','w')
for cancer0, cancer1 in pairs(cancers):
    print("\n\nPERFORMING TEST ON CANCERS {}(0) and {}(1)".format(cancer0.cancerType, cancer1.cancerType))
    
    #if (cancer0.cancerType, cancer1.cancerType) in ACacc:
    #    print("Already Computed")
    #    continue
    X0, y0, fn0 = cancer0.getData(0)
    X1, y1, fn1 = cancer1.getData(1)

    # If we have the same cancerTypes - dont double the data
    if (cancer1.cancerType == cancer0.cancerType):
        cutoff = int(len(y0)/2)
        X1 = X0[cutoff:]
        X0 = X0[:cutoff]
        y1 = [1 for i in range(cutoff)]
        y0 = y0[:cutoff]
        fn1 = fn0[cutoff:]
        fn0 = fn0[:cutoff]
    tprs = []
    aucs = []
    kf = KFold(n_splits = 4)
    fig, ax = plt.subplots()
    mean_fpr = np.linspace(0, 1, 100)
    testAccuracies = []
    testSense = []
    testSpec = []
    best_feat_set[cancer0.cancerType,cancer1.cancerType] = set()
    # Trim so we have equal data points for each
    for _ in range(5):
        trim_length = min(len(y0),len(y1))
        U0 = range(len(y0))
        U1 = range(len(y1))
        U0 = np.random.permutation(U0)
        U1 = np.random.permutation(U1)
        X0 = np.array(X0)
        y0 = np.array(y0)
        fn0 = np.array(fn0)
        X1 = np.array(X1)
        y1 = np.array(y1)
        fn1 = np.array(fn1)
        X0 = X0[U0[:trim_length]]
        y0 = y0[U0[:trim_length]]
        fn0 = fn0[U0[:trim_length]]
        X1 = X1[U1[:trim_length]]
        y1 = y1[U1[:trim_length]]
        fn1 = fn1[U1[:trim_length]]
        X0 = list(X0)
        X1 = list(X1)
        y0 = list(y0)
        y1 = list(y1)
        fn0 = list(fn0)
        fn1 = list(fn1)
        print(len(y0))
    # Join data
        X = X0 + X1
        y = y0 + y1
        filenames = fn0 + fn1
        
    # Shuffle the data
        zipped = list(zip(X, y, filenames))
        random.shuffle(zipped)
        X = [x for x, _, _ in zipped]
        y = [yi for _, yi, _ in zipped]
        filenames = [fn for _, _, fn in zipped]
        X = np.array(X)
        X = preprocessing.scale(X)
        y = np.array(y)
        filenames = np.array(filenames)
    
    #f.write(cancer0.cancerType+' vs '+cancer1.cancerType+'\n')
        for train, test in kf.split(X):
            X_train = X[train]
            y_train = y[train]
            fn_train = filenames[train]
            X_test = X[test]
            y_test = y[test]
            fn_test = filenames[test]
         
        ## Edit model here
            model = XGBClassifier(max_depth=2,n_jobs=8)
        #model = RandomForestClassifier(max_depth=2)
            model.fit(X_train, y_train)
            imp_feat = model.feature_importances_
            imp_indices = np.argsort(imp_feat)
            imp_feat.sort()
            print(imp_feat)
            thresh = imp_feat[-30]
            for bfs in range(-30,1,0):
                best_feature_set[cancer0.cancerType,cancer1.cancerType].add(imp_indices[bfs])
            selection = SelectFromModel(model,threshold = thresh, prefit = True)
            select_X_train = selection.transform(X_train)
            selection_model = XGBClassifier(max_depth = 1)
            selection_model.fit(select_X_train,y_train)
       # wr = csv.writer(f, quoting=csv.QUOTE_ALL)
       # wr.writerow(model.feature_importances_)                
        # Training
            preds = selection_model.predict(select_X_train)
            accuracy = accuracy_score(y_train, preds)
            print("Training Accuracy: %.2f%%" % (accuracy * 100.0))
        
        # Testing
            select_X_test = selection.transform(X_test)
            preds = selection_model.predict(select_X_test)
            fpr, tpr, thresholds = roc_curve(y_test, preds)
            auc = round(roc_auc_score(y_test,preds),3)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc)
            accuracy = accuracy_score(y_test, preds)
            testAccuracies.append(accuracy)
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            sensitivity = tp/(tp + fn)
            specificity = tn/(fp + tn)
            testSense.append(sensitivity)
            testSpec.append(specificity)

            print("Test Accuracy: %.2f%%" % (accuracy * 100.0))
            print("\n")
    print(len(tprs))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label='$\pm$  std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],xlabel = 'False Positive Rate',ylabel = 'True Positive Rate',title="Receiver operating characteristic")
    ax.legend(loc="lower right")

    
    plt.tight_layout()
    plt.savefig('ROC_Curve_{}_{}_XGB_5.png'.format(cancer0.cancerType,cancer1.cancerType))
    plt.close()
    print("Average Test Accuracies: {} +- {}".format(np.mean(testAccuracies), np.std(testAccuracies)))
    print(best_feature_set[cancer0.cancerType,cancer1.cancerType])
    cancerAccuracies[cancer0.cancerType, cancer1.cancerType] = np.mean(testAccuracies)
    cancerSense[cancer0.cancerType, cancer1.cancerType] = np.mean(testSense) #first one is the 0 

    accuracy_std[cancer0.cancerType, cancer1.cancerType] = np.std(testAccuracies)
    sense_std[cancer0.cancerType, cancer1.cancerType] = np.std(testSense)
        
    cancerAccuracies[cancer1.cancerType, cancer0.cancerType] = np.mean(testAccuracies)
    cancerSense[cancer1.cancerType, cancer0.cancerType] = np.mean(testSpec) #first one is the 0 (flipping to change sense to spec)


#f.close()
    
cancerNames = [c.cancerType for c in cancers]
print("\n\nFINAL MATRICES")
print("#Accuracies")
print("ACacc = {}".format(cancerAccuracies))
print("#Specificity/Sensitivity")
print("ACss = {}".format(cancerSense))
accOrder = []
ssOrder = []
#pairwise_plot(accOrder, newCancers, cancerAccuracies, accuracy_std, "Binary Classifier Accuracies", "accuracies_June_2020")
#pairwise_plot(ssOrder, newCancers, cancerSense, sense_std, "Binary Classifier Sensitivity/Specificity", "sense_spec_June_2020")
