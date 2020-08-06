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
chromosomes = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22"]
#chromosomes = ["22"]
colors = ['turquoise', 'red', 'green', 'blue', 'cyan','yellow','orange','brown','magenta','purple','maroon']
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
LUAD = Cancer("lung_D","lung","LUAD",chromosomes)
LUSC = Cancer("lung_sq_D","lung_sq","LUSC",chromosomes)
SKCM = Cancer("skin_D","skin","SKCM",chromosomes)
PAAD = Cancer("pancreas_D","pancreas","PAAD",chromosomes)
HNSC = Cancer("head_neck_D","head_neck","HNSC",chromosomes)
PRAD = Cancer("prostate_m/new_fo","prostate","PRAD",chromosomes)
LGG = Cancer("brain_lgg_D","brain_lgg","LGG",chromosomes)
BLCA = Cancer("bladder_D","bladder","BLCA",chromosomes)
STAD = Cancer("stomach_D","stomach","STAD",chromosomes)
cancers = [GBM,THCA,LUAD,LUSC,SKCM,PAAD,HNSC,PRAD,LGG,BLCA,STAD]
#cancers = [GBM,THCA,LUAD]
newCancers = cancers
cancerAccuracies = ACacc
cancerSense = ACss

accuracy_std = dict()
sense_std = dict()

# Iterate through all pairs of cancers and determine
model_top_coefs = dict()
#f = open('Important_Features_all.csv','w')
cancernames = [cancer.cancerType for cancer in cancers]
X = [[] for _ in cancers]
y = [[] for _ in cancers]
fn = [[] for _ in cancers]
val_score = [[] for _ in cancers]
trim_length = float('inf')
for i,cancer in enumerate(cancers):
    X[i], y[i], fn[i] = cancers[i].getData(i)
    if trim_length > len(y[i]):
        trim_length = len(y[i])
tprs = []
aucs = []
kf = KFold(n_splits = 4)
fig, ax = plt.subplots()
mean_fpr = np.linspace(0, 1, 100)
testAccuracies = []
testSense = []
testSpec = []
imp_feat_set = set()
f = open('2020_June_Important_Features_all_multi.csv','w')
# Trim so we have equal data points for each
for _ in range(5):
    W = []
    v = []
    filenames = []
    for i,_ in enumerate(cancers):
        U = range(len(y[i]))
        U = np.random.permutation(U)
        X[i] = np.array(X[i])
        y[i] = np.array(y[i])
        fn[i] = np.array(fn[i])
        X[i] = X[i][U[:trim_length]]
        y[i] = y[i][U[:trim_length]]
        fn[i] = fn[i][U[:trim_length]]
        X[i] = list(X[i])
        y[i] = list(y[i])
        fn[i] = list(fn[i])
        # Join data
        W+=X[i]
        v+=y[i]
        filenames+=fn[i]

    
        
    # Shuffle the data
    zipped = list(zip(W, v, filenames))
    random.shuffle(zipped)
    W = [x for x, _, _ in zipped]
    v = [yi for _, yi, _ in zipped]
    filenames = [fn for _, _, fn in zipped]
    W = np.array(W)
    W = preprocessing.scale(W)
    v = np.array(v)
    filenames = np.array(filenames)
    
    #f.write(cancer0.cancerType+' vs '+cancer1.cancerType+'\n')
    for train, test in kf.split(W):
        X_train = W[train]
        y_train = v[train]
        fn_train = filenames[train]
        X_test = W[test]
        y_test = v[test]
        fn_test = filenames[test]
         
    ## Edit model here
        model = XGBClassifier(max_depth=2,n_jobs=8,objective = 'multi:softprob',num_classes = len(cancers))
    #model = RandomForestClassifier(max_depth=2)
        model.fit(X_train, y_train)
        imp_feat = model.feature_importances_
        imp_indices = np.argsort(imp_feat)
        imp_feat.sort()
        thresh = imp_feat[-30]
        for bfs in range(-30,0,1):
            imp_feat_set.add(imp_indices[bfs])
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
        preds_prob = selection_model.predict_proba(select_X_test)
        counter = 0
        for yy in y_test:
            val_score[yy].append(preds_prob[counter])
            counter+=1
        '''fpr, tpr, thresholds = roc_curve(y_test, preds,pos_label = 0)
        auc = round(roc_auc_score(y_test,preds),3)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc)'''
        accuracy = accuracy_score(y_test, preds)
        testAccuracies.append(accuracy)
        

        print("Test Accuracy: %.2f%%" % (accuracy * 100.0))
        print("\n")
'''print(len(tprs))
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
plt.savefig('ROC_Curve_XGB_multi_5.png')
plt.close()'''
print("Average Test Accuracies: {} +- {}".format(np.mean(testAccuracies), np.std(testAccuracies)))
wr = csv.writer(f, quoting=csv.QUOTE_ALL)
wr.writerow(list(imp_feat_set))
g = open('2020_June_multi_probabilities.csv','w')
wrr = csv.writer(g, quoting=csv.QUOTE_ALL)
wrr.writerow(list())
for i in range(len(cancers)):
    probs_mean = np.mean(val_score[i],axis = 0)
    probs_std = np.std(val_score[i],axis = 0)
    wrr.writerow([cancernames[i]]+list(probs_mean))
    wrr.writerow([cancernames[i]]+list(probs_std))
    plt.bar(cancernames,probs_mean,yerr=probs_std, color = colors[:len(cancers)],align='center', capsize=10)
    plt.ylabel('Probability')
    plt.xticks(rotation = 90)
    plt.title('{} classification probability'.format(cancernames[i]))
    plt.tight_layout()
    plt.savefig('{}_multiclass.png'.format(cancernames[i]),bbox_inches = 'tight')
    plt.close()   
g.close()  
f.close()


    
