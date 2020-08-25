#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:30:58 2019

@author: logistics
"""
#aac dpc pseaac am-pseacc pcp 
import re, os, sys
import math
import numpy as np
def read_protein_sequences(file):
    if os.path.exists(file) == False:
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]
    fasta_sequences = []
    for fasta in records:
        array = fasta.split('\n')
        header, sequence = array[0].split()[0], re.sub('[^ACDEFGHIKLMNPQRSTVWY-]', '-', ''.join(array[1:]).upper())
        header_array = header.split('|')
        name = header_array[0]
        fasta_sequences.append([name, sequence])
    return fasta_sequences
    
def DPC(fastas, gap, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = [] + diPeptides

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = []
        tmpCode = [0] * 400
        for j in range(len(sequence) - 2 + 1 - gap):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+gap+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+gap+1]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        encodings.append(code)
    return np.array(encodings, dtype=float), header

from sklearn.metrics import roc_auc_score
def cv(clf, X, y, nr_fold):
    ix = []
    for i in range(0, len(y)):
        ix.append(i)
    ix = np.array(ix)
    
    allACC = []
    allSENS = []
    allSPEC = []
    allMCC = []
    allAUC = []
    for j in range(0, nr_fold):
        train_ix = ((ix % nr_fold) != j)
        test_ix = ((ix % nr_fold) == j)
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        clf.fit(train_X, train_y)        
        p = clf.predict(test_X)
        pr = clf.predict_proba(test_X)[:,1]   
        TP=0   
        FP=0
        TN=0
        FN=0
        for i in range(0,len(test_y)):
            if test_y[i]==0 and p[i]==0:
                TP+= 1
            elif test_y[i]==0 and p[i]==1:
                FN+= 1
            elif test_y[i]==1 and p[i]==0:
                FP+= 1
            elif test_y[i]==1 and p[i]==1:
                TN+= 1
        ACC = (TP+TN)/(TP+FP+TN+FN)
        SENS = TP/(TP+FN)
        SPEC = TN/(TN+FP)
        det = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        if (det == 0):            
            MCC = 0                
        else:
            MCC = ((TP*TN)-(FP*FN))/det
        AUC = roc_auc_score(test_y,pr)
        allACC.append(ACC)
        allSENS.append(SENS)
        allSPEC.append(SPEC)
        allMCC.append(MCC)
        allAUC.append(AUC)
    return np.mean(allACC),np.mean(allSENS),np.mean(allSPEC),np.mean(allMCC),np.mean(allAUC)

def test(clf, X, y, Xt, yt):
    train_X, test_X = X, Xt
    train_y, test_y = y, yt
    #clf.fit(train_X, train_y)        
    p = clf.predict(test_X)
    pr = clf.predict_proba(test_X)[:,1]   
    TP=0   
    FP=0
    TN=0
    FN=0
    for i in range(0,len(test_y)):
        if test_y[i]==0 and p[i]==0:
            TP+= 1
        elif test_y[i]==0 and p[i]==1:
            FN+= 1
        elif test_y[i]==1 and p[i]==0:
            FP+= 1
        elif test_y[i]==1 and p[i]==1:
            TN+= 1
    ACC = (TP+TN)/(TP+FP+TN+FN)
    SENS = TP/(TP+FN)
    SPEC = TN/(TN+FP)
    det = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    if (det == 0):            
        MCC = 0                
    else:
        MCC = ((TP*TN)-(FP*FN))/det
    AUC = roc_auc_score(test_y,pr)

    return ACC, SENS, SPEC, MCC, AUC

# Load Data & Feature Extraction
fasta = read_protein_sequences('train-positive.txt')
train_pos, header = DPC(fasta,0)
fasta = read_protein_sequences('train-negative.txt')
train_neg, header = DPC(fasta,0)
fasta = read_protein_sequences('test-positive.txt')
test_pos, header = DPC(fasta,0)
fasta = read_protein_sequences('test-negative.txt')
test_neg, header = DPC(fasta,0)
alltrain_data = np.concatenate((train_pos, train_neg), axis=0)
alltest_data = np.concatenate((test_pos,test_neg), axis=0) 
alltrain_class = np.concatenate( (np.zeros(len(train_pos)),np.ones(len(train_neg))) , axis=0) 
alltest_class = np.concatenate( (np.zeros(len(test_pos)),np.ones(len(test_neg))) , axis=0) 
X = alltrain_data
y = alltrain_class
Xt = alltest_data
yt =  alltest_class

################################## Compare with 8 Classifier #####################################
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

########## Cross Validation ############################
file = open("8classifier_cv.csv", "w")
allclf = []
#SVM
parC = [2 ** i for i in np.arange(0,8, dtype=float)]
parG = [2 ** i for i in np.arange(-8,8, dtype=float)]
acc = np.zeros((len(parC),len(parG))) 
sens = np.zeros((len(parC),len(parG))) 
spec = np.zeros((len(parC),len(parG))) 
mcc = np.zeros((len(parC),len(parG))) 
roc = np.zeros((len(parC),len(parG))) 
for i in range(0,len(parC)):
    for j in range(0,len(parG)):
        clf = SVC(C=parC[i],gamma=parG[j], probability=True, random_state=0)
        acc[i,j], sens[i,j], spec[i,j], mcc[i,j], roc[i,j] = cv(clf, X,y,10)
ci, cj = np.unravel_index(acc.argmax(), acc.shape)  
allclf.append(SVC(C=parC[ci],gamma=parG[cj], probability=True, random_state=0).fit(X,y))
file.write(str(acc[ci,cj])+","+str(sens[ci,cj])+","+str(spec[ci,cj])+","+str(mcc[ci,cj])+","+str(roc[ci,cj])+","+str(parC[ci])+"-"+str(parG[cj])+"\n")        

#RF
param = [20, 50, 100, 200, 500]
acc = np.zeros(len(param)) 
sens = np.zeros(len(param)) 
spec = np.zeros(len(param)) 
mcc = np.zeros(len(param)) 
roc = np.zeros(len(param)) 
for i in range(0,len(param)):
    clf = RandomForestClassifier(n_estimators=param[i], random_state=0)
    acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
choose = np.argmax(acc)
allclf.append(RandomForestClassifier(n_estimators=param[choose], random_state=0).fit(X,y))
file.write(str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")        


#XGBoost
par1 = [20, 50,100,200,500]
par2 = [2, 4, 8]
par3 = [0.5,0.8, 1]
acc = np.zeros((len(par1),len(par2),len(par3))) 
sens = np.zeros((len(par1),len(par2),len(par3))) 
spec = np.zeros((len(par1),len(par2),len(par3))) 
mcc = np.zeros((len(par1),len(par2),len(par3))) 
roc = np.zeros((len(par1),len(par2),len(par3))) 
for i in range(0,len(par1)):
    for j in range(0,len(par2)):
        for k in range(0,len(par3)):
            clf = XGBClassifier(n_estimators=par1[i],max_depth=par2[j],colsample_bytree=par3[j] ,learning_rate=0.1, random_state=0)
            acc[i,j,k], sens[i,j,k], spec[i,j,k], mcc[i,j,k], roc[i,j,k] = cv(clf, X,y,10)
ci, cj, ck = np.unravel_index(acc.argmax(), acc.shape)  
allclf.append(XGBClassifier(n_estimators=par1[i],max_depth=par2[j],colsample_bytree=par3[j] ,learning_rate=0.1, random_state=0).fit(X,y))
file.write(str(acc[ci,cj,ck])+","+str(sens[ci,cj,ck])+","+str(spec[ci,cj,ck])+","+str(mcc[ci,cj,ck])+","+str(roc[ci,cj,ck])+","+str(par1[ci])+"-"+str(par2[cj])+"-"+str(par3[ck])+"\n")        


#MLP
param = [2 ** i for i in np.arange(1,9, dtype=int)]
acc = np.zeros(len(param)) 
sens = np.zeros(len(param)) 
spec = np.zeros(len(param)) 
mcc = np.zeros(len(param)) 
roc = np.zeros(len(param)) 
for i in range(0,len(param)):  
    clf = MLPClassifier(hidden_layer_sizes=(param[i],),random_state=0)
    acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
choose = np.argmax(acc)
allclf.append(MLPClassifier(hidden_layer_sizes=(param[choose],),random_state=0).fit(X,y))
file.write(str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")        

#NB
clf = GaussianNB()
acc, sens, spec, mcc, roc = cv(clf, X,y,10) 
allclf.append(clf)
file.write(str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n") 

#1NN
clf = KNeighborsClassifier(n_neighbors=1)
acc, sens, spec, mcc, roc = cv(clf, X,y,10) 
allclf.append(clf)
file.write(str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n") 

#DT
clf = DecisionTreeClassifier(random_state=0)
acc, sens, spec, mcc, roc = cv(clf, X,y,10) 
allclf.append(clf)
file.write(str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n") 

#Logistic
param = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
acc = np.zeros(len(param)) 
sens = np.zeros(len(param)) 
spec = np.zeros(len(param)) 
mcc = np.zeros(len(param)) 
roc = np.zeros(len(param)) 
for i in range(0,len(param)):
    clf = LogisticRegression(C=param[i], random_state=0)
    acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
choose = np.argmax(acc)
allclf.append(LogisticRegression(C=param[choose], random_state=0).fit(X,y))
file.write(str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")        

file.close()

########## Test ############################
file = open("8classifier_test.csv", "w")
for i in range(0,len(allclf)):
    acc, sens, spec, mcc, roc = test(allclf[i], X, y, Xt, yt) 
    file.write(str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+"\n") 
file.close()