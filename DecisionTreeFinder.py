import numpy as np
from sklearn import tree
import graphviz
import Measures
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from matplotlib import pyplot as plt

nltk.download("stopwords")

STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()

PUNCT_REG = "[.\"\\-,*)(!?#&%$@;:_~\^+=/]"
import re



class dataPoint:

    def __init__(self,sentences,label):
        self.orig_sent1,self.orig_sent2 = sentences
        self.sent1 = re.sub(PUNCT_REG,"",self.orig_sent1)
        self.sent2 = re.sub(PUNCT_REG,"",self.orig_sent2)
        self.label = label

    def get_original_sentences(self):
        return self.orig_sent1, self.orig_sent2

    def get_features(self):
        self.features = []
        s1 = self.sent1.split()
        s2 = self.sent2.split()
        l1 = [l for l in s1 if l not in STOPWORDS]
        l2 = [l for l in s2 if l not in STOPWORDS]
        l2 = [ps.stem(l) for l in l2]
        l1 = [ps.stem(l) for l in l1]
        edit_dist,_,_,_ = Measures.fuzzy_substring_dist(l2,l1)
        full_edit_dist,_,_,_ = Measures.fuzzy_substring_dist(s2,s1)
        self.features.append(edit_dist)
        self.features.append(full_edit_dist)
        self.features.append(min(len(s1),len(s2)))
        self.features.append(min(len(l1),len(l2)))
        #Think of more features

        return self.features

    def get_label(self):
        return self.label




def read_data_file(fname):
    with open(fname,encoding="utf-8") as f:
        line = f.readline()
        X_sentences = []
        while (line):
            cur = line.split(",")
            if (len(cur) == 3):
                label = int(cur[2])
                X_sentences.append(dataPoint((cur[0],cur[1]),label))

            line = f.readline()
    return X_sentences



def get_train_test(data_points,perc):
    import random
    shuffled_points = data_points[:]
    slice = int(len(shuffled_points) *perc)
    random.shuffle(shuffled_points)
    train_points = shuffled_points[:slice]
    test_points = shuffled_points[slice:]
    train_x = [dp.get_features() for dp in train_points]
    test_x = [dp.get_features() for dp in test_points]
    train_y = [dp.get_label() for dp in train_points]
    test_y = [dp.get_label() for dp in test_points]
    return train_x,train_y,test_x,test_y


def get_accuracy(y,yhat):
    accuracy = 1 - len(np.argwhere(y != yhat)) / len(y)
    return accuracy


def get_best_decision_tree(fname,figname,max_depth):
    data_points= read_data_file(fname)
    perc = 0.8
    train_x,train_y,test_x,test_y = get_train_test(data_points,perc)
    print(len(test_y))
    print(len(test_x))
    print(len(data_points))

    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf = clf.fit(train_x,train_y)
    tree.plot_tree(clf, max_depth = 3)
    pred_y = clf.predict(test_x)
    print(pred_y)
    print(test_y)
    print(get_accuracy(pred_y,test_y))
    plt.savefig(figname)
    return clf

import joblib
import os
import pickle
# os.environ["PATH"] += os.pathsep + 'C:\\Users\\sweed\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\graphviz\\__pycache__\\dot.exe'
import sklearn

if __name__ =="__main__":
    fname = "paraphrase_data"
    figname = "decision_tree_fig.png"
    tree_pickle_fname = "paraphrase_dt"
    clf = get_best_decision_tree(fname, figname,3)
    joblib.dump(clf,tree_pickle_fname)
# plt.savefig("nirtest2.jpg")

