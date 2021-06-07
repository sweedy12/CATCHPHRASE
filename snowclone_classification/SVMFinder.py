import numpy as np
from sklearn import tree
import graphviz
import Measures
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from TFIDF import TFIDF
import Seq2SeqUtility as s2s
from S2SBERT import prepare_examples_for_torch
TFIDF_NAME = "new_tokenized_tfidf100"
from matplotlib import pyplot as plt

nltk.download("stopwords")

STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()

PUNCT_REG = "[.\"\\-,*)(!?#&%$@;:_~\^+=/]"
PUNCT_REG_WILDCARD = "[.\"\\-,)(!?#&%$@;:_~\^+=/]"
import re
import torch.nn as nn
EMB_SIZE = 704
from torchcrf import CRF

class LSTMcrf(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):

        super().__init__()
        self.crf_layer = CRF(2)
        self.LSTM = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True,num_layers=n_layers)
        self.hidden2ta = nn.Linear(2*hidden_dim,2)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, seq_emb,tags,masks):
        tags = tags.permute(1,0,2)
        tags = tags.squeeze()
        tags = torch.tensor(tags,dtype=torch.long)
        masks = masks.squeeze()
        masks = torch.tensor(masks,dtype=torch.uint8)
        seq_emb = seq_emb.permute(1,0,2)
        hidden,_ = self.LSTM(seq_emb)
        # hidden_concatanated = torch.cat((h[:,0,:,:],h_n[:,1,:,:]))
        emission = self.softmax(self.hidden2ta(hidden))
        likelihood  = self.crf_layer(emission,tags,masks)
        # hidden1_full = self.dropout(packed_output1)
        return -likelihood

    def predict(self, seq_emb):
        seq_emb = seq_emb.permute(1, 0, 2)
        hidden, _ = self.LSTM(seq_emb)
        # hidden_concatanated = torch.cat((h[:,0,:,:],h_n[:,1,:,:]))
        emission = self.softmax(self.hidden2ta(hidden))
        return torch.tensor(self.crf_layer.decode(emission))

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

MODEL_PATH = "nirnirnir"
class ParaphraseData:

    def __init__(self,fname, normalized = False, wildcarded = False,model=None, model_function = None,model_params = None):
        self.orig_sent_to_para = {}
        self.orig_sent_to_data = {}
        self.positive_data = []
        self.negative_data = []
        if (wildcarded):
            self.model = model
            self.model_function = model_function
            self.model_params = model_params

        self.get_all_data(fname, normalized,wildcarded)


    def clean_sentence(self,sent):
        sent = re.sub(PUNCT_REG," ",sent)
        sent = re.sub("\\s\\s"," ",sent)
        return sent

    def get_all_data(self, fname, normalized,wildcarded):
        with open(fname, "rb") as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                try:
                    line = line.decode("utf-8")
                except:
                    line = line.decode("cp1252")
                line_data = line.split(",")
                orig_sent = line_data[0]
                try:
                    cur_sent = self.clean_sentence(line_data[1])
                except:
                    stop = 1
                if (cur_sent == " d keep your friends close your enemies closer "):
                    stop = 1
                try:
                    label = int(line_data[2])
                except:
                    stop = 1
                if orig_sent not in self.orig_sent_to_para:
                    self.orig_sent_to_para[orig_sent] = []
                    self.orig_sent_to_data[orig_sent] = {}
                    self.orig_sent_to_data[orig_sent][0] = []
                    self.orig_sent_to_data[orig_sent][1] = []
                if (cur_sent not in self.orig_sent_to_para[orig_sent]):
                    self.orig_sent_to_para[orig_sent].append(cur_sent)
                    if (wildcarded): #getting the wildcarded version of the datapoint
                        cur_example = s2s.PhraseExample(orig_sent,[])
                        predicted_sent = self.model_function(self.model,cur_example,self.model_params)
                        if (cur_sent == "why won't he be with you if you do this "):
                            stop = 1
                        dp = SCdataPointCat((" ".join(predicted_sent[1:-1]),cur_sent,orig_sent),label,normalized=normalized)
                    else:
                        dp = dataPoint((orig_sent,cur_sent),label, normalized=normalized)
                    self.orig_sent_to_data[orig_sent][label].append(dp)
                    if (label == 0):
                        self.negative_data.append(dp)
                    else:
                        self.positive_data.append(dp)

    def get_overall_balanced_data(self):
        return self.balance_lists(self.positive_data,self.negative_data)

    def balance_by_sent(self):
        with open("para_data\\all_info","w") as f:
            import random
            pos_list = []
            neg_list = []
            length_by_sent = []
            sentences = list(self.orig_sent_to_data.keys())
            random.shuffle(sentences)
            for i,orig_sent in enumerate(sentences):
                with open("para_data\\" + orig_sent, "w", encoding="utf-8") as f2:
                    f.write(orig_sent + "\n")
                    l1 = self.orig_sent_to_data[orig_sent][0][:]
                    l2 = self.orig_sent_to_data[orig_sent][1][:]
                    f.write("neg: " + str(len(l1)) + " , ")
                    f.write("pos: " + str(len(l2)) + "\n")
                    l1,l2 = self.balance_lists(l1,l2)
                    for ind in range(len(l1)):
                        l1_sent = l1[ind].get_original_sentences()
                        l2_sent = l2[ind].get_original_sentences()
                        f2.write(l1_sent[0]+"," + l1_sent[1] + ",0" + "\n")
                        f2.write(l2_sent[0]+"," + l2_sent[1] + ",1" + "\n")
                    # print(len(l1))
                    # print(len(l2))
                    pos_list.extend(l2)
                    neg_list.extend(l1)
                    if (i == 0):
                        length_by_sent.append(len(l1))
                    else:
                        length_by_sent.append(length_by_sent[i-1]+len(l1))

        return pos_list,neg_list,length_by_sent


    def get_closest_train_val_test_split(self,train_size,valid_size,pos_list,neg_list,length_by_sent):
        train_length = train_size*len(pos_list)
        valid__length = valid_size*len(pos_list)
        last_j = length_by_sent[0]
        train_j = 0
        pos_train = []
        pos_test = []
        neg_train = []
        neg_test = []
        pos_valid = []
        neg_valid = []
        for j in length_by_sent[1:]:
            if ((last_j < train_length or last_j == length_by_sent[0]) and j >= train_length):
                train_j = j
                pos_train = pos_list[:j]
                neg_train = neg_list[:j]
            elif (last_j < train_length+valid__length and j>= train_length+valid__length):
                pos_valid = pos_list[train_j:j]
                neg_valid = neg_list[train_j:j]
                pos_test = pos_list[j:]
                neg_test = neg_list[j:]
                break
            last_j = j
        print("-------------")
        train_set = pos_train + neg_train
        valid_set = pos_valid + neg_valid
        test_set = pos_test + neg_test
        return train_set, valid_set, test_set


    def get_balanced_data_by_sent(self,train_size,valid_size):
        pos_list,neg_list,length_by_sent = self.balance_by_sent()
        return self.get_closest_train_val_test_split(train_size,valid_size,pos_list,neg_list,length_by_sent)

    def unbalance_data_help(self):
        import random
        length_by_sent = []
        all_list = []
        sentences = list(self.orig_sent_to_data.keys())
        random.shuffle(sentences)
        for i,orig_sent in enumerate(sentences):
            with open("para_data\\unbalanced_" + orig_sent, "w", encoding="utf-8") as f2:
                l1 = self.orig_sent_to_data[orig_sent][0][:]
                l2 = self.orig_sent_to_data[orig_sent][1][:]
                for ind in range(len(l1)):
                    l1_sent = l1[ind].get_original_sentences()
                    f2.write(l1_sent[0]+"," + l1_sent[1] + ",0" + "\n")
                for ind in range(len(l2)):
                    l2_sent = l2[ind].get_original_sentences()
                    f2.write(l2_sent[0]+"," + l2_sent[1] + ",1" + "\n")
                print(len(l1))
                print(len(l2))
                all_list.extend(l1)
                all_list.extend(l2)
                if (i == 0):
                    length_by_sent.append(len(l1)+len(l2))
                else:
                    length_by_sent.append(length_by_sent[i-1]+len(l1)+len(l2))
        return all_list, length_by_sent


    def get_unbalanced_data(self,train_size,valid_size):
        all_list, length_by_sent = self.unbalance_data_help()
        all_train = []
        all_val = []
        all_test  = []
        train_length = train_size*len(all_list)
        valid__length = valid_size*len(all_list)
        last_j = length_by_sent[0]
        train_j = 0
        for j in length_by_sent[1:]:
            if ((last_j < train_length or last_j == length_by_sent[0]) and j >= train_length):
                train_j = j
                all_train= all_list[:j]
            elif (last_j < train_length+valid__length and j>= train_length+valid__length):
                all_val = all_list[train_j:j]
                all_test = all_list[j:]
                break
            last_j = j
        return all_train, all_val,all_test

    def get_triplet_data(self,train_size, valid_size):
        all_train = []
        all_val = []
        all_test = []
        all_list = []
        #getting the triplet data
        length_by_sent = []
        for i,sent in enumerate(self.orig_sent_to_data):
            for pos_sent in self.orig_sent_to_data[sent][1]:
                for neg_sent in self.orig_sent_to_data[sent][0]:
                    all_list.append((pos_sent,neg_sent))
            cur_size = len(self.orig_sent_to_data[sent][1])*len(self.orig_sent_to_data[sent][0])
            if (i == 0):
                length_by_sent.append(cur_size)
            else:
                length_by_sent.append(length_by_sent[i-1]+cur_size)
        train_length = train_size*len(all_list)
        valid__length = valid_size*len(all_list)
        last_j = length_by_sent[0]
        train_j = 0
        for j in length_by_sent[1:]:
            if ((last_j < train_length or last_j == length_by_sent[0]) and j >= train_length):
                train_j = j
                all_train= all_list[:j]
            elif (last_j < train_length+valid__length and j>= train_length+valid__length):
                all_val = all_list[train_j:j]
                all_test = all_list[j:]
                break
            last_j = j
        return all_train, all_val,all_test





    def balance_lists(self,l1,l2):
        n1 = len(l1)
        n2 =len(l2)
        if (n1 > n2):
            l1 = l1[:n2]
        elif (n2 > n1):
            l2 = l2[:n1]
        return l1,l2





# class dataPoint:
#
#     RE_LOW = 0
#     RE_HI = 3
#     idf = TFIDF(TFIDF_NAME)
#     def __init__(self,sentences,label,features = None,common_str = None, normalized = False):
#         self.orig_sent1,self.orig_sent2 = sentences
#         self.sent1 = re.sub(PUNCT_REG,"",self.orig_sent1)
#         self.sent2 = re.sub(PUNCT_REG,"",self.orig_sent2)
#         self.label = label
#
#         if (not self.sent1.split()):
#             stop = 1
#         if (normalized):
#             self.normalize_factor = max(1,len(self.sent1.split()))
#         else:
#             self.normalize_factor = 1
#         if (features):
#             self.features = features
#         else:
#             self.calc_features()
#         if (common_str):
#             self.common_str = common_str
#
#     def get_original_sentences(self):
#         return self.orig_sent1, self.orig_sent2
#
#     def get_features(self):
#         return self.features
#
#     def get_common_str(self):
#         return self.common_str
#
#     def calc_features(self):
#         self.features = []
#         s1 = self.sent1.split()
#         s2 = self.sent2.split()
#         l1 = [l for l in s1 if l not in STOPWORDS]
#         l2 = [l for l in s2 if l not in STOPWORDS]
#         l2 = [ps.stem(l) for l in l2]
#         l1 = [ps.stem(l) for l in l1]
#         full_edit_dist,_,_,self.common_str = Measures.fuzzy_substring_dist(s2,s1)
#         #getting edit distance features
#         if (not l2):
#             edit_dist = len(l1)
#             ll2 = 0
#             self.common_str = ""
#         else:
#             ll2 = len(l2)
#             edit_dist,_,_,_ = Measures.fuzzy_substring_dist(l2,l1)
#             self.common_str = " ".join(self.common_str)
#         self.features.append(edit_dist)
#         self.features.append(full_edit_dist)
#         self.features.append(len(s1))
#         #self.features.append(len(s2))
#         self.features.append(len(l1))
#         cont,found_avg, unfound_avg = self.get_word_containement_measure(s1,s2)
#         self.features.append(cont)
#         self.features.append(found_avg)
#         self.features.append(unfound_avg)
#         cont,found_avg, unfound_avg = self.get_word_containement_measure(l1,l2)
#         self.features.append(cont)
#         self.features.append(found_avg)
#         self.features.append(unfound_avg)
#         #self.features.append(ll2)
#         #add longest substring match with replaces
#         self.add_substring_match_features(s1,s2,self.RE_LOW,self.RE_HI,normalize_factor = self.normalize_factor)
#         if not l1 or not l2:
#             for i in range(self.RE_HI+1):
#                 self.features.append(0)
#         else:
#             self.add_substring_match_features(l1,l2,self.RE_LOW,self.RE_HI, normalize_factor = self.normalize_factor)
#         #add longest sequence match featues:
#         self.features.append(Measures.longest_common_sequence(s1,s2) / self.normalize_factor)
#         if not l1 or not l2:
#             self.features.append(0)
#         else:
#             self.features.append(Measures.longest_common_sequence(l1,l2) / self.normalize_factor)
#
#
#
#     def get_word_containement_measure(self,l1,l2):
#         """
#         This method calculates the number of of words from l1 (the pattern) that appear in l2 (the sentence).
#         This measure might be normalized, if self.normalize_factor is not the default 1.
#         :param l1:
#         :param l2:
#         :return:
#         """
#         count = 0
#         found_idfs = []
#         unfound_idfs = []
#         for w in l1:
#             val = self.idf.get_tfidf_val(w)
#             if (val > 10):
#                 val = 10
#             if w in l2:
#                 count += 1
#                 found_idfs.append(val)
#             else:
#                 unfound_idfs.append(val)
#         if (len(found_idfs) == 0):
#             avg_found = 0
#         else:
#             avg_found = np.mean(found_idfs)
#         if (len(unfound_idfs) ==0):
#             avg_unfound = 0
#         else:
#             avg_unfound = np.mean(unfound_idfs)
#
#         return count / self.normalize_factor, avg_found, avg_unfound
#
#     def add_substring_match_features(self,lst1,lst2,re_do,re_up, normalize_factor = 1.):
#         """
#         add substring match features. we measure largest substring match between lst1 and lst2, with replacements.
#         we allow replacements to be from re_do to re_up (meaning re_up-re_do features).
#         :param lst:
#         :param re_do:
#         :param re_up:
#         :return:
#         """
#         for i in range(re_do,re_up+1):
#             self.features.append(Measures.longest_substring_with_replaces(lst1,lst2,i) / normalize_factor)
#
#     def get_label(self):
#         return self.label
#
#     def write_to_file(self,fname):
#         with open(fname,"a",encoding="utf-8") as f:
#             if (self.orig_sent2 == " d keep your friends close your enemies closer "):
#                 stop = 1
#             f.write("[")
#             for feat in self.features[:-1]:
#                 f.write(str(feat)+",")
#             f.write(str(self.features[-1]))
#             f.write("]")
#             f.write("#"+str(self.label))
#             f.write("#"+str(self.orig_sent1))
#             f.write("#"+str(self.orig_sent2))
#             f.write("#"+str(self.get_common_str())+"\n")






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

def gett_precision_recall(y,yhat):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    for i in range(len(y)):
        if y[i] == 0:
            if (yhat[i] == 0):
                true_negative += 1
            else:
                false_positive += 1
        else:
            if (yhat[i] == 1):
                true_positive += 1
            else:
                false_negative += 1

    return true_positive / (true_positive+false_positive), true_positive / (true_positive+false_negative), (true_positive+true_negative)/(len(y))

def fit_model(train_data,val_data, clf):
    # data_points= from_file_to_pd_list(fname)
    # perc = 0.8
    train_x,train_y = train_data

    clf = clf.fit(train_x,train_y)
    acc = check_model_on_data(clf,val_data)
    # print(acc)
    #plt.savefig(figname)
    return clf,acc


def from_file_to_pd_list(fname):
    pd_list = []
    with open(fname, encoding="utf-8") as f:
        line = f.readline()
        while (line):
            d = line.split("#")
            label = int(d[1])
            sentences = (d[2],d[3],d[4])
            #getting the feature vectors
            features = []
            for t in d[0][1:-1].split(","):
                features.append(float(t))
            common_str = d[5][:-1]
            pd_list.append(SCdataPointCat(sentences,label,features,common_str))
            line = f.readline()
        return pd_list


class SCdataPoint():
    RE_LOW = 0
    RE_HI = 3
    WILD_LOW= 1
    WILD_HI = 3
    idf = TFIDF(TFIDF_NAME)


    def add_fuzzy_substring_dist_with_replaces(self,l1,l2,replaces,wild_replaces):
        if (not l2):
            edit_dist  = len(l1)
        else:
            edit_dist= Measures.longest_substring_with_wildcards(l1, l2, replaces, wild_replaces)
        return edit_dist


    def calc_features(self):
        self.features = []
        s1 = self.sent1.split()
        s2 = self.sent2.split()
        l1 = [l for l in s1 if l not in STOPWORDS]
        l2 = [l for l in s2 if l not in STOPWORDS]
        l2 = [ps.stem(l) for l in l2]
        l1 = [ps.stem(l) for l in l1]
        full_edit_dist,_,_,self.common_str = Measures.fuzzy_substring_dist_with_wildcards(s2,s1)
        #getting edit distance features
        if (not l2):
            edit_dist = len(l1)
            ll2 = 0
            self.common_str = ""
        else:
            ll2 = len(l2)
            edit_dist,_,_,_ = Measures.fuzzy_substring_dist_with_wildcards(l2,l1)
            self.common_str = " ".join(self.common_str)
        self.features.append(edit_dist)
        self.features.append(full_edit_dist)
        self.features.append(len(s1))
        #self.features.append(len(s2))
        self.features.append(len(l1))
        #adding word containement measures
        cont,found_avg, unfound_avg = self.get_word_containement_measure(s1,s2)
        self.features.append(cont)
        self.features.append(found_avg)
        self.features.append(unfound_avg)
        cont,found_avg, unfound_avg = self.get_word_containement_measure(l1,l2)
        self.features.append(cont)
        self.features.append(found_avg)
        self.features.append(unfound_avg)
        #self.features.append(ll2)
        #add longest substring match with replaces
        self.add_substring_match_features(s1,s2,normalize_factor = self.normalize_factor)
        if not l1 or not l2:
            for i in range(self.RE_LOW,self.RE_HI+1):
                for j in range(self.WILD_LOW,self.WILD_HI+1):
                    self.features.append(0)
        else:
            self.add_substring_match_features(l1,l2,normalize_factor = self.normalize_factor)

        #add longest sequence match features:
        self.add_longest_sequence_features(s1,s2,self.normalize_factor)
        if not l1 or not l2:
            for i in range(self.WILD_LOW,self.WILD_HI+1):
                self.features.append(0)
        else:
           self.add_longest_sequence_features(l1,l2,self.normalize_factor)

        if (len(self.features) == 40):
            stop = 1

    def add_longest_sequence_features(self,l1,l2,normalize_factor):
        for i in range(self.WILD_LOW,self.WILD_HI+1):
            self.features.append(Measures.longest_common_sequence_with_wildcard(l1,l2,i) / normalize_factor)





    def get_word_containement_measure(self,l2,l1):
        """
        This method calculates the number of of words from l1 (the pattern) that appear in l2 (the sentence).
        This measure might be normalized, if self.normalize_factor is not the default 1.
        :param l1:
        :param l2:
        :return:
        """
        count = 0
        found_idfs = []
        unfound_idfs = []
        for w in l1:
            val = self.idf.get_tfidf_val(w)
            if (val > 10):
                val = 10
            if w in l2:
                count += 1
                found_idfs.append(val)
            else:
                unfound_idfs.append(val)
        if (len(found_idfs) == 0):
            avg_found = 0
        else:
            avg_found = np.mean(found_idfs)
        if (len(unfound_idfs) ==0):
            avg_unfound = 0
        else:
            avg_unfound = np.mean(unfound_idfs)



        return count / self.normalize_factor, avg_found, avg_unfound

    def add_substring_match_features(self,lst1,lst2,normalize_factor = 1.):
        """
        add substring match features. we measure largest substring match between lst1 and lst2, with replacements.
        we allow replacements to be from re_do to re_up (meaning re_up-re_do features).
        :param lst:
        :param re_do:
        :param re_up:
        :return:
        """
        for i in range(self.RE_LOW,self.RE_HI+1):
            for j in range(self.WILD_LOW,self.WILD_HI+1):
                self.features.append(Measures.longest_substring_with_wildcards(lst1, lst2, i, j) / normalize_factor)

    def get_label(self):
        return self.label

    def write_to_file(self,fname):
        with open(fname,"a",encoding="utf-8") as f:
            if (self.orig_sent2 == " d keep your friends close your enemies closer "):
                stop = 1
            f.write("[")
            for feat in self.features[:-1]:
                f.write(str(feat)+",")
            f.write(str(self.features[-1]))
            f.write("]")
            f.write("#"+str(self.label))
            f.write("#"+str(self.orig_sent1))
            f.write("#"+str(self.orig_sent2))
            f.write("#"+str(self.get_common_str())+"\n")


class SCdataPointCat():
    RE_LOW = 0
    RE_HI = 3
    WILD_LOW= 0
    WILD_HI = 3
    CONTAINEMENT_MAX = 8
    LENGTH_MAX = 14
    idf = TFIDF(TFIDF_NAME)

    def get_features(self):
        return self.features

    def __init__(self,sentences,label,features = None,common_str = None, normalized = False):
        self.orig_sent1,self.orig_sent2,self.orig_sent3 = sentences
        self.sent1 = re.sub(PUNCT_REG_WILDCARD,"",self.orig_sent1)
        self.sent2 = re.sub(PUNCT_REG,"",self.orig_sent2)
        self.sent3 = re.sub(PUNCT_REG,"",self.orig_sent3)
        self.label = label

        if (not self.sent1.split()):
            stop = 1
        if (normalized):
            self.normalize_factor = max(1,len(self.sent1.split()))
        else:
            self.normalize_factor = 1
        if (features):
            self.features = features
        else:
            self.calc_features()
        if (common_str):
            self.common_str = common_str

    def get_original_sentences(self):
        return self.sent1,self.sent2, self.sent3

    def add_fuzzy_substring_dist_with_replaces(self,l1,l2,replaces,wild_replaces):
        if (not l2):
            edit_dist  = len(l1)
        else:
            edit_dist= Measures.longest_substring_with_wildcards(l1, l2, replaces, wild_replaces)
        return edit_dist


    def turn_to_one_hot(self,length,ind):
        """
        creating a one-hot vector with 1 in the "ind" index.
        :param length:
        :param ind:
        :return:
        """
        one_hot = [0 for _ in range(length)]
        if (ind >= len(one_hot)):
            one_hot[-1] = 1
        else:
            one_hot[ind] = 1
        return one_hot

    def calc_features(self):

        self.features = []
        s1 = self.sent1.split()
        s2 = self.sent2.split()
        s3 = self.sent3.split()
        # l1 = [l for l in s1 if l not in STOPWORDS]
        # l2 = [l for l in s2 if l not in STOPWORDS]
        # l3 = [l for l in s3 if l not in STOPWORDS]
        # l2 = [ps.stem(l) for l in l2]
        # l1 = [ps.stem(l) for l in l1]
        full_edit_dist,_,_,self.common_str = Measures.fuzzy_substring_dist_with_wildcards(s2,s1)
        full_edit_dist2,_,_,self.common_str = Measures.fuzzy_substring_dist_with_wildcards(s2,s3)
        #getting edit distance features
        # if (not l2):
        #     edit_dist = len(l1)
        #     ll2 = 0
        #     self.common_str = ""
        # else:
        #     ll2 = len(l2)
        #     edit_dist,_,_,_ = Measures.fuzzy_substring_dist_with_wildcards(l2,l1)
        #     self.common_str = " ".join(self.common_str)
        try:
            self.features.append(full_edit_dist / len(s1))
        except:
            print(s1)
            print(s2)
            print(s3)
            raise
        self.features.append(full_edit_dist2 / len(s3))
        # self.features.append(edit_dist / len(l1))
        # self.features.extend(self.turn_to_one_hot(self.LENGTH_MAX,len(l1)))
        #self.features.append(len(s2))
        self.features.extend(self.turn_to_one_hot(self.LENGTH_MAX,len(s1)))
        self.features.extend(self.turn_to_one_hot(self.LENGTH_MAX,len(s3)))
        # self.features.extend(self.turn_to_one_hot(self.LENGTH_MAX,len(s2)))
        # self.features.extend(self.turn_to_one_hot(self.LENGTH_MAX,len(l2)))
        #adding word containement measures
        cont,found_avg, unfound_avg, found_vec, unfound_vec = self.get_word_containement_measure(s1,s2)
        self.features.extend(self.turn_to_one_hot(self.CONTAINEMENT_MAX,cont))
        self.features.append(found_avg)
        self.features.append(unfound_avg)
        self.features.extend(found_vec)
        self.features.extend(unfound_vec)
        cont, found_avg, unfound_avg, found_vec, unfound_vec = self.get_word_containement_measure(s3, s2)
        self.features.extend(self.turn_to_one_hot(self.CONTAINEMENT_MAX, cont))
        self.features.append(found_avg)
        self.features.append(unfound_avg)
        self.features.extend(found_vec)
        self.features.extend(unfound_vec)
        # cont,found_avg, unfound_avg = self.get_word_containement_measure(l1,l2)
        # self.features.extend(self.turn_to_one_hot(self.CONTAINEMENT_MAX, cont))
        # self.features.append(found_avg)
        # self.features.append(unfound_avg)
        #self.features.append(ll2)
        #add longest substring match with replaces
        self.add_substring_match_features(s1,s2,normalize_factor = self.normalize_factor)
        self.add_substring_match_features(s3,s2,normalize_factor = self.normalize_factor)
        # if not l1 or not l2:
        #     for j in range(self.WILD_LOW,self.WILD_HI+1):
        #         self.features.append(0)
        # else:
        #     self.add_substring_match_features(l1,l2,normalize_factor = self.normalize_factor)

        #add longest sequence match features:
        self.add_longest_sequence_features(s1,s2,self.normalize_factor)
        self.add_longest_sequence_features(s3,s2,self.normalize_factor)
        # if not l1 or not l2:
        #     for i in range(self.WILD_LOW,self.WILD_HI+1):
        #         self.features.append(0)
        # else:
        #    self.add_longest_sequence_features(l1,l2,self.normalize_factor)

    def add_longest_sequence_features(self,l1,l2,normalize_factor):
        for i in range(self.WILD_LOW,self.WILD_HI+1):
            self.features.append(Measures.longest_common_sequence_with_wildcard(l1,l2,i) / normalize_factor)


    def list_to_word_count_dict(self,l):
        """
        this method gets a list, and return a dictionary mapping each (word,index) to a count (0 for start)
        :param l:
        :return:
        """
        to_return = {}
        for i,word in enumerate(l):
            to_return[(word,i)] = 0
        return to_return

    def pad_or_cut_vec(self,vec,length):
        """
        this method gets a vector (list) and pads\cuts it to be of the given length
        :param vec:
        :param length:
        :return:
        """
        if len(vec) >= length:
            return vec[:length]
        else:
            to_return = []
            for i in range(length):
                if (i < len(vec)):
                    to_return.append(vec[i])
                else:
                    to_return.append(0.)
            return to_return

    def get_word_containement_measure(self,l2,l1):
        """
        This method calculates the number of of words from l2 (the pattern) that appear in l1 (the sentence).
        This measure might be normalized, if self.normalize_factor is not the default 1.
        :param l1:
        :param l2:
        :return:
        """
        count = 0
        found_idfs = []
        unfound_idfs = []
        word_count_dict = self.list_to_word_count_dict(l1)
        for w in l2:
            was_found = False
            val = self.idf.get_tfidf_val(w)
            if (val > 10):
                val = 10
            for i,w2 in  enumerate(l1):
                if (w2 == w and word_count_dict[(w2,i)] == 0):
                    word_count_dict[(w2,i)]  = 1
                    count += 1
                    found_idfs.append(val)
                    was_found = True
                    break
            if (was_found):
                unfound_idfs.append(val)
        if (len(found_idfs) == 0):
            avg_found = 0
        else:
            avg_found = np.mean(found_idfs)
        if (len(unfound_idfs) ==0):
            avg_unfound = 0
        else:
            avg_unfound = np.mean(unfound_idfs)

        # full idf features
        unfound_vec = list(sorted(unfound_idfs, reverse=True))
        found_vec = list(sorted(found_idfs, reverse=True))
        unfound_vec = self.pad_or_cut_vec(unfound_vec, self.LENGTH_MAX)
        found_vec = self.pad_or_cut_vec(found_vec, self.LENGTH_MAX)

        return count , avg_found, avg_unfound, found_vec, unfound_vec

    def add_substring_match_features(self,lst1,lst2,normalize_factor = 1.):
        """
        add substring match features. we measure largest substring match between lst1 and lst2, with replacements.
        we allow replacements to be from re_do to re_up (meaning re_up-re_do features).
        :param lst:
        :param re_do:
        :param re_up:
        :return:
        """
        for j in range(self.WILD_LOW,self.WILD_HI+1):
            self.features.append(Measures.longest_substring_with_wildcards(lst1, lst2, j) / normalize_factor)

    def get_label(self):
        return self.label

    def get_common_str(self):
        return self.common_str

    def write_to_file(self,fname):
        with open(fname,"a",encoding="utf-8") as f:
            if (self.orig_sent2 == " d keep your friends close your enemies closer "):
                stop = 1
            f.write("[")
            for feat in self.features[:-1]:
                f.write(str(feat)+",")
            f.write(str(self.features[-1]))
            f.write("]")
            f.write("#"+str(self.label))
            f.write("#"+str(self.orig_sent1))
            f.write("#"+str(self.orig_sent2))
            f.write("#"+str(self.orig_sent3))
            f.write("#"+str(self.get_common_str())+"\n")

# FEATURE_NAMES = ["edit dist_wo","edit dist","l(pattern)","l(pattern_wo)","word_cont","found_idf","unfound_idf","word_cont_wo","found_idf_wo","unfound_idf_wo","substring_w_0","substring_w_1","substring_w_2","substring_w_3"
#                     ,"substring_wo_0","substring_wo_1","substring_wo_2","substring_wo_3","sequence_wo","sequence"]
# FEATURE_NAMES_CUT = ["edit dist_wo","edit dist","substring_w_0","substring_w_1","substring_w_2","substring_w_3",
#                  "substring_wo_0","substring_wo_1","substring_wo_2","substring_wo_3","sequence_wo","sequence"]
FEATURE_NAMES = ["non"]*37
def inspect_model(model):
    """

    :param model:
    :return:
    """
    print("The feature importance for this model is ")
    feature_tup = [(FEATURE_NAMES[i],model.feature_importances_[i]) for i in range(len(FEATURE_NAMES))]
    for feat_name,feat_imp in sorted(feature_tup, key= lambda tup:tup[1],reverse=True):
        print(feat_name +":" + str(feat_imp))


def inspect_pair_with_model(model,sent1,sent2,expected_label):
    """
    checking whether the model successfully predicted the pair
    :param model:
    :param sent1:
    :param sent2:
    :param expected_label:
    :return:
    """
    dp = dataPoint((sent1,sent2),expected_label)
    pred = model.predict(np.array(dp.get_features()).reshape(1,-1))
    if (pred == expected_label):
        print("succesfully predicted")
    else:
        print("The model predicted " + str(pred) + " while we actually wanted " + str(expected_label))
        print("The features for these sentences are: ")
        inspect_model(model)



def write_train_validation_test(pd,train_perc, val_perc,train_name,val_name,test_name, balanced = True):
    """

    :param pd:
    :param train_perc:
    :param val_perc:
    :return:
    """
    found = False
    while (not found):
        if (balanced):
            train,valid,test = pd.get_balanced_data_by_sent(train_perc, val_perc)
        else:
            train,valid,test = pd.get_unbalanced_data(train_perc, val_perc)
        if (train and valid and test):
            found = True
    print("train length " + str(len(train)))
    print("val length " + str(len(valid)))
    print("test length " + str(len(test)))

    for p in train:
        p.write_to_file(train_name)
    for p in valid:
        p.write_to_file(val_name)
    for p in test:
        p.write_to_file(test_name)

def feature_file_to_x_y(pd_name):
    pd_list = from_file_to_pd_list(pd_name)
    # x = [np.array(dp.get_features() for dp in pd_list)]
    x = np.array(pd_list[0].get_features())
    for dp in pd_list[1:]:
        try:
            x = np.vstack((x,np.array(dp.get_features())))
        except:
            stop = 1
    y = [dp.get_label() for dp in pd_list]
    return (x,y)

def feature_file_to_x_y_sentences(pd_name):
    import random
    pd_list = from_file_to_pd_list(pd_name)
    random.shuffle(pd_list)
    x = [ ]
    y =  []
    orig_sents = []
    test_sents =  []
    for dp in pd_list:
        x.append(dp.get_features())
        y.append(dp.get_label())
        orig_sent1, test_sent_2,_ = dp.get_original_sentences()
        orig_sents.append(orig_sent1)
        test_sents.append(test_sent_2)

    return (x,y,orig_sents, test_sents)

def check_model_on_data(model, data):
    """

    :param model:
    :param data:
    :return:
    """
    x,y = data
    preds = model.predict(x)
    acc = get_accuracy(preds,y)
    return acc

def write_model_info_to_file(model, model_name, params, param_names, acc,val_acc, fname):
    with open(fname, "w") as f:
        f.write("Information for model: " + model_name)
        f.write("\nThe accuracy for this model is " + str(acc) +"\n")
        f.write("\nThe validation accuracy for this model is " + str(val_acc) +"\n")
        f.write("\n the parameters for the model are: \n")
        for i,name in enumerate(param_names):
            f.write(name + " : " + str(params[i])+"\n")
        if (model_name != SVM):
            f.write("The feature importance is :\n")
            for i,imp in enumerate(model.feature_importances_):
                f.write(FEATURE_NAMES[i] + " : " + str(imp) + "\n")
        else:
            try:
                f.write("\nThe coefficients are: ")
                for i, coeff in enumerate(model.coef_):
                    f.write(FEATURE_NAMES[i] + " : " + str(coeff) + "\n")
            except:
                f.write("No coefficient information - non-linear kernel")


def model_cross_validation(model_type,train_name, val_name,test_name,param_lists):
    import itertools
    train_data_x,train_data_y = feature_file_to_x_y(train_name)
    # for i in train_data_x:
    #     for j in i:
    #         print(j)
    val_data= feature_file_to_x_y(val_name)
    test_data= feature_file_to_x_y(test_name)
    max_acc = 0
    best_model = None
    best_params = None
    count = 0
    for elements in itertools.product(*param_lists):
        print("we are testing for " + str(count))
        count += 1
        # print("checking for " + str(elements))
        train_data = (train_data_x,train_data_y)
        # figname = "rf_fig_d_" + str(m1) + "s_"+str(m2)+".png"
        clf = get_classifier(model_type, elements)
        cur_model,acc = fit_model(train_data,val_data,clf)
        if (acc > max_acc):
            max_acc = acc
            best_model = cur_model
            best_params = elements
    #test the model on the test set:
    print(count)
    test_acc = check_model_on_data(best_model, val_data)
    return best_model, test_acc, max_acc,best_params

def get_accuracy_info(model,test_name):
    test_data= feature_file_to_x_y(test_name)
    test_acc = check_model_on_data(model, test_data)
    return test_acc

DECISION_TREE = "decision_tree"
RANDOM_FOREST = "random_forest"
SVM = "svm"


def get_classifier(type, params):
    """

    :param type:
    :param params:
    :return:
    """
    if (type == DECISION_TREE):
        return tree.DecisionTreeClassifier(max_depth=params[0], min_samples_split = params[1])
    if (type == RANDOM_FOREST):
        return RandomForestClassifier(n_estimators=params[0],max_depth=params[1], min_samples_split = params[2])
    if (type == SVM):
        return SVC(C=params[0], kernel=params[1], degree=params[2])



def write_model(model,acc, model_name, acc_file,best_params):
    with open (acc_file, "a") as f:
        f.write("The accuracy for model " + model_name +" is : " + str(acc)+"\n")
        f.write("The parameters for this model are " + str(best_params) + "\n")
    #saving model:
    joblib.dump(model,"models\\wildcard\\cat\\"+model_name)


def write_mistakes_to_file(model,test_name,fname,write_confidence = False):
    with open(fname,"w",encoding="utf-8") as f:
        test_data = from_file_to_pd_list(test_name)
        for i,pd in enumerate(test_data):
            features = pd.get_features()
            try:
                pred = model.predict(np.array(features).reshape(1,-1))
            except:
                stop = 1
                print(pd.orig_sent1)
                print(pd.orig_sent2)
            label = pd.get_label()
            if (pred != label):
                #writing mistakes to file:
                _,tried ,orig= pd.get_original_sentences()
                f.write("original " + orig)
                f.write("  , we tried for " + tried + "\n")
                f.write("original label is " + str(label) + " , but we gave" + str(pred))
                if (write_confidence):
                    conf = model.decision_function(np.array(features).reshape(1,-1))
                    f.write(" with confidence " + str(conf))
                f.write("\n\n")



def write_precision_recall(model,test_name,fname,write_confidence = False):
    with open(fname,"w",encoding="utf-8") as f:
        test_data = from_file_to_pd_list(test_name)
        y = []
        yhat = []
        for i,pd in enumerate(test_data):
            features = pd.get_features()
            pred = model.predict(np.array(features).reshape(1,-1))
            y.append(pd.get_label())
            yhat.append(pred)
        precision,recall,acc = gett_precision_recall(y,yhat)
        f.write("precision is: " + str(precision) + "\n")
        f.write("recall is: " + str(recall) + "\n")
        f.write("acc is: " + str(acc) + "\n")



def write_corrects_to_file(model,test_name,fname,write_confidence=False):
    with open(fname,"w",encoding="utf-8") as f:
        test_data = from_file_to_pd_list(test_name)
        for i,pd in enumerate(test_data):
            features = pd.get_features()
            pred = model.predict(np.array(features).reshape(1,-1))
            label = pd.get_label()
            if (pred == label):
                #writing mistakes to file:
                _,tried, orig = pd.get_original_sentences()
                f.write("original " + orig)
                f.write("  , we tried for " + tried + "\n")
                f.write("original label is " + str(label) + " , and we gave" + str(pred))
                if (write_confidence):
                    conf = model.decision_function(np.array(features).reshape(1,-1))
                    f.write(" with confidence " + str(conf) )
                f.write("\n\n")

def update_conf_to_freq(conf_to_freq, val, normalize, correct = True):
    """

    :param conf_to_freq:
    :param val:
    :return:
    """
    if val not in conf_to_freq:
        conf_to_freq[val] = 0
    conf_to_freq[val] += 1/normalize



def filter_dict_by_val(d,val):
    new_d = {}
    for key in d:
        if (key >= val):
            new_d[key] = d[key]
    return new_d


def confidence_dict_to_accuracy(d):
    pos = [d[key][0] for key in d]
    neg = [d[key][1] for key in d]
    return np.sum(pos) / (np.sum(pos)+np.sum(neg))

def svm_confidence_percentile_analysis(model,img_name, test_name, just_positive = False):
    from matplotlib import pyplot as plt
    """

    :param model:
    :param img_name:
    :return:
    """
    abs_confidences_to_correct = {}
    test_data = from_file_to_pd_list(test_name)
    all_confidences = []
    l = len(test_data)
    for i,pd in enumerate(test_data):
        features = pd.get_features()
        pred = model.predict(np.array(features).reshape(1,-1))
        conf = model.decision_function(np.array(features).reshape(1,-1))
        if (just_positive and conf <0):
            continue
        # print(conf)
        conf = float(abs(conf))
        all_confidences.append(conf)
        if (conf not in abs_confidences_to_correct):
            abs_confidences_to_correct[conf] = [0,0]
        if (pred == pd.get_label()):
            abs_confidences_to_correct[conf][0] +=1
        else:
            abs_confidences_to_correct[conf][1] +=1
    #getting the accuracy for each confidence percentile
    percentiles = [3*i for i in range(1,33)]
    accuracies = []
    for perc in percentiles:
        cur_val = np.percentile(all_confidences, perc)
        acc = confidence_dict_to_accuracy(filter_dict_by_val(abs_confidences_to_correct,cur_val))
        accuracies.append(acc)

    #plotting the accuracy:
    plt.plot(percentiles,accuracies)
    plt.savefig(img_name)
    plt.clf()
    return accuracies




def svm_confidence_analysis(model,to_write_file,test_name):
    from matplotlib import pyplot as plt
    with open (to_write_file,"a") as f:
        correct_conf_to_freq = {}
        mistake_conf_to_freq = {}
        all_confs_to_freq = {}
        correct_confidence = []
        mistake_confidence = []
        correct_confidence_neg = []
        correct_confidence_pos = []
        mistake_confidence_neg = []
        mistake_confidence_pos = []
        test_data = from_file_to_pd_list(test_name)
        l = len(test_data)
        for i,pd in enumerate(test_data):
            features = pd.get_features()
            pred = model.predict(np.array(features).reshape(1,-1))
            conf = model.decision_function(np.array(features).reshape(1,-1))
            # print(conf)
            conf = float(abs(conf))
            if (conf) not in all_confs_to_freq:
                all_confs_to_freq[conf] = [0,0]
            if (pred == pd.get_label()):
                all_confs_to_freq[conf][0] +=1
                update_conf_to_freq(correct_conf_to_freq,conf,l)
                correct_confidence.append(conf)
                if (pred == 0):
                    correct_confidence_neg.append(conf)
                else:
                    correct_confidence_pos.append(conf)
            else:
                all_confs_to_freq[conf][1] +=1
                update_conf_to_freq(mistake_conf_to_freq,conf,l)
                mistake_confidence.append(conf)
                if (pred == 0):
                    mistake_confidence_neg.append(conf)
                else:
                    mistake_confidence_pos.append(conf)
        f.write("----------------Confidence information--------------------")
        f.write("---------------------------------------------------------\n")
        f.write("The average confidence for the samples we got right is " +str(np.mean(correct_confidence))+"\n")
        f.write("The average confidence for the samples we got wrong is " +str(np.mean(mistake_confidence))+"\n")
        f.write("The median  confidence for the samples we got right is " +str(np.median(correct_confidence))+"\n")
        f.write("The median confidence for the samples we got wrong is " +str(np.median(mistake_confidence))+"\n\n")
        f.write("The average confidence for the samples we got right, as positive, is " +str(np.mean(correct_confidence_pos))+"\n")
        f.write("The average confidence for the samples we got wrong, as positive, is " +str(np.mean(mistake_confidence_pos))+"\n")
        f.write("The median  confidence for the samples we got right,as positive, is " +str(np.median(correct_confidence_pos))+"\n")
        f.write("The median confidence for the samples we got wrong, as positive, is " +str(np.median(mistake_confidence_pos))+"\n\n")
        f.write("The average confidence for the samples we got right, as negative, is " +str(np.mean(correct_confidence_neg))+"\n")
        f.write("The average confidence for the samples we got wrong, as negative, is " +str(np.mean(mistake_confidence_neg))+"\n")
        f.write("The median  confidence for the samples we got right,as negative, is " +str(np.median(correct_confidence_neg))+"\n")
        f.write("The median confidence for the samples we got wrong, as negative, is " +str(np.median(mistake_confidence_neg))+"\n\n")
        #plotting histograms:
        mistake_confs = []
        mistake_freqs = []
        for key in sorted(mistake_conf_to_freq.keys()):
            mistake_confs.append(key)
            mistake_freqs.append(mistake_conf_to_freq[key])
        mistake_freqs = np.cumsum(mistake_freqs)
        plt.title("cumulative histogram - confidence to mistake percentage")
        plt.plot(mistake_confs, mistake_freqs)
        plt.savefig("models\\plots\\svm_mistake_cumulative_3")
        plt.clf()
        correct_confs = []
        correct_freqs = []
        for key in reversed(sorted(correct_conf_to_freq.keys())):
            correct_confs.append(key)
            correct_freqs.append(correct_conf_to_freq[key])
        correct_freqs = np.cumsum(correct_freqs)
        plt.plot(correct_confs, correct_freqs)
        plt.savefig("models\\plots\\svm_correct_cumulative_3")
        plt.clf()

        all_confs= []
        all_conf_corrects = []
        all_conf_mistakes = []
        for key in sorted(all_confs_to_freq):
            all_confs.append(key)
            all_conf_corrects.append(all_confs_to_freq[key][0])
            all_conf_mistakes.append(all_confs_to_freq[key][1])
        all_conf_corrects = np.cumsum(all_conf_corrects)
        all_conf_mistakes = np.cumsum(all_conf_mistakes)
        all_freqs = []
        for i in range(len(all_conf_corrects)):
            all_freqs.append(all_conf_corrects[i] /(all_conf_corrects[i] +all_conf_mistakes[i]))
        plt.plot(all_confs, all_freqs,"ro")
        plt.savefig("models\\plots\\svm_all_freqs_3")
        plt.clf()

        all_confs= []
        all_conf_corrects = []
        all_conf_mistakes = []
        for key in reversed(sorted(all_confs_to_freq)):
            all_confs.append(key)
            all_conf_corrects.append(all_confs_to_freq[key][0])
            all_conf_mistakes.append(all_confs_to_freq[key][1])
        all_conf_corrects = np.cumsum(all_conf_corrects)
        all_conf_mistakes = np.cumsum(all_conf_mistakes)
        all_freqs = []
        for i in range(len(all_conf_corrects)):
            all_freqs.append(all_conf_corrects[i] /(all_conf_corrects[i] +all_conf_mistakes[i]))
        plt.plot(all_confs, all_freqs,"ro")
        plt.savefig("models\\plots\\svm_all_freqs_reversed_3")
        plt.clf()

        all_confs= []
        all_conf_corrects = []
        all_conf_mistakes = []
        for key in sorted(all_confs_to_freq):
            all_confs.append(key)
            all_conf_corrects.append(all_confs_to_freq[key][0])
            all_conf_mistakes.append(all_confs_to_freq[key][1])
        all_conf_corrects = np.cumsum(all_conf_corrects)
        all_conf_mistakes = np.cumsum(all_conf_mistakes)
        all_freqs = []
        for i in range(len(all_conf_corrects)):
            all_freqs.append(all_conf_corrects[i] +all_conf_mistakes[i])
        plt.plot(all_confs, all_freqs,"ro")
        plt.savefig("models\\plots\\svm_all_unnormalized_3")
        plt.clf()

        all_confs= []
        all_conf_corrects = []
        all_conf_mistakes = []
        for key in reversed(sorted(all_confs_to_freq)):
            all_confs.append(key)
            all_conf_corrects.append(all_confs_to_freq[key][0])
            all_conf_mistakes.append(all_confs_to_freq[key][1])
        all_conf_corrects = np.cumsum(all_conf_corrects)
        all_conf_mistakes = np.cumsum(all_conf_mistakes)
        all_freqs = []
        for i in range(len(all_conf_corrects)):
            all_freqs.append(all_conf_corrects[i] +all_conf_mistakes[i])
        plt.plot(all_confs, all_freqs,"ro")
        plt.savefig("models\\plots\\svm_all_unnormalized_eversed_3")
        plt.clf()



import joblib
import os
import pickle
from torch.utils.data import DataLoader
# os.environ["PATH"] += os.pathsep + 'C:\\Users\\sweed\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\graphviz\\__pycache__\\dot.exe'
import sklearn
import torch

class seq2seqDATA(torch.utils.data.IterableDataset):
    def __init__(self,x,y,seq_len):
        self.x = x
        self.y = y
        self.seq_len = seq_len

    def __iter__(self):
        for i in range(len(self.x)):
            try:
                y,mask = self.pad_sequence(self.y[i], 1,True)
            except:
                nir = 1
            yield (torch.FloatTensor(self.pad_sequence(np.array(self.x[i]), EMB_SIZE)),y,mask)

    def __len__(self):
        return len(self.x)

    def pad_sequence(self,seq,emb_size,return_mask=False):
        """

        :param seq:
        :param seq_len:
        :return:
        """
        seq = np.array(seq)
        to_return = np.zeros((self.seq_len, emb_size))
        for i in range(min(self.seq_len, seq.shape[0])):
            to_return[i] = seq[i]
        if (return_mask):
            mask = np.ones(self.seq_len)
            if (self.seq_len > seq.shape[0]):
                for i in range(seq.shape[0],self.seq_len):
                    mask[i] = 0
            return to_return,mask

        return to_return

def crf_exmaple_to_snowclone(model,example,w2v):
    fg = s2s.FeatureGetter(example,w2v)
    it = DataLoader(seq2seqDATA([list(fg)], [[0]*len(example.get_sent_words())], 15), batch_size=1)
    for batch in it:
        x = batch[0]
        preds = model.predict(x)
        cur_tags = [int(x) for x in list(preds[0])]
        new_sent_words = []
        orig_sent_words = example.get_sent_words()
        for i in range(min(len(orig_sent_words), 15)):
            try:
                if cur_tags[i] == 1:
                    if (i ==0 or new_sent_words[-1] != "*"):
                        new_sent_words.append("*")
                else:
                    new_sent_words.append(orig_sent_words[i])
            except:
                stop = 1
        return new_sent_words

if __name__ =="__main__":
    accuracies_by_iteration = []
    # for t in range(1,20):
    accs = []
    for t in range(420,440):
        print(t)
        test_num = str(t)
        max_depths = list(range(1,15))
        min_samples_splits = list(range(2,40,2))
        num_estimators = list(range(10,200,20))
        kernels =["linear", "poly", "rbf"]
        # kernels =["sigmoid"]
        Cs = [0.1*i for i in range(1,10)]
        degrees = [2,3,4]
        dt_model_info_name = "models\\wildcard\\cat\\decision_tree_info_"+test_num
        rf_model_info_name = "models\\wildcard\\cat\\random_forest_info_"+test_num
        svm_model_info_name = "models\\wildcard\\cat\\svm_info_"+test_num
        dt_param_names = ["max depth", "min sample split"]
        rf_param_names = ["max depth", "min sample split", "num_estimators"]
        svm_param_names = ["reg_coeff", "kernel", "degree"]
        fname = "paraphrase_data_2401"
        train_feature_fname = "paraphrase_features\\wildcard\\cat\\train_para_features_balanced_wildcard_"+test_num
        val_feature_fname = "paraphrase_features\\wildcard\\cat\\val_para_features_balanced__wildcard_"+test_num
        test_feature_fname = "paraphrase_features\\wildcard\\cat\\test_para_features_balanced__wildcard_"+test_num
    #     dt_model_path = "decision_tree_best_wildcard_"+test_num
    #     rf_model_path = "random_forest_best_wildcard"+test_num
        svm_model_path = "svm_best_wildcard"+test_num
        acc_path = "models\\wildcard\\cat\\accuracies"
        svm_analysis_path = "models\\wildcard\\cat\\svm_analysis_" + test_num
    #     svm_plot_name = "models\\plots\\cat_positive_svm_confidence_percentiles_" + test_num +".png"
    #     dt_misatkes = "models\\wildcard\\cat\\dt_mistakes_"+test_num
    #     rf_misatkes = "models\\wildcard\\cat\\rf_mistakes"+test_num
        svm_misatkes = "models\\wildcard\\cat\\svm_mistakes" + test_num
    #     dt_correct = "models\\wildcard\\cat\\dt_corrects_"+test_num
    #     rf_correct = "models\\wildcard\\cat\\rf_corrects_"+test_num
        svm_correct = "models\\wildcard\\cat\\svm_corrects" + test_num
    #     svm_misatkes = "models\\wildcard\\cat\\svm_mistakes_"+test_num
    #     # svm_misatkes = "models\\svm_mistakes"
    #     # figname = "decision_tree_fig.png"
    #     # tree_pickle_fname = "models\\" + model_name
    #     decision_tree_params = [max_depths, min_samples_splits]
    #     random_forest_params = [max_depths, min_samples_splits, num_estimators]
        svm_params = [Cs, kernels, degrees]
        # if (t>20):
        model = load_pickle("lstmcrf_k1_model")
        snowclone_db_path = "patterns_db_test"
        sp_reader = s2s.SentencePatternReader(snowclone_db_path)
        w2v = s2s.get_w2v("snowclone_w2v.pkl", sp_reader, should_create=False)
        pd = ParaphraseData(fname, normalized = True,wildcarded=True,model = model,model_function=crf_exmaple_to_snowclone,model_params=w2v)
        # pd = ParaphraseData(fname, normalized = True,wildcarded=False)
    #     # pd.balance_by_sent()
        write_train_validation_test(pd, 0.5,0.25,train_feature_fname,val_feature_fname,test_feature_fname, balanced=False)
    #     print("Testing for decision tree")
    #     #decision tree:
    #     model, acc, val_acc, best_params = model_cross_validation(DECISION_TREE,train_feature_fname,val_feature_fname,test_feature_fname,decision_tree_params)
    #     write_model(model,acc,dt_model_path,acc_path,best_params)
    #     write_model_info_to_file(model,DECISION_TREE,best_params,dt_param_names,acc,val_acc,dt_model_info_name)
    #     print("done")

        # svm:
        print("Testing for SVM")
        # svm = joblib.load("models\\wildcard\\cat\\" + svm_model_path)
        # accs.append(get_accuracy_info(svm,test_feature_fname))
        # svm = joblib.load("models\\wildcard\\" + svm_model_path)
        model, acc,al_acc, best_params = model_cross_validation(SVM,train_feature_fname,val_feature_fname,test_feature_fname,svm_params)
        write_model(model,acc,svm_model_path,acc_path,best_params)
        write_model_info_to_file(model,SVM,best_params,svm_param_names,acc,al_acc,svm_model_info_name)
        print("done")
    #
    #
    #     # #svm analysis
    #     svm = joblib.load("models\\wildcard\\cat\\"+svm_model_path)
        # write_mistakes_to_file(svm,test_feature_fname,svm_misatkes,True)
        write_precision_recall(model,test_feature_fname,"models\\wildcard\\cat\\precision_recall_"+test_num)
    #     # write_corrects_to_file(svm,test_feature_fname,svm_correct,True)
    #     # svm_confidence_analysis(svm,svm_analysis_path,test_feature_fname)
    #     # accuracies_by_iteration.append(svm_confidence_percentile_analysis(svm,svm_plot_name,test_feature_fname, just_positive=True))
    #     #
    #     # # #rf analysis
    #     # rf = joblib.load("models\\wildcard\\"+rf_model_path)
    #     # write_mistakes_to_file(rf,test_feature_fname,rf_misatkes)
    #     # write_corrects_to_file(rf,test_feature_fname,rf_correct)
    #     # #
    #     # # #decision tree analysis
    #     # dt = joblib.load("models\\wildcard\\"+dt_model_path)
    #     # write_mistakes_to_file(dt,test_feature_fname,dt_misatkes)
    #     # write_corrects_to_file(dt,test_feature_fname,dt_correct)
    #     #
    #     # svm = joblib.load("models\\wildcard\\"+svm_model_path)
    #
    # #plotting the accuracies average per confidence
    # confidences = [3*i for i in range(1,33)]
    # avg_accuracy = []
    # for i in range(32):
    #     avg_accuracy.append(np.mean([acc[i] for acc in accuracies_by_iteration ]))
    #     plt.clf()
    # plt.plot(confidences,avg_accuracy)
    # plt.savefig("models\\plots\\positive_avg_accuracy_per_confidence_percentile")


    # pd = ParaphraseData(fname, normalized = True)
    # train,valid,test = pd.get_triplet_data(0.6,0.2)

    #svm analysis
        # svm = joblib.load("models\\wildcard\\cat\\"+svm_model_path)
        # write_mistakes_to_file(svm,test_feature_fname,svm_misatkes,True)
        # write_corrects_to_file(svm,test_feature_fname,svm_correct,True)
        # svm_confidence_analysis(svm,svm_analysis_path,test_feature_fname)
# accuracies_by_iteration.append(svm_confidence_percentile_analysis(svm,svm_plot_name,test_feature_fname, just_positive=True))

    print(accs)
    print(np.mean(accs))




    # train,valid,test = pd.get_balanced_data_by_sent(0.6,0.2)
    # x = 1
    # l1.extend(l2)
    # for dp in l1:
    #     dp.write_to_file(feature_fname)
    # clf = get_best_decision_tree(feature_fname, figname,None)
    # joblib.dump(clf,tree_pickle_fname)
    # clf = joblib.load("paraphrase_dt_1005")
    # inspect_model(clf)
    # print(clf.predict(np.array([1,0,0,0]).reshape(1,4)))

    #analyzing mistakes:
    # s1 = "i am the one who knocks"
    # s2 = "andrumenusian sounds like one lol"
    # expected_label = 0
    # inspect_pair_with_model(clf,s1,s2,expected_label)

# plt.savefig("nirtest2.jpg")

