from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from nltk import tokenize
import re
import LSH
from pydantic import BaseModel
import Measures
import numpy as np
import pickle
import Seq2SeqUtility as s2s
from S2S_LSTM_CRF import crf_exmaple_to_snowclone
import joblib
from TFIDF import TFIDF
TFIDF_NAME = "new_tokenized_tfidf100"
PUNCT_REG_WILDCARD = "[.\"\\-,)(!?#&%$@;:_~\^+=/]"
import utils
import torch.nn as nn
from torchcrf import CRF
import torch
from torch.utils.data import DataLoader
import asyncio

global was_root
was_root = False

class Item(BaseModel):
    text:str



PUNCT_REG = "[\n.\"\\-,*)(!?#&%$@;:_~\^+=/]"

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def break_to_sentences(str):
    """

    :param str:
    :return:
    """
    sent_to_orig = {}
    sentences =  tokenize.sent_tokenize(str)
    new_sentences = []
    for sent in sentences:
        cur_sent = re.sub(PUNCT_REG, "", sent)
        cur_sent = cur_sent.lower()
        sent_to_orig[cur_sent]  = sent
        new_sentences.append(cur_sent)
    return new_sentences, sent_to_orig

def root_help():
    global sent_to_pattern
    sent_to_pattern = {}
    global orig_sent_to_hover
    orig_sent_to_hover = {}
    global orig_sentences
    orig_sentences = []
    # global crf_model
    # crf_model = utils.get_model()

    snowclone_db_path = "patterns_db_test"
    # sp_reader = s2s.SentencePatternReader(snowclone_db_path)
    # w2v = s2s.get_w2v("snowclone_w2v.pkl", sp_reader, should_create=False)
    clf_path = "svm_best_wildcard_"
    global clf
    clf = joblib.load(clf_path + str(113))
    with open("patterns") as f:
        line = f.readline()
        while (line):
            info = line.split(",")

            cur_orig = re.sub(PUNCT_REG, "", info[0])
            orig_sentences.append(cur_orig)
            # example = s2s.PhraseExample(cur_orig, [])
            # predicted_sent = crf_exmaple_to_snowclone(crf_model, example, w2v)
            # sent_to_pattern[cur_orig] = " ".join(predicted_sent)
            # print(info)
            sent_to_pattern[cur_orig] = info[1].split()
            orig_sent_to_hover[cur_orig] = info[-1]
            line = f.readline()

@app.get("/")
async def root():

    root_help()
    was_root = True
    return

WINDOW_SIZE = 2

@app.post("/items/")
async def create_item(item:Item):
    with open("log","w") as f:
        # if not orig_sentences:
        #     root_help()
        await root()
        import LSH
        item_ids = item.text
        # print(item_ids)
        sentences,sent_to_orig = break_to_sentences(item_ids)
        #creating the LSH:
        minhash,lsh = LSH.create_minhashes(sentences,orig_sentences)
        #lsh = LSH.create_lsh(minhash)
        queries = {}
        for sent in orig_sentences:
            # f.write(sent)
            m = LSH.get_minhash_of_sentence(sent)
            query = LSH.get_query_for_minhash(lsh, m)
            queries[sent] = query
        #going over the sentences:
        new_text = ""
        for i,sent in enumerate(sentences):
            print(sent)
            # f.write(sent+"\n")
            to_stop = False
            for orig in orig_sentences:
                if (to_stop):
                    break
                anchor_sent_size=  len(orig.split())
                query = queries[orig]
                if (sent in query):
                    test_sent_split = sent.split()
                    trials = len(test_sent_split) - anchor_sent_size - WINDOW_SIZE + 1
                    if (trials >= 0):
                        for ind in range(trials):
                            cur_sent = " ".join(test_sent_split[ind:ind + anchor_sent_size + WINDOW_SIZE])

                            pred, conf, cur_dp = tree_add_checker_help(sent_to_pattern[orig], cur_sent, orig, clf)
                            print("we're searching for " + cur_sent + ", and got " + str(pred))
                            print("using the original sentence :" + orig)
                            print("and patterns " + " ".join(sent_to_pattern[orig]))
                            print("------------------")
                            if (pred == 1):
                                new_text += "<div class='tooltip'>"
                                new_text += sent_to_orig[sent]
                                new_text += " <span class='tooltiptext'> "
                                new_text += orig_sent_to_hover[orig]
                                new_text += " </span></div>"
                                to_stop = True
                                break
                    else:
                        pred, conf, cur_dp = tree_add_checker_help(sent_to_pattern[orig], sent, orig, clf)
                        print("we're searching for " + sent + ", and got " + str(pred))
                        print("using the original sentence :" + orig)
                        if (pred == 1):
                            new_text += "<div class='tooltip'>"
                            new_text += sent_to_orig[sent]
                            new_text += " <span class='tooltiptext'> "
                            new_text += orig_sent_to_hover[orig]
                            new_text += " </span></div>"
                            to_stop = True
                            break


                    print("***************************************************************************************************")


            if (not to_stop):
                new_text += sent_to_orig[sent]



    # import re
    # new_text = re.sub(patterns[0],"nir",item_ids)
    # print(new_text)
    return {"new_item":new_text}

def tree_add_checker_help(predicted_sent, cur_sent,orig_sent,clf):
    cur_dp = SCdataPointCat((" ".join(predicted_sent), cur_sent,orig_sent), 0, normalized=True)
    features = np.array(cur_dp.get_features()).reshape(1, -1)
    li = len(features)
    pred = clf.predict(features)
    # conf = clf.decision_function(features)
    return pred, 0, cur_dp


def pattern_to_regex(pattern):
    pattern = pattern.replace("\n","")
    pattern_list = pattern.split()
    regex = []
    for x in pattern_list:
        if (x== "*"):
            regex.append("(\\*|(\\s*\\w+\\s*))")
        else:
            regex.append(x)
    return " ".join(regex)

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