import numpy as np
import nltk
import spacy
import pickle
nltk.download('averaged_perceptron_tagger')
START_WORD = "<START>"
START_TAG = 0
STOP_WORD = "<STOP>"
STOP_TAG = 0

# nlp = spacy.load("en_core_web_sm")

# START_SYMBOL = "STARTSYMBOL"
# START_TAG = "START"
POS_TAGS= [START_WORD,STOP_WORD,"CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNS","NNP","NNPS","PDT","POS","PRP","PRP$",
           "RB","RBR","RBS","RP","TO","UH","VB","VBG","VBD","VBN","VBP","VBZ","WDT","WP","WP$","WRB","$",'\'\'']
POS_TAG_TO_IND = {key:i for i,key in enumerate(POS_TAGS)}


NER_TAGS= [START_WORD,STOP_WORD,"ORGANIZATION","PERSON","LOCATION","DATE","TIME","MONEY","PERCENT","FACILITY","GPE","NONER"]
NER_TAG_TO_IND = {key:i for i,key in enumerate(NER_TAGS)}


class FileToData:

    def __init__(self, fname):
        self.examples = self.read_sent_tags(fname)
        nir = 1

    def read_sent_tags(self,fname):
        with open(fname) as f:
            examples = []
            line = f.readline()
            while (line):
                line_info = line.split(",")
                cur_tags = []
                for tag in line_info[1].split():
                    if (tag == "*"):
                        cur_tags.append("1")
                    else:
                        cur_tags.append("0")
                examples.append(PhraseExample(line_info[0],cur_tags))
                line = f.readline()
            return examples

    def get_X_y(self):

        X = []
        y = []
        for example in self.examples:
            fg = FeatureGetter(example)
            y.append(example.tags)
            X.append(list(fg))
        return X, y



class PhraseExample:

    def __init__(self,sent, tags):
        self.sent = START_WORD+" " + sent+ " " + STOP_WORD
        self.tags = [START_TAG]
        self.tags.extend(tags)
        self.tags.append(STOP_TAG)

    def get_tags(self):
        return self.tags

    def get_sent(self):
        return self.sent

    def get_sent_words(self):
        return self.sent.split()


def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


class BertFeatureGetter:
    TFIDF_NAME = "new_tokenized_tfidf100"
    from TFIDF import TFIDF
    idf = TFIDF(TFIDF_NAME)

    def __init__(self, example):
        self.example = example

    def __iter__(self):
        # tag_seq = self.example.get_tags()
        sent_words = self.example.get_sent_words()
        sent_pos_tags = [w[1] for w in nltk.pos_tag(sent_words[1:-1])]
        sent_pos_tags.append(STOP_WORD)
        sent_pos_tags = [START_WORD] + sent_pos_tags
        if ("$" in sent_pos_tags):
            stop = 1
        for i, s in enumerate(sent_words):
            # # yield self.get_dict_features(i,sent_words)
            # if (i == 0):
            #     # last_tag = START_TAG
            #     # before_tag = START_TAG
            #     last_word = START_WORD
            #     before_word = START_WORD
            # else:
            #     # last_tag = tag_seq[i-1]
            #     last_word = sent_words[i-1]
            #     if (i==1):
            #         # before_tag = START_TAG
            #         before_word = START_WORD
            #     else:
            #         # before_tag = tag_seq[i-2]
            #         before_word = sent_words[i-2]
            yield self.get_features(i, sent_words, sent_pos_tags)

    def get_pos_one_hot(self, pos):
        feature_vec = np.zeros((len(POS_TAGS)))
        ind = POS_TAG_TO_IND[pos]
        feature_vec[ind] = 1
        return feature_vec


    def add_w2v_features(self, features, name, size, cur_word):
        if (cur_word not in self.w2v):
            cur_emb = np.zeros((300,))
        else:
            cur_emb = self.w2v[cur_word]
        for i in range(size):
            features[name + str(i)] = cur_emb[i]

    def get_features(self, i, sent_words, sent_pos_tags):
        if (i == 0):
            cur_word = START_WORD
            last_word = START_WORD
            cur_pos = START_WORD
            last_pos = START_WORD
        elif (i == len(sent_words) - 1):
            cur_word = STOP_WORD
            cur_pos = STOP_WORD
            last_word = sent_words[i - 1]
            last_pos = sent_pos_tags[i - 1]
        else:
            cur_word = sent_words[i]
            cur_pos = sent_pos_tags[i - 1]
            last_word = sent_words[i]
            last_pos = sent_pos_tags[i - 1]

        cur_pos_vec = self.get_pos_one_hot(cur_pos)
        last_pos_vec = self.get_pos_one_hot(last_pos)
        # second_pos = self.get_pos_one_hot(last_word)
        w2v_word = self.w2v[cur_word] if cur_word in self.w2v else np.zeros(300)
        w2v_last_word = self.w2v[last_word] if last_word in self.w2v else np.zeros(300)
        # w2V_before_word = self.w2v[before_word] if before_word in self.w2v else np.zeros(300)
        try:
            to_return = np.concatenate((cur_pos_vec, last_pos_vec, w2v_word, w2v_last_word,
                                        [self.idf.get_tfidf_val(cur_word), self.idf.get_tfidf_val(last_word)]))
        except:
            stop = 1
        # [self.idf.get_tfidf_val(cur_word), self.idf.get_tfidf_val(last_word)]
        # cur_pos_vec,last_pos_vec,w2v_word,w2v_last_word
        return to_return

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

class FeatureGetter:
    TFIDF_NAME = "new_tokenized_tfidf100"
    from TFIDF import TFIDF
    idf = TFIDF(TFIDF_NAME)


    def __init__(self,example, w2v):
        self.example = example
        self.w2v = w2v

    def get_ner_of_doc(self,doc):
        return {(' '.join(c[0] for c in chunk), chunk.label() ) for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(doc))) if hasattr(chunk, 'label') }


    def __iter__(self):
        # tag_seq = self.example.get_tags()
        sent_words = self.example.get_sent_words()
        sent = self.example.get_sent()
        sent_ner = self.get_ner_of_doc(sent)
        ner_count = 0
        sent_pos_tags = [w[1] for w in nltk.pos_tag(sent_words[1:-1])]
        sent_pos_tags.append(STOP_WORD)
        sent_pos_tags = [START_WORD] +sent_pos_tags
        if ("$" in sent_pos_tags):
            stop = 1
        for i,s in enumerate(sent_words):
            # # yield self.get_dict_features(i,sent_words)
            # if (i == 0):
            #     # last_tag = START_TAG
            #     # before_tag = START_TAG
            #     last_word = START_WORD
            #     before_word = START_WORD
            # else:
            #     # last_tag = tag_seq[i-1]
            #     last_word = sent_words[i-1]
            #     if (i==1):
            #         # before_tag = START_TAG
            #         before_word = START_WORD
            #     else:
            #         # before_tag = tag_seq[i-2]
            #         before_word = sent_words[i-2]
            yield self.get_features(i,sent_words, sent_pos_tags,sent_ner)



    def get_pos_one_hot(self,pos):
        feature_vec = np.zeros((len(POS_TAGS)))
        ind = POS_TAG_TO_IND[pos]
        feature_vec[ind]  = 1
        return feature_vec

    def get_ner_one_hot(self,ner):
        feature_vec = np.zeros((len(NER_TAGS)))
        ind = NER_TAG_TO_IND[ner]
        feature_vec[ind] = 1
        return feature_vec

    def get_dict_features(self,i, sent_words):
        features = {}
        cur_word = sent_words[i]
        features["cur_idf"] = self.idf.get_tfidf_val(cur_word)
        features["loc"] = i
        features["cur_pos_tag"] = nltk.pos_tag([cur_word])[0][1]
        self.add_w2v_features(features,"cur_word_w2v",300,cur_word)
        if (i==0):
            features["is_first"] = True
            prev_word = START_WORD
        else:
            features["is_first"] = False
            prev_word = sent_words[i-1]
            features["prev_idf"] = self.idf.get_tfidf_val(prev_word)
            features["prev_pos_tag"] = nltk.pos_tag([prev_word])[0][1]
            if (i==1):
                features["is_second"] = True
            else:
                features["is_second"] = False
                back2_word = sent_words[i - 2]
                features["back2__idf"] = self.idf.get_tfidf_val(back2_word)
                features["back_2_pos_tag"] = nltk.pos_tag([back2_word])[0][1]
                # features["back2_tags"] = sent_tags[i-2]
        self.add_w2v_features(features, "prev_word_w2v", 300, prev_word)
        return features

    def add_w2v_features(self,features,name,size,cur_word):
        if (cur_word not in self.w2v):
            cur_emb = np.zeros((300,))
        else:
            cur_emb =self.w2v[cur_word]
        for i in range(size):
            features[name+str(i)] = cur_emb[i]


    def find_ner(self,ner_list,word):
        for pair in ner_list:
            if (pair[0] == word):
                return pair[1]
        return "NONER"

    def get_features(self,i,sent_words,sent_pos_tags,sent_ner):
        if (i ==0):
            cur_word = START_WORD
            last_word = START_WORD
            cur_pos = START_WORD
            last_pos = START_WORD
            cur_ner = START_WORD
            last_ner = STOP_WORD
        elif (i== len(sent_words)-1):
            cur_word = STOP_WORD
            cur_pos = STOP_WORD
            cur_ner = STOP_WORD
            last_word = sent_words[i-1]
            last_pos = sent_pos_tags[i-1]
            last_ner = self.find_ner(sent_ner,last_word)
        else:
            cur_word = sent_words[i]
            cur_pos = sent_pos_tags[i-1]
            last_word = sent_words[i]
            last_pos = sent_pos_tags[i-1]
            last_ner = self.find_ner(sent_ner, last_word)
            cur_ner = self.find_ner(sent_ner, cur_word)



        cur_pos_vec = self.get_pos_one_hot(cur_pos)
        last_pos_vec = self.get_pos_one_hot(last_pos)
        last_ner_vec = self.get_ner_one_hot(last_ner)
        cur_ner_vec = self.get_ner_one_hot(cur_ner)
        # second_pos = self.get_pos_one_hot(last_word)
        w2v_word = self.w2v[cur_word] if cur_word in self.w2v else np.zeros(300)
        w2v_last_word = self.w2v[last_word] if last_word in self.w2v else np.zeros(300)
        # w2V_before_word = self.w2v[before_word] if before_word in self.w2v else np.zeros(300)
        try:
            to_return = np.concatenate((cur_ner_vec,last_ner_vec,cur_pos_vec,last_pos_vec,w2v_word,w2v_last_word,[self.idf.get_tfidf_val(cur_word), self.idf.get_tfidf_val(last_word)]))
        except:
            stop = 1
        #[self.idf.get_tfidf_val(cur_word), self.idf.get_tfidf_val(last_word)]
        #cur_pos_vec,last_pos_vec,w2v_word,w2v_last_word
        return to_return



# import math
# import torch
# from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
# # Load pre-trained model (weights)
# model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
# model.eval()
# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
#
# def score(sentence):
#     tokenize_input = tokenizer.tokenize(sentence)
#     tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
#     loss=model(tensor_input, lm_labels=tensor_input)
#     return math.exp(loss)
#
# print(score("to be or not to be"))
class EmbeddingDict:
    def __init__(self, dict, embedding_size):
        self.dict = dict
        self.unkown_vec = np.zeros((embedding_size,))
    def get (self,key):
        if (key in self.dict):
            return self.dict[key]
        else:
            return self.unkown_vec
    def __getitem__(self, key):
        return self.get(key)




class SnowCloneReader:
    STOPWORDS = ["by", "a", "the","is","in","at","was","how"]
    WILDCARD_SYMBOL = "*"
    WILDCARD_LETTERS = ["X","Y","Z","N","M","V"]


    def __init__(self, fname):
        self.sn_name = fname


    def is_year(self, word):
        import re
        year_reg = "\\d{2,4}"
        if re.match(year_reg,word):
            return True
        else:
            return False


    def clean_pattern(self,pattern):
        """
        this method clears the pattern from wildcard letters (replaces with a single WILDCARD_SYMBOL for any sequence
        of letters).
        :param pattern:
        :return:
        """
        pattern_words = pattern.split()
        new_pattern = []
        for i,word in enumerate(pattern_words):
            if (word in self.WILDCARD_LETTERS):
                if (i == 0 or pattern_words[i-1] not in self.WILDCARD_LETTERS):
                    new_pattern.append(self.WILDCARD_SYMBOL)
            else:
                new_pattern.append(word.lower())
        return new_pattern

    def clean_str(self, str):
        """
        this method cleans the given str from:
        * dates
        *stop words at the end of the string
        :param str:
        :return:
        """
        new_str_words = []
        cur_str_words = str.split()
        for i,word in enumerate(cur_str_words):
            to_add = True
            if (i == len(cur_str_words) - 1):
                if word  in  self.STOPWORDS:
                    to_add = False
                if (self.is_year(word)):
                    to_add = False
            if (to_add):
                new_str_words.append(word.lower())
        return new_str_words

    def write_pairs_to_file(self,to_write):
        """
        this method writes the (Sentence, pattern) pairs that are stored in the original file (given by self.sn_name)
        to the file given in to_write
        :param to_write:
        :return:
        """
        with open (self.sn_name, encoding="utf-8") as f:
            with open(to_write, "w", encoding="utf-8") as wf:
                line = f.readline()
                i = 0
                while(line):
                    if (i == 1345):
                        stop = 1
                    i+=1
                    pattern,sentence,orig_pattern = self.read_line_to_pair(line)
                    sentence = sentence.replace("\n","")
                    wf.write(sentence + " , " + pattern + "," + orig_pattern + "\n")
                    # try:
                    line = f.readline()
                    # except:
                    #     print(line)
                    #     exit()


    def read_line_to_pair(self, line):
        """
        this method reads a line to its (sentence,pattern) pair
        :param line:
        :return:
        """
        if (line == 'The end of the X as we know it   ###     the end of the store as we know it\n'):
            stop = 1
        #splitting line into pattern, sentence:
        new_pattern_words = []
        s = line.split("###")
        sentence = s[1]
        #cleaning the sentence
        sentence_words = self.clean_str(sentence)
        sentence = " ".join(sentence_words)
        pattern_words = self.clean_pattern(s[0])
        pattern_counter = 0
        pattern_size = len(pattern_words)
        if (line.startswith("X and by X I mean Y   ###     proem  and by sweet i mean cover by")):
            stop = 1
        try:
            for i,word in enumerate(sentence_words):
                if (pattern_counter == pattern_size):
                    new_pattern_words.append(self.WILDCARD_SYMBOL)

                elif (pattern_words[pattern_counter] == word):
                    new_pattern_words.append(word)
                    pattern_counter += 1
                elif (pattern_words[pattern_counter] == self.WILDCARD_SYMBOL):
                    if (pattern_counter == pattern_size - 1):
                        pattern_counter += 1
                    #current word is a wildcard, checking if the next one is fixed again so we should increment counter
                    elif pattern_words[pattern_counter+1] == sentence_words[i+1]:
                        pattern_counter += 1
                    new_pattern_words.append(self.WILDCARD_SYMBOL)
        except:

            print(line)
        return " ".join(new_pattern_words), sentence, s[0]





class SentencePatternReader:

    WILCARD_SYMBOL = "*"
    def __init__(self, fname):
        self.pair_db_path = fname
        self.examples_calculated = False
        self.train_perc = 0
        self.val_perc = 0
        self.orig_pattern_to_pair = self.read_all_examples()


    def pattern_to_tags(self, pattern):
        """
        This method turns a string representing a pattern to its binary tags (0\1 for word\wildcard).
        :param pattern:
        :return:
        """
        tags = []
        for word in pattern.split():
            if (word ==  self.WILCARD_SYMBOL):
                tags.append(1.)
            else:
                tags.append(0.)
        return tags

    def get_all_examples(self):
        """

        :return:
        """
        examples = []
        for key in self.orig_pattern_to_pair:
            examples.extend(self.orig_pattern_to_pair[key])
        return examples

    def get_X_y(self, examples,w2v):

        X = []
        y = []
        for example in examples:
            if (example.sent == 'performance ipc considered harmful '):
                stop  = 1
            fg = FeatureGetter(example,w2v)
            y.append(example.tags)
            X.append(list(fg))
            if (len(list(fg)) != len(example.tags)):
                print(example.sent)
            # print("pleasure")
        return X, y

    def read_all_examples(self):
        """
        this method reads the sentence,pattern pairs into a dictionary mapping the original pattern to the pair
        of (sentence, tag)
        :return:
        """
        orig_pattern_to_pair = {}
        with open(self.pair_db_path, encoding="utf-8") as f:
            line = f.readline()
            while (line):
                line_information = line.split(",")
                sent  = line_information[0]
                tags = self.pattern_to_tags(line_information[1])
                cur_example = PhraseExample(sent,tags)
                orig_pattern = line_information[2][:-1]
                if (orig_pattern not in orig_pattern_to_pair):
                    orig_pattern_to_pair[orig_pattern] = []
                orig_pattern_to_pair[orig_pattern].append(cur_example)
                line = f.readline()
        return orig_pattern_to_pair

    def train_val_test_split(self, train_perc, val_perc):
        """

        :param train_perc:
        :param val_perc:
        :return:
        """
        import random
        self.train_perc = train_perc
        self.val_perc = val_perc
        sizes_array = []
        all_size = 0
        keys =list(self.orig_pattern_to_pair.keys())
        random.shuffle(keys)
        for pat in keys:
            if (pat != ""):
                cur_size = len(self.orig_pattern_to_pair[pat])
                all_size += cur_size
                sizes_array.append((pat,cur_size))
        #sorting the sizes array:
        sizes_array=  list(reversed(sorted(sizes_array, key=lambda  x: x[1])))
        train_approx_size = all_size*train_perc
        val_approx_size = all_size*val_perc
        train_patterns = []
        val_patterns = []
        test_patterns = []
        #getting train patterns
        pattern_counter = 0
        i = 0
        while( pattern_counter < train_approx_size):
            pattern_counter += sizes_array[i][1]
            train_patterns.append(sizes_array[i][0])
            i += 1
        #getting val patterns
        pattern_counter = 0
        while (pattern_counter < val_approx_size):
            pattern_counter += sizes_array[i][1]
            val_patterns.append(sizes_array[i][0])
            i += 1
        #getting test patterns
        for j in range(i,len(sizes_array)):
            test_patterns.append(sizes_array[j][0])

        #returning the examples for the train, test, validation:
        self.train_examples = self.pattern_list_to_examples(train_patterns)
        self.val_examples = self.pattern_list_to_examples(val_patterns)
        self.test_examples = self.pattern_list_to_examples(test_patterns)
        self.examples_calculated = True




    def get_train_val_test_examples(self, train_perc, val_perc, reshuffle=False):
        """

        :param train_perc:
        :param val_perc:
        :return:
        """
        if (not self.examples_calculated or  self.train_perc != train_perc or  self.val_perc != val_perc or reshuffle):
            self.train_val_test_split(train_perc,val_perc)
        return self.train_examples, self.val_examples, self.test_examples



    def pattern_list_to_examples(self, pattern_list):
        """

        :param pattern_list:
        :return:
        """
        examples = []
        for pattern in pattern_list:
            examples.extend(self.orig_pattern_to_pair[pattern])
        return examples

    def get_train_val_test_X_y(self,w2v):
        """

        :param train_perc:
        :param val_perc:
        :return:
        """
        if (not self.examples_calculated):
            print("Error: should first split to train-val-test")
            exit()
        return (self.get_X_y(self.train_examples,w2v), self.get_X_y(self.val_examples,w2v), self.get_X_y(self.test_examples,w2v))


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_words_from_examples(examples):
    """

    :param examples:
    :return:
    """
    words = set()
    for example in examples:
        cur_words = example.get_sent_words()
        for word in cur_words:
            words.add(word)
    return words

def get_words_from_train_val_test(train_exmaples, val_examples, test_examples):
    """

    :param train_exmaples:
    :param val_examples:
    :param test_examples:
    :return:
    """
    all_words = set()
    train_words = get_words_from_examples(train_exmaples)
    all_words.union(train_words)
    val_words = get_words_from_examples(val_examples)
    all_words.union(val_words)
    test_words = get_words_from_examples(test_examples)
    all_words.union(test_words)
    return all_words


def create_or_load_slim_w2v(w2v_path, words_list):
    import os
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        w2v_emb_dict[START_WORD] = np.random.normal(size=(300,))
        save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict

def get_w2v(w2v_path,sp_reader, should_create):
    words_list = []
    if (should_create):
        #getting the words list:
        all_examples = sp_reader.get_all_examples()
        words_list = get_words_from_examples(all_examples)
    return create_or_load_slim_w2v(w2v_path, words_list)



# w2v = load_word2vec()
# w2v = EmbeddingDict(w2v,300)
# w2v = {"nir": np.random.normal(size = (300,)),"is": np.random.normal(size = (300,)),"a": np.random.normal(size = (300,)),"king": np.random.normal(size = (300,)),START_SYMBOL:np.random.normal(size=(300,))}
# s = "nir is a king"
# f = FeatureGetter(s, [1,2,3,4])
# for i in f:
#     print(i.shape)
# nltk.download('tagsets')
# print(nltk.help.upenn_tagset())
# fd = FileToData("sentence_patterns.txt")
# X,y = examples_to_X_y(fd.examples)
# nir = 1
# x = nlp("to be or")
# print(x)

# # str = "2 2012   time to make an offer fhfa can't refuse by"
nir = SnowCloneReader("snowcones_patterns_db_2.txt")
nir.write_pairs_to_file("patterns_db_test_upper")
# sp_reader = SentencePatternReader("patterns_db_test")
# w2v = get_w2v("snowclone_w2v.pkl", sp_reader,should_create=False)
# example = sp_reader.get_all_examples()[0]
# fg = FeatureGetter(example,w2v)
# for i in fg:
#     print(i)
#     break

# train,val,test = nir.get_train_val_test_X_y(0.6,0.2)
# print(len(train[1]))
# print(len(val[1]))
# print(len(test[1]))