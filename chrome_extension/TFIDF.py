import numpy as np
import regex as re
import os
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer

ps = PorterStemmer()


STOPWORDS = set(stopwords.words('english'))
TEXT_PATH = "C:\\Users\\sweed\\Desktop\\Masters\\Second\\Lab\\Quotes\\movie_scripts\\dialogs" \
            "\\Action\\miamivice_dialog.txt"



SPEAKER_REG = "[A-Z0-9\\s]+(\\(cont'd\\))*"
QUOTE_REG = "(?!\\w)\\'.*?\\'(?!\\w)"
PUNCT_REG = "[\\[\\].\"\\-,*)(!?#&%$@;:_~\\\^+=/]"
PARANTHESIS_REG = "\\s*\\(.*\\)\\s*\n"
PARANTHESIS_PATTERN = re.compile(PARANTHESIS_REG)
EMPTY_LINE_PATTERN = re.compile("\\s*\n\\s*")
HTML_LINE = re.compile(".*<.*>.*")

SENTENCE_REG = "\\.{3}|[!?\\.]"


def file_to_sentence_list(filename):
    """
    :param filename:
    :return:
    """
    f = open(filename,"r")
    x = f.readlines()
    #cleaning unimportant lines:
    # x_str = ""
    i = 0
    x_size = len(x)

    # for i in range(len(x)):
    while i < x_size:
        if ("thatsall" in x[i]):
            print(x[i])
        if (PARANTHESIS_PATTERN.match(x[i])): #checking for script notes
            # continue
            del x[i]
            x_size -= 1
        elif (EMPTY_LINE_PATTERN.match(x[i])): #check for empty line
            # continue
            del x[i]
            x_size -= 1
        elif (HTML_LINE.match(x[i])):
            # continue
            del x[i]
            x_size -= 1
        else:
            # x_str += " " + x[i][:-1]
            i += 1
    x = get_lines_from_dialogue(x)
    sent_list = get_str_list_by_sentence(x)

    return sent_list



def get_lines_from_dialogue(dialogue_text):
    diag_lines = []
    quote_accu = ""
    has_started = False #a flag to capture whether we found the first speaker or not
    for line in dialogue_text:
        if ("security" in line):
            nir = 1
        if (line.upper() == line): #reached a speaker line
            if (has_started): #the quote has already started
                if (quote_accu != ""):
                    quote_accu = re.sub("\\s+\\s+"," ", quote_accu)
                    diag_lines.append(quote_accu)
                    quote_accu = ""
            else:
                has_started = True
                quote_accu = ""
        else: #accumulating the current string (without any new line char)
            cur_str = line.replace("\n"," ")
            quote_accu += " "+ cur_str
    return diag_lines


def get_str_list_by_sentence(str_list):
    """

    :param str_list:
    :return:
    """
    sentence_list = []
    for quote in str_list:
     #going over all quotes
        sent_list = re.split(SENTENCE_REG, quote)
        for sent in sent_list:
            if (sent != "" and sent !=" " and sent != "\n"):
                sentence_list.append(sent.lower())
    return sentence_list


def str_to_tokens(str):
    """
    this function tokenizes the string given in str to it's words.
    :param str:
    :return:
    """
    word_str = re.sub(PUNCT_REG," ",str)
    # word_str = clean_quote_marks(word_str)
    word_str  = re.sub("\\s\\s"," ", word_str)
    # word_list = word_str.split()
    # if ("thatsall"  in word_list):
    #     nir = 1
    word_list = word_tokenize(word_str)
    #removing the ' :
    final_list = []
    for token in word_list:
        fixed_tok = re.sub("\\s*'","",token)
        final_list.append(fixed_tok)
    return final_list


def create_word_to_idf(tfidf):
    """

    :param tfidf:
    :return:
    """
    word_to_idf = {}
    for word in tfidf.vocabulary_:
        word_to_idf[word] = tfidf.idf_[tfidf.vocabulary_[word]]
    return word_to_idf


DIR_PATH = "C:\\Users\\sweed\\Desktop\\Masters\\Second\\Lab\\Quotes\\movie_scripts\\dialogs"

def get_all_script_lists(folder_path):
    """

    :param folder_name:
    :return:
    """
    all_text = []
    for dirname in os.listdir(folder_path):
        cur_path = os.path.join(folder_path,dirname)
        for script in os.listdir(cur_path):
            script_path  = os.path.join(cur_path,script)
            all_text += file_to_sentence_list(script_path)
    return all_text


#------------------- pickle methods-----------------------------------------------
def save_ojb(obj, name):
    """

    :param obj:
    :param name:
    :return:
    """
    with open('idf\\'+name +".pkl", 'wb') as f:
        pickle.dump(obj,f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('idf\\'+name +".pkl", 'rb') as f:
        return pickle.load(f)





class TFIDF:
    def __init__(self,dict_name):
        self.dict = load_obj(dict_name)

    def get_tfidf_val(self,token):
        if (token in self.dict):
            return self.dict[token]
        elif token in STOPWORDS:
            return 0
        else:
            return 100

    def get_percentile(self, perc):
        return np.percentile(list(self.dict.values()),perc)

    def get_average(self):
        return np.mean(list(self.dict.values()))



def update_word_count(words_lst,word_count):
    """

    :param words_lst:
    :param word_count:
    :return:
    """
    cur_found = []
    for word in words_lst:
        stem_word = word
        if (stem_word.endswith("'")):
            stem_word = stem_word[-1]
        if (stem_word.startswith("'")):
            stem_word = stem_word[1:]
        if (stem_word not in cur_found):
            cur_found.append(stem_word)
            if (stem_word not in word_count):
                word_count[stem_word] = 0
            word_count[stem_word] += 1
    return word_count


def count_doc(document,word_count):
    """

    :param document:
    :param word_count:
    :return:
    """
    tokens = str_to_tokens(document)
    if ("shouldn't" in tokens):
        print(1)
    word_count = update_word_count(tokens,word_count)
    return word_count


def cutoff_word_dict(word_dict, cutoff = 3):
    """
    this methods gets a dictionary mapping words to their counts, and cuts off all words that have a count lower than
    the given cutoff.
    :param word_dict: a dictionary mapping words to their count
    :param cutoff:  the cutoff value, default value is 3
    :return:
    """
    for key in list(word_dict.keys()):
        if (word_dict[key] <= cutoff):
            #deleting the key
            del word_dict[key]
    return word_dict

def get_tfidf(documents):
    D = len(documents)
    word_count = {}
    for doc in documents:
        word_count = count_doc(doc, word_count)
    vals = word_count.values()
    plt.hist(vals)
    plt.show()
    word_count = cutoff_word_dict(word_count,10)
    vals = word_count.values()
    plt.hist(vals)
    plt.show()
    f = open("count_words_10_new.txt","w",encoding="utf-8")
    sorted_x = sorted(word_count.items(), key=operator.itemgetter(1))
    for key in sorted_x[:80000]:
        f.write(key[0] + ": " + str(key[1]))
        f.write("\n")
    f.close()
    # write_keys_to_file(word_count, "fixed_unt.txt")
    # plt.hist(word_count.values(), bins=100)
    # plt.show()
    tfidf_vec = {}
    for key in word_count:
        tfidf_vec[key] = np.log(D / word_count[key])
    return tfidf_vec

def write_keys_to_file(word_cound,fname):
    with open(fname, "w") as f:
        for word in word_cound:
            f.write(word + "\n")

# file_str = file_to_sentence_list(TEXT_PATH)
# tfidf = TfidfVectorizer(tokenizer=str_to_tokens, stop_words="english", use_idf=True)
# tfidf.fit_transform(scripts_list)
# vec = create_word_to_idf(tfidf)
# save_ojb(vec, "full_tfidf3")

# nir = load_obj("full_tfidf")
# sorted_tuples = list(reversed(sorted(nir.items(), key=operator.itemgetter(1))))
# print(sorted_tuples[-1])
# with open("idf.txt", "w") as f:
#     f.write("highest idf\n")
#     for tup in sorted_tuples[:1000]:
#         f.write("%s \n" %str(tup))
#     f.write("lowest idf\n")
#     for tup in sorted_tuples[-1000:]:
#         f.write("%s \n" %str(tup))

#
# tfidf = TFIDF("full_tfidf3")
# print(tfidf.get_percentile(8))
# print(tfidf.get_tfidf_val(" back"))

# nir = file_to_sentence_list("C:\\Users\\sweed\\Desktop\\Masters\\Second\\Lab\\Quotes\\movie_scripts"
#                             "\\dialogs\\Adventure\\127hours_dialog.txt")
# tokens  = [str_to_tokens(x) for x in nir]
# nirnir = 1
# tfidf = TfidfVectorizer(tokenizer=str_to_tokens, stop_words="english", use_idf=True)
# tfidf.fit_transform(nir)
# vec = create_word_to_idf(tfidf)

def join_strings(strngs, size):
    """

    :param lsts:
    :param size:
    :return:
    """
    joined_strngs = []
    i = 0
    while (i < len(strngs)): #going over all lists
        cur_str = ""
        l = min(size, len(strngs)-i)
        for j in range(l):
            cur_str += " " + strngs[i]
            i += 1
        joined_strngs.append(cur_str)
    return joined_strngs

def clean_quote_marks(str):
    """

    :param str:
    :return:
    """
    finds = re.findall( QUOTE_REG,str)
    for x in finds:
        str = str.replace(x,x[1:-1])
    return str



# tfidf_vec = get_tfidf(scripts_list)

# vec = load_obj("nirs_tfidf")
# print(vec["back"])
# with open("keys.txt","w") as f:
#     for key in vec:
#         f.write(key +"\n")

import operator

if __name__ == "__main__":

    # nir = 1
    # scripts_list = get_all_script_lists(DIR_PATH)
    # with open("RC_2011-04_tokenized",encoding="utf-8") as f:
    #     scripts_list = f.readlines()
    # scripts_list = [str_to_tokens("here's looking at")]
    # joined = scripts_list
    # print(len(scripts_list))
    # joined = join_strings(scripts_list,100)
    # print("done")
    # tfidf = get_tfidf(joined)
    # print("done")
    # save_ojb(tfidf,"new_tokenized_tfidf100")
    tfidf = TFIDF("new_tokenized_tfidf100")
    print(tfidf.get_tfidf_val(ps.stem("bazinga")))
    print(tfidf.get_percentile(99.999999))
    # vec = load_obj("tfidf92")
    # vals = list(tfidf.values())
    # plt.hist(vals)
    # plt.show()
    # f = open("lowundict.txt","w")
    # sorted_x = sorted(tfidf.items(), key=operator.itemgetter(1))
    # for key in sorted_x[-8000:]:
    #     f.write(key[0] + ": " + str(key[1]))
    #     f.write("\n")
    # f.close()
    # tfidf_object = TFIDF("unstemmed_tfidf100")
    # print(tfidf_object.get_percentile(20))
    # print(tfidf_object.get_tfidf_val("disturbance"))
    # print(ps.stem("Thanos"))
    # print(word_tokenize("wouldn't"))
    # nirstr = "  no, that was an accident  ken shot him  i never told him charlie was on my side  ken liked to talk  " \
    #          "he shouldn't've been here  but his dad asked me to watch over him  i guess i didn't do the best job  " \
    #          "excuse me, jim  i hate to interrupt and all, but could we just get the money and get the hell out of here  " \
    #          "i'm gonna ask you one more time  and before you think of bullshitting me again, keep in mind i have had a very " \
    #          "frustrating night  and while i know i'll never get the money if i kill you, it's getting to the point where " \
    #          "i just don't care  i'll tell you where it is  but it's not going to do you any good  the guard may not " \
    #          "be coming, but someone else sure as hell is  i'll take my chances  there  well, go get it"
    # count_doc(nirstr,{})
