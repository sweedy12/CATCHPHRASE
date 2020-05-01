
WILDCARD = "(\\*|(\\s*\\w+\\s*))"
# WILDCARD = ".+"
WILDCARD_SYMBOL = "*"

import TreeFromFile as TFF



class OptionNode:
    def __init__(self, father_opt,pattern,pattern_list,freq,id,switched,depth):
         self.father = father_opt
         self.pattern = pattern
         self.pattern_list = pattern_list
         self.freq = freq
         self.id = id
         self.children = []
         self.switched = switched
         self.depth = depth


    def get_pattern(self):
        return self.pattern

    def add_child(self,child):
        self.children.append(child)

    def has_switched(self,i):
        return i in self.switched

    def get_swithced(self):
        return self.switched

    def get_id(self):
        return self.id
    def get_children(self):
        return self.children
    def get_depth(self):
        return self.depth

    def get_full_sent(self):
        return self.pattern

    def get_pattern_list(self):
        return self.pattern_list




class PatternTreeCreator:
    WORD_OR_SPACE = "(\\w|\\s)*"
    def __init__(self, tree_path):
        self.root = TFF.TreeFromFile(tree_path)
        self.get_sentences()
        self.start_sent = self.root.get_root().get_int_sent()
        self.get_all_patterns()

    def count_pattern(self, pattern):
        import re
        count = 0
        for i,sent in enumerate(self.sentences):
            if re.match(pattern,sent):
                count += self.appearances[i]
        return count / self.total_appearances
    def get_all_patterns(self):
        id = 0
        pattern_list = self.start_sent.split()
        patterns_to_options = {}
        n = len(pattern_list)
        freq = self.count_pattern(self.start_sent)
        self.start_pat = OptionNode(None, self.start_sent,pattern_list,freq,id,[],0)
        patterns_to_options[self.start_sent] = self.start_pat
        nodes = [self.start_pat]
        id +=1
        while(nodes):
            cur_node = nodes.pop()
            cur_depth = cur_node.get_depth()+1
            switched = cur_node.get_swithced()
            pattern_list = cur_node.get_pattern_list()
            for i in range(n):
                if not cur_node. has_switched(i):
                    cur_switched = switched[:]
                    cur_switched.append(i)
                    cur_pattern_list = pattern_list[:]
                    cur_pattern_list[i] = self.WORD_OR_SPACE
                    new_pattern_list = []
                    last_l = ""
                    for i in range(len(cur_pattern_list)):
                        if (cur_pattern_list[i] == self.WORD_OR_SPACE):
                            if (last_l != self.WORD_OR_SPACE):
                                last_l = self.WORD_OR_SPACE
                                new_pattern_list.append(cur_pattern_list[i])

                        else:
                            new_pattern_list.append(cur_pattern_list[i])
                            last_l=  cur_pattern_list[i]
                    pattern_str = " ".join(new_pattern_list)
                    pattern_str = pattern_str.replace(self.WORD_OR_SPACE+" ",self.WORD_OR_SPACE)
                    # if (pattern_str in patterns_to_options):
                    #     cur_option = patterns_to_options[pattern_str]
                    # else:
                    cur_freq = self.count_pattern(pattern_str)
                    cur_option = OptionNode(cur_node,pattern_str,cur_pattern_list,cur_freq,id,cur_switched,cur_depth)
                    id +=1
                    patterns_to_options[pattern_str] = cur_option
                    if (pattern_str != cur_node.get_pattern()):
                        cur_node.add_child(cur_option)
                        nodes.append(cur_option)






    def get_sentences(self):
        self.sentences = []
        self.appearances = []
        self.total_appearances = 0
        nodes = [self.root.get_root()]
        while (nodes):
            cur_node = nodes.pop()
            cur_sent = cur_node.get_int_sent()
            self.sentences.append(cur_sent)
            cur_app = cur_node.get_times()
            self.appearances.append(cur_app)
            self.total_appearances += cur_app
            children = cur_node.get_children()
            if (children):
                nodes.extend(children)


class ReversePatternFinder:
    WORD_OR_SPACE = "(\\w|\\s)*"
    def __init__(self, tree_path, threshold):
        self.threshold = threshold
        self.root = TFF.TreeFromFile(tree_path)
        self.get_sentences()
        self.start_sent = self.root.get_root().get_int_sent()
        self.count_all_patterns()

    def get_sentences(self):
        self.sentences = []
        self.appearances = []
        self.total_appearances = 0
        nodes = [self.root.get_root()]
        while (nodes):
            cur_node = nodes.pop()
            cur_sent = cur_node.get_int_sent()
            self.sentences.append(cur_sent)
            cur_app = cur_node.get_times()
            self.appearances.append(cur_app)
            self.total_appearances += cur_app
            children = cur_node.get_children()
            if (children):
                nodes.extend(children)


    def count_all_patterns(self):
        pattern_list = self.start_sent.split()
        self.n_to_pattern_freq = {}
        pattern_to_freq = {}
        pattern_to_freq[self.start_sent] = self.count_pattern(self.start_sent)
        self.n_to_pattern_freq[0] = pattern_to_freq
        n = len(pattern_list)
        subsets = [[]]
        for i in range(1,n):
            pattern_to_freq = {}
            new_subsets = []
            for j in range(n):
                ind = 0
                s = len(subsets)
                while (ind < s):
                    sub = subsets[ind]
                    if (j not in sub):
                        cur_sub = list(sorted(sub + [j]))
                        if (cur_sub not in new_subsets and cur_sub not in subsets):
                            new_subsets.append(cur_sub)
                            cur_pattern = []
                            last_t = ""
                            for v in range(len(pattern_list)):
                                if (v not in cur_sub):
                                    cur_pattern.append(pattern_list[v])
                                    last_t = ""
                                elif (last_t != self.WORD_OR_SPACE):
                                    cur_pattern.append(self.WORD_OR_SPACE)
                                    last_t = self.WORD_OR_SPACE
                            pattern_str = " ".join(cur_pattern)
                            pattern_str = pattern_str.replace(self.WORD_OR_SPACE+" ",self.WORD_OR_SPACE)
                            if (pattern_str not in pattern_to_freq):
                                print(pattern_str)
                                cur_count = self.count_pattern(pattern_str)
                                print(cur_count)
                                print("------------------------------------")
                                pattern_to_freq[pattern_str] = cur_count
                    ind += 1
            self.n_to_pattern_freq[i] = pattern_to_freq
            print("*********************************************************************")
            print("*********************************************************************")

            subsets.extend(new_subsets)

    def write_patterns(self,fname):
        pattern_list = self.start_sent.split()
        n = len(pattern_list)
        with open (fname,"w", encoding="utf-8") as f:
            f.write("Pattern frequency for " + self.start_sent+"\n")
            for i in range(n):
                cur_dict = self.n_to_pattern_freq[i]
                f.write("Pattern frequency for switching " + str(i) + " words\n")
                for key in cur_dict:
                    f.write(key + " : " + str(cur_dict[key]) +"\n")
                f.write("------------------------------------------------------------------\n")

    def get_best_for_n(self,n,pattern_n):
        import numpy as np
        best_list = []
        smallest_ind = -1
        smallest_val = np.inf
        if (n >= len(self.start_sent.split())):
            n = len(self.start_sent.split()) -1
        for key in self.n_to_pattern_freq[n]:
            to_add = self.n_to_pattern_freq[n][key]
            if (len(best_list) < pattern_n):
                best_list.append((key,to_add))
                if (smallest_val > to_add):
                    smallest_val = to_add
                    smallest_ind = len(best_list) -1
            else:
                if (to_add > smallest_val):
                    best_list[smallest_ind] = (key,to_add)
                    smallest_val = to_add
                    for i,val in enumerate(best_list):
                        if (val[1] < smallest_val):
                            smallest_val =val[1]
                            smallest_ind = i
        return n,best_list


    def find_best_patterns(self,n,max_n,pattern_n,high_threshold, low_threshold):
        cur_n = 0
        biggest = 0
        threshold = low_threshold
        max_n = min(len(self.start_sent.split()),max_n)
        while (biggest < low_threshold and cur_n <= n):
            cur_n, best_list = self.get_best_for_n(cur_n, pattern_n)
            best_vals = [tup[1] for tup in best_list]
            biggest = max(best_vals)
            best_n = cur_n
            cur_n += 1
        if (biggest < low_threshold):
            threshold = high_threshold
            while (biggest < high_threshold and cur_n < max_n ):
                cur_n, best_list = self.get_best_for_n(cur_n, pattern_n)
                best_vals = [tup[1] for tup in best_list]
                biggest = max(best_vals)
                best_n = cur_n
                cur_n += 1
            if (biggest < high_threshold):
                threshold = 0
                best_n, best_list = self.get_best_for_n(n, pattern_n)
        return best_n, best_list, threshold




    def write_best_pattern(self,fname,n,max_n,pattern_n,high_threshold, low_threshold):
        best_n, best_list,threshold = self.find_best_patterns(n,max_n,pattern_n,high_threshold,low_threshold)
        with open(fname,"w") as f:
            f.write("Using threshold " + str(threshold)+"\n")
            f.write("The best patterns for this quote for switching " + str(best_n) + " words with wildcards are:\n")
            for tup in best_list:
                f.write(tup[0] +" with value: " + str(tup[1])+"\n\n")






    # def get_all_subsets(self,n):

    def count_pattern(self, pattern):
        import re
        count = 0
        for i,sent in enumerate(self.sentences):
            if re.match(pattern,sent):
                count += self.appearances[i]
        return count / self.total_appearances


class PatternFinder:
    def __init__(self, tree_path, threshold):
        self.threshold = threshold
        self.root = TFF.TreeFromFile(tree_path)
        self.ngram_to_count = {}
        self.sent_to_ngrams ={}
        self.get_higest_n()
        # self.count_ngrams()
        self.get_sentences()
        self.count_ngrams_wildcard()
        # self.turn_to_wildcard(1,2)
        # self.count_word_place()
        # self.set_place_to_max()



    def count_ngrams_wildcard(self):
        i = 1
        cont = True
        while(cont):
            has_turned = True
            while (has_turned):
                self.count_ngrams_help(i)
                has_turned = self.turn_to_wildcard(i, self.threshold)
                print(i)
                try:
                    print(self.ngram_to_count[i])
                except:
                    nir = 1
                if (self.ngram_to_count[i] == {}):
                    print("hellloooo")
                    self.max_n = i - 1
                    cont = False
                    break
            i += 1
        print("out")



    def count_ngrams_help(self,n):
        import re
        count = {}
        total_count = 0
        for ap,sent in enumerate(self.sentences):
            sent_str = " ".join(sent)
            self.sent_to_ngrams[(sent_str,n)] = []
            total_count += self.appearances[ap]
            for i in range(len(sent)-n+1):
                cur_sent = " ".join(sent[i:i+n])
                if (cur_sent not in count):
                    count[cur_sent] = 0
                if (cur_sent not in self.sent_to_ngrams[(sent_str,n)]):
                    count[cur_sent] += self.appearances[ap]
                    self.sent_to_ngrams[(sent_str,n)].append(cur_sent)
        print("start")
        new_counts = {}
        for key in count:
            new_counts[key] = count[key]
            if (WILDCARD_SYMBOL in key.split()):
                new_key = key
                new_key = new_key.replace(WILDCARD_SYMBOL,WILDCARD)
                for key2 in count:
                    if (key != key2):
                        if re.match(new_key,key2):
                            new_counts[key] += count[key2]
        print("end")

        for key in count:
            new_counts[key] /= total_count
            if new_counts[key] > 1:
                new_counts[key] = 1
        self.ngram_to_count[n] = new_counts



    def get_pattern(self):
        patterns = []
        max_pat = ""
        max_val = 0
        cur_n = self.max_n
        to_continue = True
        while (to_continue):
            for pat in self.ngram_to_count[cur_n]:
                cur_val = self.ngram_to_count[cur_n][pat]
                if cur_val >=self.threshold:
                    patterns.append(pat)
                if (cur_val > max_val and cur_n == self.max_n):
                    max_val = cur_val
                    max_pat = pat
            print(patterns)
            if (patterns):
                to_continue = False
            cur_n -= 1
        return patterns, max_pat



    def get_sentences(self):
        self.sentences = []
        self.appearances = []
        nodes = [self.root.get_root()]
        while (nodes):
            cur_node = nodes.pop()
            cur_sent = cur_node.get_int_sent()
            self.sentences.append(cur_sent.split())
            self.appearances.append(cur_node.get_times())
            children = cur_node.get_children()
            if (children):
                nodes.extend(children)



    def check_all_patterns(self,i,threshold,cur_pat):
        import re
        if (self.ngram_to_count[i][cur_pat] >= threshold):
            return True
        if (i < 2):
            return False
        for pat in self.ngram_to_count[i]:
            pat_reg = pat.replace(WILDCARD_SYMBOL,WILDCARD)
            if (re.match(pat_reg,cur_pat)):
                if (self.ngram_to_count[i][pat] >= threshold):
                    return True
        return False


    def turn_to_wildcard(self,n,threshold):
        has_turned = False
        for ind,sent in enumerate(self.sentences):
            if (len(sent) < n):
                continue
            if (ind == 54):
                stop = 1
            should_change = [True]*len(sent)
            for i in range(len(sent)-n+1):
                cur_sent = " ".join(sent[i:i+n])
                try:
                    if (self.check_all_patterns(n,threshold,cur_sent)):
                        for t in range(n):
                            should_change[i+t] = False
                    else:
                        stop = 1
                except:
                    stop = 1
            new_sent = []
            for i,d in enumerate(should_change):
                if not d:
                    new_sent.append(sent[i])
                if (d):
                    x = len(new_sent)
                    if (x == 0):
                        new_sent.append(WILDCARD_SYMBOL)
                        if (sent[i] != WILDCARD_SYMBOL):
                            has_turned = True
                    else:
                        if (new_sent[x-1] != WILDCARD_SYMBOL):
                            new_sent.append(WILDCARD_SYMBOL)
                            if (sent[i] != WILDCARD_SYMBOL):
                                has_turned = True
            final_sent = []
            for i in range(len(new_sent)):

                if (new_sent[i] != WILDCARD_SYMBOL):
                    final_sent.append(new_sent[i])
                elif (i == len(new_sent)-1):
                    final_sent.append(WILDCARD_SYMBOL)
                elif new_sent[i+1] != WILDCARD_SYMBOL:
                    final_sent.append(WILDCARD_SYMBOL)

            self.sentences[ind] = final_sent
        return has_turned



    def count_ngrams(self):
        for i in range(1,self.max_n+1):
            self.ngram_to_count[i]= {}

        nodes = [self.root.get_root()]
        while (nodes):
            cur_node = nodes.pop()
            cur_sent_list = cur_node.get_int_sent().split()
            l = len(cur_sent_list)
            for i in range(l):
                for j in range(1,self.max_n+1):
                    #checking if we can extract n-gram:
                    if (i+j) <= l:
                        cur_n_gram = " ".join(cur_sent_list[i:i+j])
                        if cur_n_gram not in self.ngram_to_count[j]:
                            self.ngram_to_count[j][cur_n_gram] = 0
                        self.ngram_to_count[j][cur_n_gram] += 1
                        #checking if any other key needs an update:

            children = cur_node.get_children()
            if children:
                nodes.extend(children)



    def get_higest_n(self):
        max_n = 0
        self.sent_num = 0
        nodes = [self.root.get_root()]
        while (nodes):
            cur_node  = nodes.pop()
            self.sent_num += 1
            cur_sent = cur_node.get_int_sent()
            cur_n = len(cur_sent.split())
            if (cur_n > max_n):
                max_n = cur_n
            children = cur_node.get_children()
            if (children):
                nodes.extend(children)
        self.max_n = max_n


    def count_word_place(self):
        # self.word_count = {}
        self.place_count = {}
        self.sent_num = 0
        self.word_place_count = {}
        nodes = [self.root.get_root()]
        while(nodes):
            cur_node = nodes.pop()
            cur_sent  = cur_node.get_int_sent()
            self.count_sent(cur_sent)
            children = cur_node.get_children()
            if (children):
                nodes.extend(children)
            self.sent_num += 1
        for key in self.word_place_count:
            word, place=  key
            self.word_place_count[(word,place)] /= self.place_count[place]
        print(self.word_place_count)
    def get_word_place(self):
        return self.word_place_count


    def count_sent(self,sent):
        for i,word in enumerate(sent.split()):
            if (i not in self.place_count):
                self.place_count[i] = 0
            if ((word,i) not in self.word_place_count):
                self.word_place_count[(word,i)] = 0
            self.word_place_count[(word,i)] += 1
            self.place_count[i] += 1

    def set_place_to_max(self):
        self.place_to_max = {}
        self.place_to_options = {}
        for key in self.word_place_count:
            word, place = key
            if (place not in self.place_to_options):
                self.place_to_options[place] = ([],0)
            if (word not in self.place_to_options[place][0]):
                x = self.place_to_options[place][0]
                count = self.place_to_options[place][1]
                x.append(word)
                self.place_to_options[place] = (x,count + 1)
            if (place not in self.place_to_max):
                self.place_to_max[place] = (word, self.word_place_count[key])
            else:
                if (self.place_to_max[place][1] < self.word_place_count[key]):
                    self.place_to_max[place] = (word, self.word_place_count[key])

    def recover_best_pattern(self):

        vals = self.place_to_max.keys()
        print(max(vals))
        pat = [0]*(max(vals)+1)
        for v in vals:
            pat[v] = (self.place_to_max[v],self.place_to_options[v][1])
        new_pat = []
        i = 1
        new_pat.append(pat[0][0][0])
        while (i < len(pat)):
            if pat[i][0] == pat[i-1][0]:
                i += 1
                continue
            else:
                new_pat.append(pat[i][0][0])
                i += 1

        return pat, new_pat

    def save_to_file(self, fname):
        print("in again")
        patterns, max_pat = self.get_pattern()
        print("out now")
        with open(fname, "w", encoding="utf-8") as f:
            f.write("The best pattern found is :\n" + max_pat + "\n")
            f.write("The patterns that passed the threshold are:\n")
            for pat in patterns:
                f.write(pat + "\n")
            f.write("The n_gram relative count for each n is: \n")
            for i in range(1,self.max_n+1):
                f.write("n = " + str(i) + "\n")
                f.write(dict_to_str(self.ngram_to_count[i]))
                f.write("\n---------------------------------\n")

def dict_to_str(d):
    new_str = ""
    for key in d:
        new_str += str(key) + ":" + str(d[key]) + "\n"
    return new_str

def list_tup_to_str(ltup):
    new_str = ""
    for l in ltup:
        for t in l:
            new_str += str(t) + ","
        new_str += "  "
    return new_str



def read_patterns_from_file(fname):
    """

    :param fname:
    :return:
    """
    with open(fname, encoding="utf-8") as f:
        line  = f.readline()
        to_parse = False
        patterns_acc = []
        while(line):
            if (line == "The patterns that passed the threshold are:\n"):
                to_parse = True
            elif (line == "The n_gram relative count for each n is: \n"):
                break
            elif to_parse:
                patterns_acc.append(line[:-1])
            line = f.readline()
    return patterns_acc

def switch_pattern(pattern):
    """

    :param pattern:
    :return:
    """
    return pattern.replace("*","\w+")

def write_all_patterns(folder_path,fname):
    """

    :param fname:
    :return:
    """
    import sqlite3 as sql
    import os
    conn = sql.connect("QUOTES.db")
    with open(fname,"w", encoding="utf-8") as f:
        for d in os.listdir(folder_path):
            line = conn.execute("SELECT * FROM QUOTE_DESC WHERE QUOTE =?",(d,))
            for l in line:
                text = l[1]
            cur_path = folder_path + "\\" + d
            cur_patterns = read_patterns_from_file(cur_path)
            for pat in cur_patterns:
                f.write(switch_pattern(pat) + ","+text+"\n")


#
import os
import TreePlotter as tf
THRESHOLD = 0.4
path = "3003_depth1"
# pf = ReversePatternFinder(path+"\\one does not simply walk into mordor",0.3)
# pf.count_all_patterns()
# nir = 1
try:
    os.mkdir("best_reverse_patterns_1204_2")
    os.mkdir("reverse_patterns_0904")
except:
    pass
n = 3
pattern_n = 3
max_n = 5
for d in os.listdir(path):
    print(d)
    new_path = path + "\\" + d
    pf = PatternTreeCreator(new_path)
    tf.plot_tree(pf.start_pat,"nirtest_" + d)
    x = 1
    # pf.write_patterns("reverse_patterns_0904\\" + d)
    # pf.write_best_pattern("best_reverse_patterns_1204_2\\"+d,n,max_n,pattern_n,0.7,0.35)
    # patterns = pf.get_pattern()
    # for pattern in patterns:
    #     f.write(d + " : " + pattern + "\n")
#     pf.save_to_file("patterns\\night\\" + d)

# print(read_patterns_from_file("ngram_patterns3003\\i see dead people"))
# print(switch_pattern("artic kartiv * banana *"))



# pf = PatternFinder(path, 0.1)
# print(pf.get_pattern())
# nir = 1
# vec = pf.get_word_place()
# for v in vec:
#     word, place = v
#     if place == 6:
#         print(word)
#         print(vec[v])

# bp,np = pf.recover_best_pattern()
# print(bp)
# print(np)
# pf.save_to_file("nirnirnir")
# write_all_patterns("ngram_patterns3103_03","all_patterns_03.txt")


# def get_all_subsets(n):
#     l = list(range(n))
#     for subset in subsets_generator(l):
#         print(subset)
#
# def subsets_generator(l,i=0):
#
#     if (i != len(l)):
#         x = subsets_generator(l, i+1)
#         yield [l[i]]
#         for l1 in x:
#             yield l1
#             yield l1+[l[i]]
#

#
# def incremental_subsets(n):
#     subsets = [[]]
#     for i in range(n):
#         new_subsets = []
#         for j in range(n):
#             ind = 0
#             s = len(subsets)
#             while (ind < s):
#                 sub = subsets[ind]
#                 if (j not in sub):
#                     cur_sub = list(sorted(sub + [j]))
#                     if (cur_sub not in new_subsets and cur_sub not in subsets):
#                         new_subsets.append(cur_sub)
#                 ind += 1
#         subsets.extend(new_subsets)
#     return subsets




# get_all_subsets(3)

# for sub in incremental_subsets(5):
#     print(sub)


