import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re as re
from nltk.stem import PorterStemmer
import nltk

nltk.download("stopwords")

STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()


PUNCT_REG = "[.\"\\-,*)(!?#&%$@;:_~\^+=/]"


def substring_edit_dist(l1, l2):
    """

    :param s1:
    :param s2:
    :return:
    """
    l1 = [ps.stem(l) for l in l1]
    l2 = [ps.stem(l) for l in l2]
    if (len(l1) > len(l2)):
      return edit_dist_help(l1,l2,len(l1),len(l2))
    #going through every substring
    s = len(l1)
    best_dist = np.inf
    for i in range(len(l2)-s+1):
        cur_l2 = l2[i:i+s]
        cur_dist = edit_dist_help(l1,cur_l2,len(l1), len(cur_l2))
        if (cur_dist < best_dist):
            best_dist = cur_dist
    return best_dist

def edit_dist_help(l1,l2,n,m):
    """

    :param l1:
    :param l2:
    :param n:
    :param m:
    :return:
    """
    if (m ==0):
        return n
    if (n == 0):
        return m
    if (l1[n-1]==l2[m-1]):
        return edit_dist_help(l1,l2,n-1,m-1)
    else:
        edit_list = [edit_dist_help(l1,l2,n-1,m),edit_dist_help(l1,l2,n,m-1),edit_dist_help(l1,l2,n-1,m-1)]
        return 1+min(edit_list)



STOP = "stop"


def list_to_stop_mapping(l):
    """
    """
    stop_map = {}
    stopped_list = []
    j = 0
    for i,x in enumerate(l):
        if x not in STOPWORDS:
            stop_map[j] = i
            stopped_list.append(x)
            j += 1
    return stopped_list, stop_map

def fuzzy_decision_tree(p1,p2):
    """
    deciding whether or not to create an edge between p1 and p2.
    """
    s1 = p1.split()
    s2 = p2.split()
    s = len(s1)
    l1,stop_map = list_to_stop_mapping(s1)
    l2 = [l for l in s2 if l not in STOPWORDS]
    l2 = [ps.stem(l) for l in l2]
    l1 = [ps.stem(l) for l in l1]
    if (not l1 or not l2):
        return False
    lp = min(len(s1),len(s2))
    ls = min(len(l1),len(l2))
    d, i1,i2,caught_str  = fuzzy_substring_dist(l1,l2)
    orig_str = " ".join(s1[stop_map[i1]:stop_map[i2]+1])
    #deciding
    if (ls >=2 and d ==0):
        return STOP
    if (lp == 4 and ls ==4 and d<=1):
        return orig_str
    if (lp == 5 and ls >5 and d<=1):
        return orig_str
    if (lp ==6 and ls>=5 and d<=1):
        return orig_str
    if (lp > 6 and ls >3 and d<=2):
        return orig_str
    return False


def decision_tree(p1,p2):
    """"
    """

    # p1 = re.sub(PUNCT_REG," ",p1)
    # p2 = re.sub(PUNCT_REG," ",p2)
    s1 = p1.split()
    s2 = p2.split()
    s = len(s1)
    l1 = [l for l in s1 if l not in STOPWORDS]
    l2 = [l for l in s2 if l not in STOPWORDS]
    l1 = [ps.stem(l) for l in l1]
    l2 = [ps.stem(l) for l in l2]
    #going through every substring
    min_index = np.inf
    max_index = -1
    was_added = False
    for i in range(len(s2)-s+1):
        cur_l2 = s2[i:i+s]
        lp = min(len(s1),len(cur_l2))
        cur_l2 = [l for l in cur_l2 if l not in STOPWORDS]
        cur_l2 = [ps.stem(l) for l in cur_l2]
        ls = min(len(l1),len(cur_l2))
        d = edit_dist_help(l1,cur_l2,len(l1), len(cur_l2))
        #deciding
        added = False
        if (ls >=2 and d ==0):
            return STOP
        if (lp == 4 and ls ==4 and d<=1):
            added = True
        if (lp == 5 and ls >5 and d<=1):
            added = True
        if (lp ==6 and ls>=5 and d<=1):
            added = True
        if (lp > 6 and ls >3 and d<=2):
            added = True
        if (added): # should be added, thus we should append it's indices:
            if (not was_added):
                min_index = i
                was_added = True
            max_index = i+s
    if (was_added):
        return (True, " ".join(s2[min_index:max_index]))
    return False

def fuzzy_substring_dist(l1,l2):
    E = {}
    m = len(l1)
    n = len(l2)
    for i in range(m+1):
        E[(0,i)] = (0,None)
    for i in range(n+1):
        E[(i,0)] = (i,None)

    for i in range(1,n+1):
        for j in range(1,m+1):
            if (l1[j-1]==l2[i-1]):
                E[(i,j)] = (E[(i-1,j-1)][0], "0")
            else:
                edit_list = [E[(i-1,j-1)][0],E[(i,j-1)][0],E[(i-1,j)][0]]
                val = 1+np.min(edit_list)
                direction = str(np.argmin(edit_list))
                E[(i,j)] = (val, direction)

    #extracting result
    min_val = np.inf
    min_index = 0
    for i in range(1,m+1):
        # print(E[(n,i)])
        cur_val = E[(n,i)][0]
        if (cur_val <= min_val):
            min_val = cur_val
            min_index = i
    stop_last_index = min_index
    cur_val = min_val
    cur_list = [l1[min_index-1]]
    ind = n
    direction = E[(ind,min_index)][1]
    while(direction):
        if (direction == "0"):
            min_index -= 1
            ind -= 1
            if (ind != 0 and min_index != 0):
                cur_list = [l1[min_index-1]] + cur_list
        if (direction == "2"):
            ind -= 1
        if (direction == "1"):
            min_index -= 1
            cur_list = [l1[min_index-1]] + cur_list
        direction = E[(ind,min_index)][1]
    stop_first_index = min_index
    return cur_val, stop_first_index,stop_last_index-1, cur_list








if __name__ == "__main__":
    # s1 = "I don't know what to tell you brother there is no place like home i think i think tell me nir i think i think tell me nir i think i think tell me nir i think i think tell me nir i think i think tell me nir i think i think tell me nir i think i think tell me nir i think"
    # s2 = "there's no place like home tell me nir i think tell me nir i think i think tell me nir i think i think tell me nir i think i think tell me nir i think"
    s1 = "i love the smell of napalm in the morning"
    s2 = 'chlorine  don\'t know why but i love the smell of it'
    # print(decision_tree(s1,s2))
    # s1 = "T A T T G G C T A T A C G G T T"
    # s1 = re.sub(PUNCT_REG," ",s1)
    # s2 = "G C G T A T G C"
    # s2 = re.sub(PUNCT_REG," ",s2)
    print(fuzzy_decision_tree(s2,s1))
    print(fuzzy_substring_dist(s2.split(), s1.split()))