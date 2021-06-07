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


def longest_substring_with_replaces(X,Y,replaces):
    """
    find the longest words-sequence shared between X and Y, allowing replaces
    :param X: first list
    :param Y: second list
    :param replaces: number of allowed replaces
    :return:
    """
    max_val = 0
    n = len(X)
    m = len(Y)
    table = [[(0,0) for k in range(m+1)] for l in range(n+1)]
    for i in range(n+1):
        for j in range(m+1):
            if (i==0 or j==0):
                table[i][j] = (0,0)
            else:
                val, cur_replacements = table[i-1][j-1]
                cur_val = 0
                if (X[i-1] == Y[j-1]):
                    #checking if we should take the left or upward value:
                    table[i][j] = (val+1, cur_replacements)
                    cur_val  = val + 1
                elif (cur_replacements == replaces): #can't make any further replacements
                    #finding the first replaced cell:
                    if (replaces == 0):
                        table[i][j] = (0,0)
                    else:
                        br = cur_replacements
                        lbr = cur_replacements
                        one_val = val
                        bval = val
                        two_val = val
                        count = 1
                        while (br != 0):
                            lbr = br
                            one_val = bval
                            bval,br = table[i-count][j-count]
                            if (lbr == 2 and br ==1 ):
                                two_val = one_val
                            count += 1
                        #updating the current value:
                        cur_val = two_val-one_val + 1
                        table[i][j] = (cur_val, cur_replacements)
                else:
                    cur_val = val + 1
                    table[i][j] = (cur_val, cur_replacements+1)
                max_val = max(max_val, cur_val)
    return max_val

WILCARD_SYMBOL = "*"


def add_wildcard_words(lst1,lst2,replaces):
    """
    This method gets 2 lists: one containing words with wildcards, and the other contains all words.
    we replace each
    :param lst1:
    :param lst:
    :param replaces:
    :return:
    """
    new_X = []
    i = 0
    l = len(lst1)
    cur_replacs = 0
    first_encounter = True
    j = 0
    while (i < l):
        if (j >= len(lst2)):
            if (lst1[i] != "*"):
                new_X.append(lst1[i])
            i+=1
        elif lst1[i] != "*":
            new_X.append(lst1[i])
            i+= 1
        else:
            cur_replacs = 0
            #if theres nothing prior to the wildcard, add
            if (i == 0):
                while (cur_replacs < replaces and j < len(lst2) and (i>= l-1 or lst1[i+1] != lst2[j])):
                    cur_replacs += 1
                    new_X.append(lst2[j])
                    j+=1
                i+=1
            else:
                #checking if we can find the starting j:
                while (j < len(lst2) and lst2[j] != lst1[i-1]):
                    j+=1
                j+=1
                while (cur_replacs < replaces and j < len(lst2) and (i >=l-1 or lst1[i+1] != lst2[j])):
                    cur_replacs += 1
                    new_X.append(lst2[j])
                    j+=1
                i+=1


    return new_X



def file_to_test(fname):
    test_info = {"patterns":[], "sentences":[], "replaces":[],"result":[]}
    with open(fname) as f:
        line = f.readline()
        while(line):
            cur_info = line.split(",")
            test_info["patterns"].append(cur_info[0])
            test_info["sentences"].append(cur_info[1])
            test_info["replaces"].append(cur_info[2])
            test_info["result"].append(cur_info[3])
            line = f.readline()
    return test_info

def test_longest_substring(fname):
    test_info = file_to_test(fname)
    for i in range(len(test_info["patterns"])):
        cur_pattern = test_info["patterns"][i]
        cur_sentence = test_info["sentences"][i]
        cur_replaces = int(test_info["replaces"][i])
        expected_result = int(test_info["result"][i])

        cur_result = longest_substring_with_wildcards(cur_pattern.split(),cur_sentence.split(), cur_replaces)
        print(cur_pattern+"," + cur_sentence + " with "  + str(cur_replaces) + " replaces")
        print("expected result was " + str(expected_result) + ", and we got " + str(cur_result))
        print()


def remove_wildcard_words(lst1,lst2,replaces):
    """
    this method gets 2 lists: one containing words with wildcards, and the other contains all words.
    we replace all words that fall under the wildcard, up to *replaces times.
    :param lst1:
    :param lst2:
    :param replaces:
    :return:
    """
    new_list = []
    i = 0
    l = len(lst1)
    cur_replacs = 0
    first_encounter = True
    j = 0
    while (j < len(lst2)):
        if (i >= l):
            while (j< len(lst2)):
                new_list.append(lst2[j])
                j+=1
            break
        if (lst1[i] == "*" and first_encounter):
            cur_replacs = 0
            first_encounter = False
        if (lst1[i] == lst2[j]):
            new_list.append(lst2[j])
            i+=1
            j+=1
            first_encounter = True
        elif (lst1[i] != "*" ):
            new_list.append(lst2[j])
            i+=1
            j+=1
            first_encounter = True
        elif(i<l-1 and lst1[i+1]==lst2[j]):
            new_list.append(lst2[j])
            i+=2
            first_encounter = True
        elif (cur_replacs < replaces):
            if (i==l-1):
                while (j < len(lst2)):
                    new_list.append(lst2[j])
                    j+=1
            else:
                while (j < len(lst2) and lst1[i+1] != lst2[j] ):
                    new_list.append(lst2[j])
                    j+=1
                i+=1
                first_encounter = True
        else:
            cur_replacs += 1
            j+=1

    #now, removing wildcards from x:
    new_X = []
    for j in lst1:
        if (j != "*"):
            new_X.append(j)
    return new_list, new_X



def longest_common_sequence(X,Y):
    """
    this method dins the longest commob sequence between X and Y
    :param X: a list of strings
    :param Y: a list of strings
    :return:
    """
    max_val = 0
    n = len(X)
    m = len(Y)
    table = [[0 for k in range(m+1)] for l in range(n+1)]
    for i in range(n+1):
        for j in range(m+1):
            cur_val = 0
            if (i==0 or j==0):
                table[i][j] = 0
            elif (X[i-1]==Y[j-1]):
                cur_val = table[i-1][j-1] + 1
                table[i][j] = cur_val
            else:
                cur_val = max(table[i-1][j],table[i][j-1])
                table[i][j] = cur_val
        max_val = max(cur_val,max_val)
    return max_val




#---------------------------------------- wilcard s2s measures------------------------------


def longest_substring_with_wildcards(X, Y, wildcard_rep):
    """
    find the longest words-sequence shared between X and Y, allwing for *wildcard_rep replaces of each wildcard in
    X.
    :param X: first list
    :param Y: second list
    :param replaces: number of allowed replaces
    :return:
    """
    #removing wildcards
    X = add_wildcard_words(X,Y,wildcard_rep)
    max_val = 0
    n = len(X)
    m = len(Y)
    table = [[0 for _ in range(m+1)] for _ in range(n+1)]
    # highest_count = wildcard_rep
    for i in range(n+1):
        is_wildcard_new = False
        if (i>=1 and X[i-1]=="*"):
            is_wildcard_new = True
        for j in range(m+1):
            if (i==0 or j==0):
                table[i][j] = 0
            else:
                val = table[i - 1][j - 1]
                if (X[i-1] == Y[j-1]):
                    table[i][j] = val+1
                    cur_val = val + 1
                else:
                    cur_val = 0
                    table[i][j] = 0
                max_val = max(max_val, cur_val)
    return max_val

def fuzzy_substring_dist_with_wildcards(l1,l2):
    E = {} #E(i,j) is the minimum edit distance
    m = len(l1)
    n = len(l2)
    for i in range(m+1):
        E[(0,i)] = (0,None,0)
    for i in range(n+1):
        E[(i,0)] = (i,None,0)
    i = 1
    while(i < n+1):
        was_last = False
        for j in range(1,m+1):
            if (l1[j-1]==l2[i-1]):
                pattern_count = 0
                E[(i,j)] = (E[(i-1,j-1)][0], "0")

            elif (l2[i-1] == WILCARD_SYMBOL and j>=i):
                if (was_last):
                    edit_list = [E[(i - 1, j - 1)][0], E[(i, j - 1)][0], E[(i - 1, j)][0]]
                    val = np.min(edit_list)
                    direction = str(np.argmin(edit_list))
                    if (direction == "2"):
                        val +=1
                    E[(i, j)] = (val, direction)

                else:
                    was_last = True
                    E[(i,j)] = (E[i-1,j-1][0],"0")
            else:
                pattern_count = 0
                edit_list = [E[(i-1,j-1)][0],E[(i,j-1)][0],E[(i-1,j)][0]]
                val = 1+np.min(edit_list)
                direction = str(np.argmin(edit_list))
                E[(i,j)] = (val, direction)

        i+=1

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


def longest_common_sequence_with_wildcard(X,Y,replaces):
    """
        this method dins the longest common sequence between X and Y
        :param X: a list of strings
        :param Y: a list of strings
        :return:
        """
    max_val = 0
    highest_count = 0
    X = add_wildcard_words(X,Y,replaces)
    n = len(X)
    m = len(Y)
    table = [[0 for k in range(m + 1)] for l in range(n + 1)]
    for i in range(n + 1):
        if (i>=1 and X[i-1]=="*"):
            highest_count += replaces - 1
        for j in range(m + 1):
            cur_val = 0
            if (i == 0 or j == 0):
                table[i][j] = 0
            elif (X[i - 1] == Y[j - 1]):
                cur_val = table[i - 1][j - 1] + 1
                table[i][j] = cur_val

            else:
                cur_val = max(table[i - 1][j], table[i][j - 1])
                table[i][j] = cur_val
        max_val = max(cur_val, max_val)
    return max_val



if __name__ == "__main__":
    # s1 = "I don't know what to tell you brother there is no place like home i think i think tell me nir i think i think tell me nir i think i think tell me nir i think i think tell me nir i think i think tell me nir i think i think tell me nir i think i think tell me nir i think"
    # s2 = "there's no place like home tell me nir i think tell me nir i think i think tell me nir i think i think tell me nir i think i think tell me nir i think"
    # s1 = "i love the smell of napalm in the morning"
    # s2 = 'chlorine  don\'t know why but i love the smell of it'
    # print(decision_tree(s1,s2))
    # s1 = "T A T T G G C T A T A C G G T T"
    # s1 = re.sub(PUNCT_REG," ",s1)
    # s2 = "G C G T A T G C"
    # s2 = re.sub(PUNCT_REG," ",s2)
    # print(fuzzy_decision_tree(s2,s1))
    # print(fuzzy_substring_dist(s2.split(), s1.split()))
    # s2 = "i don't know men some men  jerks just like to watch the coo burn"
    # s1 = "some men just want to watch the world burn"
    # s1 = "to * or not to *"
    # s2 = "to take giant shit or not to take giant shit"
    # # print(longest_substring_with_replaces_with_wildcards(s1.split(),s2.split(),2,2))
    # # print(longest_substring_with_replaces(s1.split(),s2.split(),2))
    # print(longest_common_sequence_with_wildcard(s1.split(), s2.split(), 1))
    # print(longest_substring_with_replaces(s1.split(), s2.split(),3))
    # print(add_wildcard_words(s1.split(),s2.split(),3))
    # l1 = s1.split()
    # l2 = s2.split()
    # l2 = [ps.stem(l) for l in l2 if l not in STOPWORDS]
    # l1 = [ps.stem(l) for l in l1 if l not in STOPWORDS]
    # print(s1)
    # print(longest_substring_with_replaces(s2.split(),s1.split(),0))
    test_longest_substring("longest_substring_test")