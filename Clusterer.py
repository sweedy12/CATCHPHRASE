import numpy as np
import re
import nltk.data
import Measures
from ClusterNode import  ClusterNode
from AllClusters import AllClusters
import json
import os
import LSH

# nltk.download("punkt")

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def parse_file(fname):
    """

    :param fname:
    :return:
    """
    with open(fname,"r") as f:
        data = f.read()
        sentences = tokenizer.tokenize(data)
        sentences = [sent.lower() for sent in sentences]
        return sentences




PUNCT_REG = "[.\"\\-,*)(!?#&%$@;:_~\^+=/]"
def tree_add_checker(root,sent,id):
    """
    this method performs bfs to check whether we should add something
    :param root: The root of the tree.
    :param sent: The sentence to be added .
    :return:
    """
    sent = sent.lower()
    nodes = [root]
    # sent = re.sub(PUNCT_REG," ",sent)
    while (nodes):
        cur_node = nodes.pop()
        cur_sent = cur_node.get_int_sent().lower()
        full_sent = cur_node.get_full_sent()
        if (cur_sent == sent):
            return False

        res = Measures.fuzzy_decision_tree(sent,cur_sent)
        if (res):
            if (res == Measures.STOP):
                cur_node.increase_times()
                return False
            int_sent = res
            if (id==1):
                stop = 1
            #add the sentence to the cluster
            new_node = ClusterNode(cur_node,sent,int_sent,cur_node.get_depth()+1, id+1)
            cur_node.add_child(new_node)
            return True
        else:
            if (cur_node.get_depth() < 0):
                children = cur_node.get_children()
                if (children):
                    nodes.extend(cur_node.get_children())
    return False


def expand_clusters_with_sent(sent,all_clusters,id):
    """

    :param sent:
    :param id:
    :return:
    """
    for root in all_clusters.get_clusters():
        if (tree_add_checker(root,sent,id)):
            id +=1
            break
    return id



def clean_str_from_digits(str):
    m = re.search(r"\d(?!\d)",str)
    if m:
        str = str[m.start()+1:]
    str = str[::-1]
    m = re.search(r"\d(?!\d)",str)
    if m:
        str = str[m.start()+1:]
    return str[::-1]

def expand_clusters_lsh(orig_file, lsh_path, id = 0):
    orig_sentences = parse_file(orig_file)
    orig_sentences =[ re.sub(PUNCT_REG,"",sent) for sent in orig_sentences]
    all_clusters = AllClusters(orig_sentences)
    sent_to_cluster = all_clusters.get_sent_to_cluster()
    #going through every original sentence lsh clustering file:
    try:
        os.mkdir("1503_depth1")
    except:
        pass
    for sent in orig_sentences:
        print(sent)
        # sent = re.sub(PUNCT_REG,"",sent)
        with open(lsh_path+"\\"+sent,"r", encoding="utf8") as f:
            cur_cluster = sent_to_cluster[sent]
            line = f.readline()
            line = line.lower()
            i = 1
            while(line):
                #clean line from numbers:
                if (line[:5] == "49522"):
                    stop = 1
                line = clean_str_from_digits(line)
                id = expand_cluster_with_sent(cur_cluster,line,id)
                if (i%10000 == 0):
                    print(i)
                #     os.mkdir(sent+"_"+str(i))x`
                #     cur_cluster.write_tree_to_file("trial\\"+sent)
                i+=1
                line = f.readline()
        cur_cluster.write_tree_to_file("1503_depth1\\"+sent)
    with open("id.txt","w") as f:
        f.write(str(id))
    return all_clusters
def expand_cluster_with_sent(root,sent,id):
    if (tree_add_checker(root,sent,id)):
            id +=1
    return id

def expand_clusters(orig_file, data_path, id = 0):
    """

    :param orig_file:
    :param data_file:
    :param id:
    :return:
    """
    #reading the original sentences:
    orig_sentences = parse_file(orig_file)
    all_clusters = AllClusters(orig_sentences)
    #reading the dats sentences:
    with open (data_path,encoding="utf-8") as df:
        line = df.readline()
        i =0
        #expanding clusters with sentences:
        while(line):
            line = line.replace("\n","")
            if (i%1000 == 0):
                print(i)
            i+=1
            if (i%200001 == 0):
                os.mkdir(str(i-1))
                all_clusters.write_all_clusters(str(i-1))

            id = expand_clusters_with_sent(line,all_clusters,id)
            line = df.readline()
    with open("id.txt","w") as f:
        f.write(str(id))
    return all_clusters

def tokenize_data_file(data_path):
    with open(data_path+"_tokenized", "w", encoding="utf-8") as df:
        with open(data_path, "r") as jf:
            line = jf.readline()
            while (line):
                cur_line = json.loads(line)
                cur_tokenized = tokenizer.tokenize(cur_line["body"])
                for sent in cur_tokenized:
                    try:
                        df.write(sent + "\n")
                    except:
                        df.write(sent+"\n")
                line = jf.readline()


import pickle

def write_lsh_to_file(orig_path,lsh,dir):
    """

    :param orig_path:
    :param lsh_path:
    :param dir:
    :return:
    """
    original_sentences = parse_file(orig_path)
    # lsh = LSH.load_lsh(lsh_path)
    for sent in original_sentences:
        print("trying for sentence " +sent)
        sent = re.sub(PUNCT_REG,"",sent)
        with open(dir+"\\"+sent,"a", encoding = "utf8") as f:
            m = LSH.get_minhash_of_sentence(sent)
            query = LSH.get_query_for_minhash(lsh,m)
            for q in query:
                f.write(q + "\n")

sentnir = "495225He needs to come up with some gesture of truly revolting subservience to ingratiate himself to jews if he wants to put himself on the fast track  for goyim anyway  1000000"
import os
if __name__ == "__main__":
    dir = "C:\\Users\\sweed\\Desktop\\Masters\\Second\\Lab\\clusterer\\"
    # print(sentnir[:5])
    orig_name = "orig_sentences2"
    # orig_sent = parse_file(orig_name)
    # all_clusters = AllClusters(orig_sent)
    # print(clean_str_from_digits(sentnir))
      # sent = "538808However long it takes for Coach Cal to realize the those nba caliber players he recruits play exactly like nba players  without the ability to shoot free throws 1000000"
    # cluster = all_clusters.get_sent_to_cluster()[orig_sent[0]]
    # c1 = ClusterNode(cluster,sent,1,1)
    # c2 = ClusterNode(c1,sent,1,1)
    # cluster.add_child(c1)
    # c1.add_child(c2)
    # expand_cluster_with_sent(cluster,sent,1)
    # nir = 1
    expand_clusters_lsh(orig_name,"lsh_1503")
    # str  = "440620This fact alone is enough to show that the concept of God is man made 1000000"
    # print(clean_str_from_digits(str))
    # data_path = dir + "RC_2011-04_tokenized"
    # data_path = "data.txt"
    # data_path = "RC_2011-04"
    # tokenize_data_file(data_path)
    # data_sentences = parse_file(data_path + "_tokenized")
    # nir = 1
    # all_clusters = expand_clusters(orig_name,data_path)
    # all_clusters.write_all_clusters("wednesday_test")
    # with open(data_path,encoding=("utf8")) as f:
    #     print(len(f.readlines()))
    # index = 0
    # with open("lsh","rb") as f1:
    #     lsh = pickle.loads(f1.read())
    # lsh = None
    # orig_sentences = parse_file(orig_name)
    # try:
    #     os.mkdir("lsh_1503_2")
    # except:
    #     pass
    # data_path = "RC_2011-04_tokenized"
    # index = 0000000
    # for i in range(100):
    #     minhash,lsh = LSH.create_minhashes(data_path, sentences= orig_sentences,index=index, lsh=None, jump= 5000000)
    #     index += 5000000
    # #     # lsh = LSH.create_lsh(minhash)
    # #     # with open("lsh","wb") as f:
    # #     #     d = pickle.dumps(lsh)
    # #     #     f.write(d)
    #     print("done with index "+ str(index))
    #     print("-------------------------------------------")
    #     write_lsh_to_file(orig_name,lsh,dir+"lsh_1503_2")
    #     print("done with index " + str(index))
    #     print("-------------------------------------------------------")







