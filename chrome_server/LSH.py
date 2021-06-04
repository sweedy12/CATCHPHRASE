from datasketch import MinHash, MinHashLSH
import re as re
PUNCT_REG = "[.\"\\-,*)(!?#&%$@;:_~\^+=/]"
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))







def get_words_from_sentences(sentences):
    word_map = {}
    for sent in sentences:
        for w in sent.split():
            if (w not in STOPWORDS):
                word_map[w] = "1"
    return word_map


def create_minhashes(sentences, orig_sentences, lsh = None, index = 0, jump = 1000000):
    orig_sentences = [re.sub(PUNCT_REG," ", sent) for sent in orig_sentences]
    word_map = get_words_from_sentences(orig_sentences)
    lsh = MinHashLSH(threshold = 0.003, num_perm = 128)
    # lines = f.readlines()
    minhash = []
    i = 1
    for line in sentences:

        add_lsh = False
        for d in line.split():
            if d in word_map:
                add_lsh = True

        if (add_lsh):
            m = MinHash(num_perm=128)
            for d in line.split():
                m.update(d.encode('utf8'))
            minhash.append((line,m))
        if (i%10000 == 0):
            print(i)
        i += 1
    for i,tup in enumerate(minhash):
        line,m = tup
        if (i%10000 ==0 ):
            print("lshing " + str(i))
        try:
            lsh.insert(line,m)
        except:
            print(line)

    return minhash, lsh


def get_minhash_of_sentence(sent):
    """

    :param sent:
    :return:
    """
    sent = re.sub(PUNCT_REG," ",sent)
    m = MinHash()
    for d in sent.split():
        m.update(d.encode('utf8'))
    return (sent,m)


def get_query_for_minhash(lsh, minhash):
    try:
        lsh.insert(minhash[0],minhash[1])
    except:
        pass
    return lsh.query(minhash[1])




def create_lsh(minhash):
    lsh = MinHashLSH(threshold=0.8,num_perm=128)
    for key,m in minhash:
        lsh.insert(key,m)
    return lsh




def load_lsh(path):
    import pickle
    """

    :param path:
    :return:
    """
    with open (path,"rb") as f:
        lsh = pickle.loads(f.read())
        return lsh




import pickle
# nir = "i need to tell you boy that escalated quickly".split()
# m = MinHash()
# for d in nir:
#     m.update(d.encode("utf8"))
#
# minhash,lsh = create_minhashes("data.txt")
# lsh = create_lsh(minhash)
# lsh.insert("nir",m)
# print(lsh.query(minhash[0][1]))
# print(m.jaccard(minhash[0][1]))
# with open("pickly", "rb") as f:
#     lsh = pickle.loads(f.read())
#     print(lsh.query(m))
# a = pickle.dumps(lsh)
# with open ("pickly","wb") as f:
#     f.write(a)









    # m1, m2 = MinHash(), MinHash()
    # for d in data1:
    #     m1.update(d.encode('utf8'))
    # for d in data2:
    #     m2.update(d.encode('utf8'))
    # print("Estimated Jaccard for data1 and data2 is", m1.jaccard(m2))