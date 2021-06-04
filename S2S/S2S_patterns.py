import numpy as np



def read_tags_sentences(fname):
    with open(fname) as f:
        sentence_to_tags = {}
        l = f.readline()
        while (l):
            if l != "":
                inf = l.split(",")
                sentence_to_tags[inf[0]] = inf[1].split()
            l = f.readline()
        return sentence_to_tags



def print_all_wildcard_agreement(all_dicts):
    per_sentence = []
    total_agreed = 0
    total_count = 0
    ref = all_dicts[0]
    for sent in ref:
        tag1 = all_dicts[0][sent]
        tag2 = all_dicts[1][sent]
        agreed_list = [1 if tag1[i] == tag2[i] and tag1[i]== "*" else 0 for i in range(len(tag1))]
        total_list = [1 if tag1[i] == "*" or tag2[i] == "*" else 0 for i in range(len(tag1))]
        cur_agreed = sum(agreed_list)
        cur_total = sum(total_list)
        if (cur_total != 0):
            per_sentence.append(cur_agreed/cur_total)
            total_agreed += cur_agreed
            total_count += cur_total

    print(per_sentence)
    print(total_agreed / total_count)
    print(sum([1 if per_sentence[i] == 1 else 0 for i in range(len(per_sentence))]))
    print(sum([1 if per_sentence[i] == 1 else 0 for i in range(len(per_sentence))]) / len(per_sentence))


def print_all_agreement(all_dicts):
    per_sentence = []
    total_agreed = 0
    total_count = 0
    ref = all_dicts[0]
    for sent in ref:
        tag1 = all_dicts[0][sent]
        tag2 = all_dicts[1][sent]
        agreed_list = [1 if tag1[i] == tag2[i]  else 0 for i in range(len(tag1))]
        cur_agreed = sum(agreed_list)
        cur_total = len(tag1)
        if (cur_total != 0):
            per_sentence.append(cur_agreed/cur_total)
            if (cur_agreed / cur_total == 1):
                print(" ".join(tag1))
            total_agreed += cur_agreed
            total_count += cur_total

    print(per_sentence)
    print(total_agreed / total_count)
    print(sum([1 if per_sentence[i] == 1 else 0 for i in range(len(per_sentence))]))
    print(sum([1 if per_sentence[i] == 1 else 0 for i in range(len(per_sentence))]) / len(per_sentence))



names = ["nir","salz"]
nonmeme_dir = "C:\\Users\\User\\Desktop\\nir\\Desktop\\Masters\\Second\\Lab\\nonmeme_patterns_test\\nonmeme_tags_"

all_dicts = []

for name in names:
    all_dicts.append(read_tags_sentences(nonmeme_dir+name))

print_all_agreement(all_dicts)
