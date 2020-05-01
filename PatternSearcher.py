WILDCARD = "(\\*|(\\s*\\w+\\s*))"
# WILDCARD = ".+"
WILDCARD_SYMBOL = "*"
WORD_OR_SPACE = "(\\w|\\s)*"



class Option:

    def __init__(self,prev_opt, pattern, swithced):
        self.prev_opt = prev_opt
        self.pattern_list = pattern
        self.switched = swithced

    def has_switched(self,i):
        return i in self.switched


    def get_pattern_list(self):
        return self.pattern_list
    def get_prev_opt(self):
        return self.prev_opt

    def get_switched(self):
        return self.switched

    def switch_place(self,i):
        self.pattern_list[i] = WORD_OR_SPACE
        self.switched.append(i)

    def get_pattern_str(self):
        new_pattern_list = []
        cur_switched = []
        last_l = ""
        for i in range(len(self.pattern_list)):
            if (self.pattern_list[i] == WORD_OR_SPACE):
                if (last_l != WORD_OR_SPACE):
                    last_l = WORD_OR_SPACE
                    new_pattern_list.append(self.pattern_list[i])
                    cur_switched.append(i)
            else:
                new_pattern_list.append(self.pattern_list[i])
                last_l=  self.pattern_list[i]

        pat = " ".join(new_pattern_list)
        pat = pat.replace(WORD_OR_SPACE+" ",WORD_OR_SPACE)
        return pat

class ReverseOption(Option):
    def __init__(self,prev_opt, pattern, swithced,orig_sent_list):
        super(ReverseOption, self).__init__(prev_opt, pattern, swithced)
        self.orig_sent_list = orig_sent_list

    def switch_place(self,i):
        self.pattern_list[i] = self.orig_sent_list[i]
        self.switched.append(i)


import TreeFromFile as TFF

class BeamSearchDown:

    def __init__(self, beam_size,tree_path,cutoff_ratio):
        self.beam_size = beam_size
        self.root = TFF.TreeFromFile(tree_path)
        self.get_sentences()
        self.start_sent = self.root.get_root().get_int_sent()
        self.sent_list = self.start_sent.split()
        self.n = len(self.sent_list)
        self.cutoff_ratio = cutoff_ratio
        self.count_down()

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

    def get_starting_options(self):
        options = []
        freqs = []
        start_sent_list = self.start_sent.split()
        for i in range(len(start_sent_list)):
            cur_list = start_sent_list[:]
            cur_list[i] = WORD_OR_SPACE
            pattern_str = " ".join(cur_list)
            freq = self.count_pattern(pattern_str)
            options.append(self.create_option(None,cur_list, [i]))
            freqs.append(freq)
        return options,freqs



    def create_option(self,option,pattern,switched):
        return Option(option,pattern[:],switched[:])


    def count_down(self):
        self.n_to_options = {}
        first_options,start_freqs = self.get_starting_options()
        first_dict = {}
        try:
            for i,option in enumerate(first_options):
                first_dict[option] = start_freqs[i]

        except:
            first_dict[first_options] = start_freqs
        self.n_to_options[0] = first_dict
        self.final_level = self.n-2
        for i in range(1,self.n-1):
            cur_options = list(self.n_to_options[i-1].keys())
            cur_dict = self.check_current_patterns(cur_options,self.n_to_options[i-1])
            if (cur_dict):
                self.n_to_options[i] = cur_dict
            else:
                print("no soup for you")
                print(i)
                self. final_level = i-1
                break
        self.paths = []
        for val in self.n_to_options[self.final_level]:
            self.paths.append(self.recover_path(val))



    def recover_path(self,start_option):
        cur_option = start_option
        path_list = []
        for i in range(self.final_level,-1,-1):
            cur_freq = self.n_to_options[i][cur_option]
            path_list.append((cur_option.get_pattern_str(),cur_freq))
            cur_option = cur_option.get_prev_opt()
        return path_list

    def print_paths(self):
        for path in self.paths:
            print("startin pattern path:")
            for tup in path:
                print("pattern " + tup[0] +" (freq " + str(tup[1])+" ) <---", end=" ")
            print()
            print()

    def write_paths_to_file(self,fname):
        with open (fname,"w", encoding="utf-8") as f:
            for path in self.paths:
                f.write("startin pattern path:\n")
                for tup in list(reversed(path)):
                    f.write("pattern " + tup[0] +" (freq " + str(tup[1])+" ) <---\n")
                f.write("\n")
                f.write("\n")


    def count_pattern(self, pattern):
        import re
        count = 0
        for i,sent in enumerate(self.sentences):
            if re.match(pattern,sent):
                count += self.appearances[i]
        return count / self.total_appearances

    def check_current_patterns(self,options,last_options_to_freq):
        options_to_freq = {}
        freq_to_option = {}
        min_freq = 0
        added = 0
        patterns_checked = []
        for option in options:
            pattern = option.get_pattern_list()
            switched = option.get_switched()
            last_freq = last_options_to_freq[option]
            for i in range(len(pattern)):
                if (not option.has_switched(i)):
                    cur_option = self.create_option(option,pattern[:],switched[:])
                    cur_option.switch_place(i)
                    pat = cur_option.get_pattern_str()
                    if (pat not in patterns_checked):
                        patterns_checked.append(pat)
                        cur_freq = self.count_pattern(pat)
                        if (self.did_pass_cutoff(cur_freq,last_freq) and added < self.beam_size):
                            added += 1
                            options_to_freq[cur_option] = cur_freq
                            if (cur_freq not in freq_to_option):
                                freq_to_option[cur_freq] = []
                            freq_to_option[cur_freq].append(cur_option)
                            if (added == 1):
                                min_freq = cur_freq
                            else:
                                min_freq = min(cur_freq,min_freq)
                        elif (self.did_pass_cutoff(cur_freq,last_freq) and cur_freq >= min_freq):
                            del_opt = freq_to_option[min_freq][0]
                            if (len(freq_to_option[min_freq])==1):
                                del freq_to_option[min_freq]
                            else:
                                del freq_to_option[min_freq][0]
                            del options_to_freq[del_opt]
                            min_freq = cur_freq
                            options_to_freq[cur_option] = cur_freq
                            if (cur_freq not in freq_to_option):
                                freq_to_option[cur_freq] =[]
                            freq_to_option[cur_freq].append(cur_option)
        return options_to_freq

    def did_pass_cutoff(self,current_freq,last_freq):
        if (last_freq > 0 and current_freq / last_freq >= self.cutoff_ratio):
            return True
        return False




class BeamSearchUp(BeamSearchDown):

    def create_option(self,option,pattern,switched):
        return ReverseOption(option,pattern,switched,self.sent_list)

    def get_starting_options(self):
        pattern_list = [WORD_OR_SPACE]*len(self.sent_list)
        return ReverseOption(None,pattern_list,[],self.sent_list),1.

    def did_pass_cutoff(self,current_freq,last_freq):
        if (current_freq > 0 and last_freq / current_freq <= self.cutoff_ratio):
            return True
        return False





class Pattern:

    def __init__(self,pattern_list,switched):
        self.pattern_list = pattern_list
        self.switched = switched

    def has_switched(self,i):
        return i in self.switched

    def get_pattern_str(self):
        new_pattern_list = []
        cur_switched = []
        last_l = ""
        for i in range(len(self.pattern_list)):
            if (self.pattern_list[i] == WORD_OR_SPACE):
                if (last_l != WORD_OR_SPACE):
                    last_l = WORD_OR_SPACE
                    new_pattern_list.append(self.pattern_list[i])
                    cur_switched.append(i)
            else:
                new_pattern_list.append(self.pattern_list[i])
                last_l=  self.pattern_list[i]

        pat = " ".join(new_pattern_list)
        pat = pat.replace(WORD_OR_SPACE+" ",WORD_OR_SPACE)
        return pat

    def get_pattern_list(self):
        return self.pattern_list

    def get_switched(self):
        return self.switched








class GreedyPathSearcherDown:
    def __init__(self,tree_path,cutoff_ratio):
        self.cutoff_ratio = cutoff_ratio
        self.root = TFF.TreeFromFile(tree_path)
        self.get_sentences()
        self.start_sent = self.root.get_root().get_int_sent()
        self.sent_list = self.start_sent.split()
        self.n = len(self.sent_list)
        # self.countPaths()


    def countPaths(self):

        self.options = self.get_starting_options()
        # self.last_levels = [self.n-1]*len(self.options)
        to_check =[True]*len(self.options)
        for i in range(self.n-1):
            print(i)
            for t,option in enumerate(self.options):
                if (to_check[t]):
                    max_freq = 0
                    max_pattern = None
                    cur_pattern = option[i][0]
                    last_freq = option[i][1]
                    swithced = cur_pattern.get_switched()
                    pattern_list = cur_pattern.get_pattern_list()
                    for j in range(self.n):
                        if (not cur_pattern.has_switched(j)):
                            cur_pattern_list = pattern_list[:]
                            cur_switched = swithced[:]
                            cur_switched.append(j)
                            cur_pattern_list = self.switch_word(cur_pattern_list,j)
                            cur_pattern = Pattern(cur_pattern_list,cur_switched)
                            pattern_str = cur_pattern.get_pattern_str()
                            cur_freq = self.count_pattern(pattern_str)
                            if (self.did_pass_cutoff(cur_freq,last_freq)):
                                if (cur_freq > max_freq):
                                    max_freq = cur_freq
                                    max_pattern = cur_pattern
                    if (max_pattern):
                        option.append((max_pattern,max_freq))
                    else:
                        to_check[t] = False
                        # self.last_levels[t] = i-1


    def did_pass_cutoff(self,current_freq,last_freq):
        if (last_freq > 0 and current_freq / last_freq >= self.cutoff_ratio):
            return True
        return False


    def write_paths_to_file(self,fname):
        with open(fname,"w",encoding="utf-8") as f:
            last_found_options = []
            for i,option in enumerate(self.options):
                f.write("Current path:\n")
                for tup in option:
                    f.write(tup[0].get_pattern_str() + "," +str(tup[1]) +"-->\n")
                f.write("\n")
                # last_tup = option[self.last_levels[i]]
                last_found_options.append(option[len(option)-1])
            #sorting the last level:
            last_found_options.sort(key = lambda x:x[1])
            f.write("\n\n")
            f.write("The last level options are:\n")
            for tup in reversed(last_found_options):
                f.write(tup[0].get_pattern_str() + "," +str(tup[1])+"\n")




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



    def switch_word(self,pattern_list,i):
        pattern_list[i] = WORD_OR_SPACE
        return pattern_list


    def get_starting_options(self):
        options = []
        for i in range(self.n):
            cur_list = self.sent_list[:]
            cur_list = self.switch_word(cur_list,i)
            cur_pattern = Pattern(cur_list,[i])
            pattern_str = cur_pattern.get_pattern_str()
            cur_freq = self.count_pattern(pattern_str)
            options.append([(cur_pattern,cur_freq)])
        return options

    def count_pattern(self, pattern):
        import re
        count = 0
        for i,sent in enumerate(self.sentences):
            if re.match(pattern,sent):
                count += self.appearances[i]
        return count / self.total_appearances


class GreedyPathSearcherUp(GreedyPathSearcherDown):

    def __init__(self,tree_path,cutoff_ratio):
        super(GreedyPathSearcherUp, self).__init__(tree_path,cutoff_ratio)
        self.orig_sent_list = self.sent_list[:]
        self.sent_list = [WORD_OR_SPACE]*self.n

    def switch_word(self,pattern_list,i):
        pattern_list[i] = self.orig_sent_list[i]
        return pattern_list

    def did_pass_cutoff(self,current_freq,last_freq):
        if (current_freq > 0 and last_freq / current_freq <= self.cutoff_ratio):
            return True
        return False



import os
# save_dir = "beam_search_up2_patterns"
save_dir = "sliced_greedy_search_down"
try:
    os.mkdir(save_dir)
except:
    pass
path = "2304_depth1"
n = 3
pattern_n = 3
max_n = 5
beam_size = 5
level = 5
cutoff_ratio = 1.2
for d in os.listdir(path):
    # if (d== "some men just want to watch the world burn"):
    print("---------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------")
    print("Patterns for " + d)
    new_path = path + "\\" + d
    # pf = BeamSearchUp(beam_size,new_path,cutoff_ratio)
    pf = GreedyPathSearcherDown(new_path,2)
    pf.countPaths()
    pf.write_paths_to_file(save_dir +"\\"+d)
    print("---------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------")