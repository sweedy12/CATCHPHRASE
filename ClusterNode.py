import re
class ClusterNode:


    def __init__(self, head,full_sent,int_sent,depth,id):
        self.head = head
        self.full_sent = full_sent
        self.full_sent = re.sub("\n"," ",self.full_sent)
        self.int_sent = int_sent
        self.children = []
        self.depth = depth
        self.id = id
        self.times = 1



    def increase_times(self):
        self.times += 1
    def get_int_sent(self):
        return self.int_sent

    def get_head(self):
        return self.head

    def get_full_sent(self):
        return self.full_sent
    def add_child(self,child):
        self.children.append(child)
    def get_depth(self):
        return self.depth

    def get_id(self):
        return self.id

    def get_children(self):
        return self.children

    def set_times(self, times):
        self.times = times

    def get_times(self):
        return self.times


    def write_tree_to_file(self,fname):
        """

        :param fname:
        :return:
        """
        with open(fname, "w",encoding="utf-8") as f:
            nodes = [self]
            while(nodes):
                cur_node = nodes.pop()
                f.write(cur_node.get_full_sent()+","+str(cur_node.get_depth())+","+str(cur_node.get_id()))
                if (not cur_node.get_head()):
                    head_id = -1
                else:
                    head_id = cur_node.get_head().get_id()
                f.write(","+str(head_id)+"," + cur_node.get_int_sent()+","+str(cur_node.times)+"\n")
                children = cur_node.get_children()
                if (children):
                    nodes.extend(children)




