from ClusterNode import ClusterNode
import  TreePlotter as tp


class TreeFromFile:

    def __init__(self, fname):
        self.create_tree_from_fname(fname)

    def create_tree_from_fname(self, fname):
        """

        :param fname:
        :return:
        """
        id_to_node = {}
        with open (fname, "r",encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                params = line.split(",")
                sent = params[0]
                depth = int(params[1])
                id = int(params[2])
                head_id = int(params[3])
                int_sent = params[4]
                times = int(params[5])
                if (head_id == -1):
                    head = None
                else:
                    head = id_to_node[head_id]
                cur_node = ClusterNode(head,sent,int_sent,depth,id)
                cur_node.set_times(times)
                id_to_node[id] = cur_node
                if (head):
                    head.add_child(cur_node)
                else:
                    self.root = cur_node
    def get_root(self):
        return self.root






if __name__ == "__main__":
    tree = TreeFromFile("C:\\Users\\sweed\\Desktop\\Masters\\Second\\Lab\\clusterer\\new_lsh_test2\\with great power comes great responsibility")
    tp.plot_tree(tree.root, "2013_great_power_depth2")