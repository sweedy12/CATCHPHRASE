from ClusterNode import ClusterNode
import re
PUNCT_REG = "[.\"\\-,*)(!?#&%$@;:_~\^+=/]"

class AllClusters:

    def __init__(self, original_sentences):
        self.clusters = []
        self.original_sentences = original_sentences
        self.sent_to_cluster = {}
        self.create_initial_clusters()


    def create_initial_clusters(self):
        for sent in self.original_sentences:
            cur_cluster = ClusterNode(None, sent,sent, 0, 0)
            self.clusters.append(cur_cluster)
            self.sent_to_cluster[sent] = cur_cluster

    def get_clusters(self):
        return self.clusters



    def get_sent_to_cluster(self):
        return self.sent_to_cluster


    def write_all_clusters(self,dir):
        """

        :param dir:
        :return:
        """
        for root in self.clusters:
            cur_fname = dir+"\\"+root.get_full_sent()
            root.write_tree_to_file(cur_fname)

