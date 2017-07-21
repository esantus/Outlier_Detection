from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from itertools import izip
import sys, re, os, time, logging, argparse, codecs, scipy, math
import numpy as np
from os import path

class Cluster:
    """
        Contains two lists:
            - member: words belonging to a cluster
            - outlier: one word that does not belong to the cluster

        It also has an informative method:
            - print_info(): public, to print information
    """
    
    def __init__(self, cluster_list):
        self._cluster_list = cluster_list
        self._member = self._cluster_list[:-1]
        self._outlier = self._cluster_list[-1:]
        
    def print_info(self):
        print "Members of the cluster:", len(self._member), "-", ", ".join(self._member).encode("utf8")
        print "Outliers in the cluster:", len(self._outlier), "-", ", ".join(self._outlier).encode("utf8")

    def get_cluster(self):
        return self._cluster_list

class Dataset:
    """
        Contains the dataset, which is a list of clusters.
        
        It also has two methods:
            - _load_dataset(): private, activated when declaring the object
            - print_dataset(): public, to print the dataset
    """
    
    def __init__(self, input_path):
        self._input_path = input_path
        self._num_clusters = 0
        self._dataset = self._load_dataset()
        
    def _load_dataset(self):
        
        ds = []
        num_no_full_recall = 0
        with codecs.open(self._input_path, "r", encoding='utf8') as dataset_f:
            cluster = []
            for line in dataset_f:
                line = line.strip()
                if line != "***" and line != "":
                    if CASED:
                        cluster.append(line)
                    else:
                        cluster.append(line.lower())
                elif line == "***":
                    if DEBUG == True:
                        print ",".join(cluster).encode("utf8")
                    if len(cluster) != 0:
                        self._num_clusters += 1
                        ds.append(Cluster(cluster))
                        cluster = []  
        return ds
            
    def print_dataset(self):
        i = 1
        for item in self._dataset:
            print i
            item.print_info()
            i += 1

    def get_dataset(self):
        return self._dataset

    def get_num_cluster(self):
        return self._num_clusters

class Embeddings:
  
    def __init__(self, emb_file):

        self._emb_file = emb_file
        self._str2idx = {}
        self._pretrained_embeddings = None

        self.load_embeddings()

    def load_embeddings(self):

        embeds = []
        cur_idx = 0
        with codecs.open(self._emb_file, encoding='utf8') as f:
            for line_num, line in enumerate(f):
                line = line.strip().split()
                if line:
                    try:
                        embeds.append(line[1:])
                        word = line[0]
                        self._str2idx[word] = cur_idx
                        cur_idx += 1
                    except:
                        raise ValueError('The embedding file is misformatted at line %d' % (line_num+1))

        self._pretrained_embeddings = np.array(embeds, dtype=np.float32)
        del embeds
        return

    def embedding_lookup(self, word):
        if self._str2idx.has_key(word):
            v = self._pretrained_embeddings[self._str2idx.get(word),:]
        else:
            print "\'%s\' is not found in pretrained embeddings"%word.encode("utf8")
            if "_" in word:
                temp = []
                for lex in word.split('_'):
                    if self._str2idx.has_key(word): 
                        temp.append(self._pretrained_embeddings[self._str2idx.get(word),:])
                        break
                if temp:
                    v = np.mean(np.array(temp))
                else:
                    v = np.array([0]*300)
            else:
                v = np.array([0]*300)
        return v

    def get_vocab(self):
        return self._str2idx

def APSyn(x_row, y_row):
    """
    APSyn(x, y) = (\sum_{f\epsilon N(f_{x})\bigcap N(f_{y})))} \frac{1}{(rank(f_{x})+rank(f_{y})/2)})
    :param x_row:
    :param y_row:
    :return: similarity score
    """

    # Sort y's contexts
    y_contexts_cols = sort_by_value_get_col(y_row) # tuples of (row, col, value)
    y_context_rank = { c : i + 1 for i, c in enumerate(y_contexts_cols) }

    # Sort x's contexts
    x_contexts_cols = sort_by_value_get_col(x_row)

    assert len(x_contexts_cols) == len(y_contexts_cols)

    x_context_rank = { c : i + 1 for i, c in enumerate(x_contexts_cols) }

    # Average of 1/(rank(w1)+rank(w2)/2) for every intersected feature among the top N contexts
    intersected_context = set(y_contexts_cols).intersection(set(x_contexts_cols))
    
    if formula == F_ORIGINAL:
        score = sum([2.0 / (x_context_rank[c] + y_context_rank[c]) for c in intersected_context]) #Original
    elif formula == F_POWER:
        score = sum([2.0 / (math.pow(x_context_rank[c], POWER) + math.pow(y_context_rank[c], POWER)) for c in intersected_context])
    elif formula == F_BASE_POWER:
        score = sum([math.pow(BASE, (x_context_rank[c]+y_context_rank[c])/2.0) for c in intersected_context])
    else:
        sys.exit('Formula value not found!')

    return score

def sort_by_value_get_col(mat):
    """
    Sort a sparse coo_matrix by values and returns the columns (the matrix has 1 row)
    :param mat: the matrix
    :return: a sorted list of tuples columns by descending values
    """

    sorted_tuples = sorted(mat, key=lambda x: x[2], reverse=True)

    if len(sorted_tuples) == 0:
        return []

    rows, columns, values = zip(*sorted_tuples)
    return columns

def detection(dataset, embeddings, avg_proto_vec):

    '''
    Read clusters one by one and perform detection under different schemes

    '''

    output = []
    op = []
    correct = 0

    for cluster in dataset.get_dataset():

        sim_list = []
        cluster = cluster.get_cluster()

        if len(cluster) == RANGE:

            for idx in xrange(RANGE):

                score = 0
                mem = list(cluster)
                outlier = mem.pop(idx)

                mem_vecs =[embeddings.embedding_lookup(word) for word in mem]
                outlier_vec = embeddings.embedding_lookup(outlier)
                outlier_vec_mat = [(1, i, v) for i, v in enumerate(outlier_vec)]
                
                if avg_proto_vec == RANGE-1:
                    kmeans = KMeans(n_clusters=1, random_state=0).fit(mem_vecs)
                    mem_avg_vec = kmeans.cluster_centers_[0]

                    if APSYN:
                        mem_avg_vec_mat = [(1, i, v) for i, v in enumerate(mem_avg_vec)]

                        score = APSyn(mem_avg_vec_mat, outlier_vec_mat)
                    else: 
                        score = cosine_similarity(mem_avg_vec.reshape(1,-1),outlier_vec.reshape(1,-1))
                elif avg_proto_vec == 1:
                    for m in mem_vecs:

                        m_mat = [(1, i, v) for i, v in enumerate(m)]

                        if APSYN:
                            score += APSyn(m_mat, outlier_vec_mat)
                        else:
                            score += cosine_similarity(m.reshape(1,-1),outlier_vec.reshape(1,-1))

                    score /= (RANGE-1)
                else:
                    sys.exit('avg_proto_vec value is NOT valid')
                    
                sim_list.append(score)

            output.append(np.argmin(sim_list))
            sim_mat = [(1, i, v) for i, v in enumerate(sim_list)]
            col = sort_by_value_get_col(sim_mat)
            op.append(col.index(RANGE-1))

    for i in output:
        if i == RANGE-1:
            correct += 1

    opp = sum(op)
    L.info('Total Number of clusters tested = %d'%dataset.get_num_cluster())
    opp = opp*100.0/(dataset.get_num_cluster()*(RANGE-1))
    return correct*1.0/len(output), opp

#--------------------------------------------------------------------#
#------------------------------- MAIN -------------------------------#
#--------------------------------------------------------------------#

if __name__ == "__main__":

    #-------------- Logging and Argparse  ----------------#
    program = os.path.basename(sys.argv[0])
    L = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    L.info("Running %s" % ' '.join(sys.argv))
    RANGE = 9
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action='store_true')
    p.add_argument("-c", "--cased", action='store_true')
    p.add_argument("-d", "--debug", action='store_true')
    p.add_argument("-s", "--apsyn", action='store_true')
    p.add_argument("-a", "--avg_proto_vec", dest='avg_proto_vec', type=int, default=RANGE-1)
    p.add_argument("-p", "--power", dest="power", help="power", type=float, default=0.10)
    p.add_argument("-b", "--base", dest="base", help="base", type=float, default=0.95)
    p.add_argument("-f", "--formula", dest="formula", help="formula", type=int, default=0)
    p.add_argument("-i", "--input", dest="input_path", help="input file path for dataset")
    p.add_argument("-e", "--embedding", dest="embedding", help="input file path for pretrained embeddings")
    p.add_argument("-o", "--output", dest="output_path", help="output file path", default=path.abspath('../../output/temp/temp.out'))
    args = p.parse_args()
    input_path = path.abspath(args.input_path)
    embedding_path = path.abspath(args.embedding)
    output_path = path.abspath(args.output_path)
    DEBUG = args.debug
    APSYN = args.apsyn
    APV = args.avg_proto_vec
    CASED = args.cased
    formula = args.formula
    POWER = args.power
    BASE = args.base
    F_ORIGINAL = 0
    F_POWER = 1
    F_BASE_POWER = 2
    #------------ End Logging and Argparse ---------------#

    my_dataset = Dataset(input_path)
    my_embeddings = Embeddings(embedding_path)

    result, opp = detection(my_dataset, my_embeddings, avg_proto_vec=APV)

    if APSYN:  
        L.info("Score given by --> APSyn")
        if formula == F_ORIGINAL:
            L.info('Fomula for APSyn --> Original')
        elif formula == F_POWER:
            L.info('Fomula for APSyn --> Power = %.2f'%POWER)
        elif formula == F_BASE_POWER:
            L.info('Fomula for APSyn --> Base Power = %.2f'%BASE)
        else:
            sys.exit('Formula value not found!')
    else:
        L.info("Score given by --> cosine_similarity")

    if APV == RANGE-1:
        L.info("Averaging member vectors as proto --> 1vs8")
    elif APV == 1:
        L.info("Averging outlier-member scores --> 1vs1")

    L.info('Accuracy = %.2f%%'%(result*100.0))
    L.info('OPP = %.2f'%opp)
    