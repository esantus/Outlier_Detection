from sklearn.metrics.pairwise import cosine_similarity
from itertools import izip
import sys, re, os, time, logging, argparse, codecs, scipy, math
import numpy as np
from os import path

class Dataset:
    """
        Contains the dataset, which is a list of clusters.
        
        It also has two methods:
            - _load_dataset(): private, activated when declaring the object
            - print_dataset(): public, to print the dataset
    """
    
    def __init__(self, input_path):
        self._input_path = input_path
        self._dataset = self._load_dataset(self._input_path)
        
    def _load_dataset(self, file_name):
        
        ds = []
        pair = []
        with codecs.open(file_name, "r", encoding='utf8') as dataset_f:
            cluster = []
            for line in dataset_f:
                line = line.strip()
                for item in line.split():
                    pair.append(item.split('-')[0].lower())

                ds.append(pair)
                pair = []

        return ds
            
    def print_dataset(self):
        i = 1
        for item in self._dataset:
            print i
            item.print_info()
            i += 1

    def get_dataset(self):
        return self._dataset

class Embeddings:
  
    def __init__(self, emb_file):

        self._emb_file = emb_file
        self._str2idx = {}
        self._pretrained_embeddings = None

        self.load_embeddings()

    def load_embeddings(self):

        embeds = []
        cur_idx = 0
        with open(self._emb_file) as f:
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
            return v
        else:
            print "\'%s\' is not found in pretrained embeddings"%word.encode("utf8")


def APSyn(x_row, y_row):
    """
    APSyn(x, y) = (\sum_{f\epsilon N(f_{x})\bigcap N(f_{y})))} \frac{1}{(rank(f_{x})+rank(f_{y})/2)})
    :param x_row:
    :param y_row:
    :return:
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

    # if formula == F_ORIGINAL:
    score_original = sum([2.0 / (x_context_rank[c] + y_context_rank[c]) for c in intersected_context]) #Original

    if formula == F_POWER:
        score_power = sum([2.0 / (math.pow(x_context_rank[c], POWER) + math.pow(y_context_rank[c], POWER)) for c in intersected_context])
    elif formula == F_BASE_POWER:
        score_power = sum([math.pow(BASE, (x_context_rank[c]+y_context_rank[c])/2.0) for c in intersected_context])
    else:
        score_power = score_original
        # sys.exit('Formula value not found!')

    return score_original, score_power

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

def detection(dataset, embeddings):

    output = []
    correct = 0

    result = []

    for pair in dataset.get_dataset():


        v1 = embeddings.embedding_lookup(pair[0].lower())
        v2 = embeddings.embedding_lookup(pair[1].lower())
            
        if APSYN:
            v1_mat = [(1, i, v) for i, v in enumerate(v1)]
            v2_mat = [(1, i, v) for i, v in enumerate(v2)]

            score_original, score_power = APSyn(v1_mat, v2_mat)
            pair.append(str(score_original))
            pair.append(str(score_power))
        else: 
            score = cosine_similarity(v1.reshape(1,-1),v2.reshape(1,-1))[0][0]
            pair.append(str(score))
        
        result.append(pair)

    return result

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
    p = argparse.ArgumentParser()
    p.add_argument("-v", "--verbose", action='store_true')
    p.add_argument("-d", "--debug", action='store_true')
    p.add_argument("-a", "--apsyn", action='store_true')
    p.add_argument("-i", "--input", dest="input_path", help="input file path for dataset")
    p.add_argument("-e", "--embedding", dest="embedding", help="input file path for pretrained embeddings")
    p.add_argument("-o", "--output", dest="output_path", help="output file path", default=path.abspath('output'))
    p.add_argument("-f", "--formula", dest="formula", help="formula", type=int, default=0)
    p.add_argument("-b", "--base", dest="base", help="base", type=float, default=0.95)
    p.add_argument("-p", "--power", default=0.05, type=float)
    args = p.parse_args()
    input_path = path.abspath(args.input_path)
    embedding_path = path.abspath(args.embedding)
    output_path = path.abspath(args.output_path)
    DEBUG = args.debug
    APSYN = args.apsyn
    BASE = args.base
    POWER = args.power
    formula = args.formula
    F_ORIGINAL = 0
    F_POWER = 1
    F_BASE_POWER = 2
    #------------ End Logging and Argparse ---------------#

    my_dataset = Dataset(input_path)
    my_embeddings = Embeddings(embedding_path)

    result = detection(my_dataset, my_embeddings)

    if APSYN:

        if formula == F_ORIGINAL:
            L.info('Fomula for APSyn --> Original')
            out_file = path.join(output_path, path.basename(input_path) + '.apsyn')
        elif formula == F_POWER:
            L.info('Fomula for APSyn --> Power = %.2f'%POWER)
            out_file = path.join(output_path, path.basename(input_path) + '.apsynPower-%.2f'%POWER)
        elif formula == F_BASE_POWER:
            L.info('Fomula for APSyn --> Base Power = %.2f'%BASE)
            out_file = path.join(output_path, path.basename(input_path) + '.apsynBasePower-%.2f'%BASE)
        else:
            sys.exit('Formula value not found!')
    else:
        L.info('Fomula for APSyn --> Original')
        out_file = path.join(output_path, path.basename(input_path) + '.cosine')

    with open(out_file, 'w+') as fout:
        for pair in result:
            fout.write('\t'.join(pair)+'\n')
    
