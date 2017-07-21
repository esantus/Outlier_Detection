# cluster_outlier

This repository contains the code and embeddings for Outlier Detection and Word Similarity estimation.

1. Outlier Detection:

  * Code for the task is outlier_detection/prototype.py and it can be run on the [Camacho-Collados_Dataset](http://lcl.uniroma1.it/outlier-detection/).
  * A preprocessed version is preprocessed_datasets/Camacho-Collados_Dataset.txt.
  * Embeddings are filtered to contain only the required ones and provided in embeddings/

  * The basic command to run the script:

           python prototype.py --input $DIR_DATA --embedding $DIR_EMBEDDING

  * For options provided by the script, run the following for help:

          python prototype.py --help

  * Tip : With minor changes, the script can be run on preprocessed version of Blair datasets in the folder preprocessed_datasets/. The required modifications is to accomodate the varying number of items in each cluster.

2. Word Similarity:

  * Code for the task is word_sim/wordsim.py and it can be run on the [MEN](https://staff.fnwi.uva.nl/e.bruni/MEN), [SimLEX](https://www.cl.cam.ac.uk/~fh295/simlex.html), and [WordSIM353](http://alfonseca.org/eng/research/wordsim353.html) datasets.

  * The command to run the script:

           python wordsim.py --input $DIR_DATA --embedding $DIR_EMBEDDING

  * For options provided by the script, run the following for help:

           python wordsim.py --help
