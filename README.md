# cluster_outlier

This repository contains the code and embeddings for Outlier Detection and Word Similarity estimation.

1. Outlier Detection:

  * The script to run the task is outlier_detection/prototype.py and it can be run on the [Camacho-Collados_Dataset](http://lcl.uniroma1.it/outlier-detection/).
  * A preprocessed version of the [Camacho-Collados_Dataset](http://lcl.uniroma1.it/outlier-detection/) can be found in preprocessed_datasets/Camacho-Collados_Dataset.txt.
  * Embeddings are filtered to contain only the required vectors, and they are provided in embeddings/

  * The basic command to run the script:

           python prototype.py --input $DIR_DATA --embedding $DIR_EMBEDDING

  * To learn about the options provided by the script, run the following command:

          python prototype.py --help

  * Tip : With minor changes, the script can be run on a preprocessed version of Blair datasets, also provided in the folder preprocessed_datasets/. The script requires to accomodate the varying number of items in each Blair cluster.

2. Word Similarity:

  * The script to run the task is word_sim/wordsim.py and it can be run on the [MEN](https://staff.fnwi.uva.nl/e.bruni/MEN), [SimLEX](https://www.cl.cam.ac.uk/~fh295/simlex.html), and [WordSIM353](http://alfonseca.org/eng/research/wordsim353.html) datasets.

  * The command to run the script:

           python wordsim.py --input $DIR_DATA --embedding $DIR_EMBEDDING

  * To learn about the options provided by the script, run the following command:

           python wordsim.py --help
