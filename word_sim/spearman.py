from scipy import stats
import sys

col1, col2 = sys.argv[2].split(",")
col1, col2 = int(col1), int(col2)
list1, list2 = [], []

with open(sys.argv[1], "rb") as corpus:
	for line in corpus:
	
		l = line.split("\t")
		c1, c2 = float(l[col1 - 1]), float(l[col2 - 1])
		list1.append(c1)
		list2.append(c2)
		
print stats.spearmanr(list1, list2)
	
	