import _pickle as cPickle
import numpy as np


translate, _ = cPickle.load(open('save/vocab_cotra.pkl', 'rb'))
positive_examples = []
positive_file = 'save/realtrain_cotra.txt'
with open(positive_file)as fin:
    for line in fin:
        line = line.strip()
        line = line.split()
        parse_line = [int(x) for x in line]
        positive_examples.append(parse_line)
sentences = np.array(positive_examples)

x = np.bincount(sentences[:,0].astype(int)) / 80000
for i in range(len(x)):
    if x[i] > 0.01:
        print(i, x[i], translate[i])
print(translate[38], translate[1001], translate[1948],translate[2088],translate[2195],translate[4359])

