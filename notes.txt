max seq len
108

max seq len sample
62

TODO:
keep batches variable
save the outputs in a config file
have a common data helpers file

In save_vectors only load the first of all words in batch

papers
-------
https://arxiv.org/pdf/1502.06922.pdf
https://www.cs.cornell.edu/home/cardie/papers/ozan-nips14drsv.pdf
https://cs.stanford.edu/~quocle/paragraph_vector.pdf
https://arxiv.org/pdf/1707.02377.pdf

https://arxiv.org/pdf/1703.03130.pdf

datasets
--------
sentiment
textual entailment
author profiling

http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools

MR : Movie review snippet sentiment on a five
star scale (Pang and Lee, 2005).
CR : Sentiment of sentences mined from customer
reviews (Hu and Liu, 2004).
SUBJ : Subjectivity of sentences from movie reviews
and plot summaries (Pang and Lee, 2004).
MPQA : Phrase level opinion polarity from
news data (Wiebe et al., 2005).
TREC : Fine grained question classification
sourced from TREC (Li and Roth, 2002).
SST : Binary phrase level sentiment classification
(Socher et al., 2013).
STS Benchmark : Semantic textual similarity
(STS) between sentence pairs scored by Pearson
correlation with human judgments (Cer et al.,
2017).
4
For the datasets MR, CR, and SUBJ, SST, and TREC we
use the preparation of the data provided by Conneau et al.
(2017).
WEAT : Word pairs from the psychology literature
on implicit association tests (IAT) that are
used to characterize model bias (Caliskan et al.,
2017).

IAT and WEAT measure
https://arxiv.org/pdf/1608.07187.pdf

https://github.com/UKPLab/arxiv2018-xling-sentence-embeddings/tree/master/data/AC