import itertools
import time
import numpy as np
from tqdm import tqdm
import nltk
import string
def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    queries = np.unique(y[:,1])
    arrays =[np.where(y[:,1]==b)[0] for b in queries]
    combinations = []
    for array in arrays:
        combinations.append(itertools.combinations(array,2))
    for comb in combinations:
        for k, (i, j) in enumerate(comb):
            if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
                # skip if same target or different group
                continue
            X_new.append(X[i] - X[j])
            y_new.append(np.sign(y[i, 0] - y[j, 0]))
            # output balanced classes
            if y_new[-1] != (-1) ** k:
                y_new[-1] = - y_new[-1]
                X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()


def create_documents_queries_and_relevence():
    doc_set = {}
    doc_id = ""
    doc_text = ""
    with open('data/CISI.ALL') as f:
        lines = ""
        for l in f.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")
    doc_count = 0
    for l in lines:
        if l.startswith(".I"):
            doc_id = int(l.split(" ")[1].strip()) - 1
        elif l.startswith(".X"):
            doc_set[doc_id] = doc_text.lstrip(" ")
            doc_id = ""
            doc_text = ""
        else:
            doc_text += l.strip()[3:] + " "  # The first 3 characters of a line can be ignored.

    ### Processing QUERIES
    with open('data/CISI.QRY') as f:
        lines = ""
        for l in f.readlines():
            lines += "\n" + l.strip() if l.startswith(".") else " " + l.strip()
        lines = lines.lstrip("\n").split("\n")

    qry_set = {}
    qry_id = ""
    for l in lines:
        if l.startswith(".I"):
            qry_id = int(l.split(" ")[1].strip()) - 1
        elif l.startswith(".W"):
            qry_set[qry_id] = l.strip()[3:]
            qry_id = ""

    ### Processing QRELS
    rel_set = {}
    with open('data/CISI.REL') as f:
        for l in f.readlines():
            qry_id = int(l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[0]) - 1
            doc_id = int(l.lstrip(" ").strip("\n").split("\t")[0].split(" ")[-1]) - 1
            if qry_id in rel_set:
                rel_set[qry_id].append(doc_id)
            else:
                rel_set[qry_id] = []
                rel_set[qry_id].append(doc_id)
    return doc_set, qry_set, rel_set


def preprocess_string(txt, stemmer, stop_words, tokenizer, detokenizer):
    txt = txt.lower()
    punc_to_remove = string.punctuation
    punc_to_remove = punc_to_remove.replace('.', '')
    punc_to_remove = punc_to_remove.replace(',', '')
    punc_to_remove = punc_to_remove.replace("'", '')
    txt = txt.translate(str.maketrans(punc_to_remove, "".join([' ' for i in range(len(punc_to_remove))])))
    txt = txt.translate(str.maketrans('', '', ",.'"))
    tokens = tokenizer.tokenize(txt)
    tokens = [tk for tk in tokens if tk not in stop_words]
    tokens = [stemmer.stem(tk) for tk in tokens]
    txt = detokenizer.detokenize(tokens)
    return txt


def preprocess_data(docs, queries):
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    detokenizer = nltk.tokenize.TreebankWordDetokenizer()
    stemmer = nltk.stem.PorterStemmer()
    stop_words = nltk.corpus.stopwords.words('english')
    processed_docs = {}
    processed_queries = {}
    for key, doc in docs.items():
        processed_docs[key] = preprocess_string(doc, stemmer, stop_words, tokenizer, detokenizer)
    for key, query in queries.items():
        processed_queries[key] = preprocess_string(query, stemmer, stop_words, tokenizer, detokenizer)
    return processed_docs, processed_queries

def create_queries_worth(queries,seed = 42):
    np.random.seed(seed)
    queries_worth = {}
    for query in queries:
        queries_worth[query] = np.random.randint(0,100,1)[0]
    return queries_worth


