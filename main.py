import string
import time
import optuna

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from models import RankSVM, DocsRankSVM, Cosiniesimilaritymodel
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from sklearn.metrics import ndcg_score


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


def main():
    #top_docs = trial.suggest_int('top_docs',1,100)
    #bottom_docs = trial.suggest_int('bottom_docs',0,100)
    #alpha = trial.suggest_float('a',0.8,1.2)
    #beta = trial.suggest_float('b',0.6,0.9)
    #gamma = trial.suggest_float('c',0,0.3)
    test_size = 0.2
    model_name = "svm"
    k = 5
    amount_of_queries = 50
    docs, queries, rel = create_documents_queries_and_relevence()
    queries_with_relevant_docs = list(rel.keys())
    queries = {k: queries[k] for k in queries_with_relevant_docs}
    docs, queries = preprocess_data(docs, queries)
    chosen_queries = [x for x in queries.keys()][:amount_of_queries]
    cv = KFold(k)
    ndcg_values = []
    for train_indices, test_indices in cv.split(chosen_queries):
        train_ids = [chosen_queries[x] for x in train_indices]
        test_ids = [chosen_queries[x] for x in test_indices]
        train_queries = {k: queries[k] for k in train_ids}
        train_rel = {k: rel[k] for k in train_ids}
        tokenized_corpus = [(index, x) for index, x in docs.items()]
        tokenized_corpus = sorted(tokenized_corpus)
        tokenized_corpus = [x[1].split(" ") for x in tokenized_corpus]
        if model_name == "bm25":
            model = BM25Okapi(tokenized_corpus)
        elif model_name == "rocchio":
            model = Cosiniesimilaritymodel(docs)
        elif model_name == "svm":
            bm25 = BM25Okapi(tokenized_corpus)
            model = DocsRankSVM(docs, bm25)
            print("Starting SVM training")
            start = time.time()
            model.train(train_queries, train_rel)
            print(f"SVM training took {time.time() - start:.4f} seconds")
        else:
            raise ValueError("No such model exist")
        for id in test_ids:
            gt = np.array([[1 if i in rel[id] else 0 for i in docs.keys()]])
            if model_name == "rocchio":
                scores = model.predict(queries[id])
            elif model_name == "bm25":
                scores = model.get_scores(queries[id])
            elif model_name == "svm":
                scores = model.predict(queries[id])
                print(scores)
            else:
                raise ValueError("No such model")
            ndcg_values.append(ndcg_score(gt, scores))
    avg_ndcg = np.mean(ndcg_values)
    print(avg_ndcg)
    return avg_ndcg


if __name__ == "__main__":
    # study = optuna.create_study(direction="maximize")
    # study.optimize(lambda trial: main(trial, "rocchio"),n_trials=250)
    # print(study.best_params)
    # print(study.best_value)
    main()