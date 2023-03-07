import time
from sklearn.model_selection import KFold
import numpy as np
from models import  DocsRankSVM, Cosiniesimilaritymodel
from rank_bm25 import BM25Okapi
from sklearn.metrics import ndcg_score



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