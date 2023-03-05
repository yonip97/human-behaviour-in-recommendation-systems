import numpy as np
from sklearn import svm
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi


class RankSVM(svm.LinearSVC):
    """Performs pairwise ranking with an underlying LinearSVC model
    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.
    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    def __init__(self):
        super(RankSVM, self).__init__(dual=False)

    def fit(self, X, y):
        """
        Fit a pairwise ranking model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)
        Returns
        -------
        self
        """
        X_trans, y_trans = transform_pairwise(X, y)
        super(RankSVM, self).fit(X_trans, y_trans)
        return self

    def decision_function(self, X):
        return np.dot(X, self.coef_.ravel())

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.
        The item is given such that items ranked on top have are
        predicted a higher ordering (i.e. 0 means is the last item
        and n_samples would be the item ranked on top).
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            return np.dot(X, self.coef_.ravel())
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans = transform_pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)


class DocsRankSVM(RankSVM):
    def __init__(self, docs, bm25):
        super(DocsRankSVM, self).__init__()
        self.bm25 = bm25
        self.docs = docs

    def extract_features(self, query, doc, doc_id, bm25_score):
        features = [0, 0, 0, 0, 0, 0, 0]
        for word in query.split():
            word_count_in_doc = 0
            if word in self.bm25.doc_freqs[doc_id].keys():
                word_count_in_doc = self.bm25.doc_freqs[doc_id][word]
            word_count_in_corpus = 1
            if word in self.word_counts_in_corpus.keys():
                word_count_in_corpus = self.word_counts_in_corpus[word]
            features[0] += np.log2(word_count_in_doc + 1)
            features[1] += np.log2(self.amount_of_words_in_corpus / word_count_in_corpus + 1)
            idf = 0
            if word in self.bm25.idf.keys():
                idf = self.bm25.idf[word]
            features[2] += np.log2(idf + 1)
            features[3] += np.log2(word_count_in_doc / len(doc) + 1)
            features[4] += np.log2(word_count_in_doc / len(doc) * idf + 1)
            features[5] += np.log2(
                (word_count_in_doc / len(doc)) * (self.amount_of_words_in_corpus / word_count_in_corpus) + 1)
        features[6] = np.log2(bm25_score + 1)
        return features

    def create_features(self, queries):
        query_docs_features = {}
        for query_id in queries:
            bm25_scores = self.bm25.get_scores(queries[query_id].split(" "))
            query_docs_features[query_id] = {}
            for doc_id in self.docs.keys():
                query_docs_features[query_id][doc_id] = self.extract_features(queries[query_id], self.docs[doc_id],
                                                                              doc_id, bm25_scores[doc_id])
        return query_docs_features

    def train(self, queries, rel=None):
        all_corpus = [x for x in self.docs.values()]
        word_counts_in_corpus = {}
        for doc in self.bm25.doc_freqs:
            for word, count in doc.items():
                if word not in word_counts_in_corpus:
                    word_counts_in_corpus[word] = 0
                word_counts_in_corpus[word] += count
        self.word_counts_in_corpus = word_counts_in_corpus
        self.amount_of_words_in_corpus = sum([len(x) for x in all_corpus])
        query_docs_features = self.create_features(queries)
        features, target = self.prepare_for_svm(query_docs_features, queries, rel)
        super(DocsRankSVM, self).fit(features, target)

    def prepare_for_svm(self, query_docs_features, queries, rel=None):
        features = []
        target = [[], []]
        for query in queries.keys():
            for doc in query_docs_features[query].keys():
                features.append(query_docs_features[query][doc])
        features = np.stack(features)
        if rel is not None:
            for query in queries.keys():
                for doc in query_docs_features[query].keys():
                    target[0].append(1 if doc in rel[query] else 0)
                    target[1].append(query)
            target = np.array(target).T
            return features, target
        return features

    def predict(self, query):
        query = {"temp": query}
        query_docs_features = self.create_features(query)
        features = self.prepare_for_svm(query_docs_features, query)
        return super(DocsRankSVM, self).predict(features).reshape((1, -1))


class Cosiniesimilaritymodel():
    def __init__(self, docs):
        self.vectorizer = TfidfVectorizer()
        all_corpus = sorted([(k, x) for k, x in docs.items()])
        all_corpus = [x[1] for x in all_corpus]
        self.tfidf_vectors = self.vectorizer.fit_transform(all_corpus).toarray()

    def predict(self, query):
        query_vector = self.vectorizer.transform([query]).toarray()
        similarity = cosine_similarity(query_vector, self.tfidf_vectors)
        return similarity


class BM25():
    def __init__(self, all_corpus):
        self.bm_25 = BM25Okapi(all_corpus)

    def predict(self, query):
        return self.bm_25.get_scores(query).reshape((1, -1))
