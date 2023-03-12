from models import DocsRankSVM, BM25, Cosiniesimilaritymodel
import numpy as np
from scipy.special import softmax


class Agent():
    def __init__(self, id, string: str, strategic: bool, number_of_word_to_add: int, strategy: str, k: int = 1,t:int = 100):
        self.id = id
        self.string = string
        self.strategic = strategic
        self.number_of_word_to_add = number_of_word_to_add
        self.strategy = strategy
        self.k = k
        self.t = t

    def create_strategy(self, queries: dict, queries_worth: dict):
        """Creates a strategy for the agent."""
        if not self.strategic:
            return
        self.add_words(queries, queries_worth)

    def add_words(self, queries: dict, queries_worth: dict):
        """Returns the agent's string with the added words."""
        if self.strategy == "top_k_softmax":
            self.top_k_softmax(queries, queries_worth)
        elif self.strategy == "dominant_word_not_present":
            self.dominant_word_not_present(queries, queries_worth)
        else:
            raise ValueError("No such strategy exist")

    def top_k_softmax(self, queries: dict, queries_worth: dict):
        words_worth = {}
        for query_id, query in queries.items():
            word_tf = {}
            for word in query.split():
                if word not in word_tf:
                    word_tf[word] = 0
                word_tf[word] += 1
            for word in word_tf:
                if word not in words_worth.keys():
                    words_worth[word] = 0
                words_worth[word] += np.log2(word_tf[word]+1) * queries_worth[query_id]
        words_worth = [(k, v) for k, v in words_worth.items()]
        words_worth = sorted(words_worth, reverse=True, key=lambda x: x[1])[:self.k]
        top_k_words = np.array([k[0] for k in words_worth])
        top_k_scores_after_temperature = [k[1]/self.t for k in words_worth]
        top_k_scores_after_temperature = softmax(top_k_scores_after_temperature)
        words = np.random.choice(top_k_words, size=self.number_of_word_to_add, p=top_k_scores_after_temperature)
        self.string += " ".join(words)

    def dominant_word_not_present(self, queries: dict, queries_worth: dict):
        words_worth = {}
        for query_id, query in queries.items():
            word_tf = {}
            for word in query.split():
                if word not in word_tf:
                    word_tf[word] = 0
                word_tf[word] += 1
            for word in word_tf:
                if word not in words_worth.keys():
                    words_worth[word] = 0
                words_worth[word] += np.log2(word_tf[word]+1) * queries_worth[query_id]
        words_worth = [(k, v) for k, v in words_worth.items()]
        words_worth = sorted(words_worth, reverse=True, key=lambda x: x[1])
        chosen_word = None
        for (word, value) in words_worth:
            if word not in self.string:
                chosen_word = word
                break
        self.string += " ".join([chosen_word for x in range(self.number_of_word_to_add)])


class Env():
    def __init__(self, docs: dict, strategic_percentage: float, model_type: str, number_of_word_to_add: int,
                 train_queries=None, train_relevance_ranking=None, seed=42, agents_params = None):
        self.agents = None
        self.docs = docs
        self.strategic_percentage = strategic_percentage
        self.number_of_words_to_add = number_of_word_to_add
        self.model = self.create_model(model_type, train_queries, train_relevance_ranking)
        self.curr_epoch = 0
        self.rng = np.random.default_rng(seed=seed)
        self.agents_params = agents_params

    def change_config(self):
        possible_agents = np.array([id for id in self.docs.keys()])
        strategic_agents = self.rng.choice(possible_agents, size=int(len(possible_agents) * self.strategic_percentage),
                                           replace=False)
        self.agents = self.create_agents(strategic_agents, possible_agents, self.number_of_words_to_add)

    def create_agents(self, strategic_agents, possible_agents, number_of_words_to_add: int):
        """Creates the agents."""
        agents = {}
        for index in possible_agents:
            if index in strategic_agents:
                agents[index] = Agent(index, self.docs[index], True, number_of_words_to_add, **self.agents_params)
            else:
                agents[index] = Agent(index, self.docs[index], False, number_of_words_to_add, **self.agents_params)
        return agents

    def create_model(self, model_type: str, train_queries, train_relevance_ranking):
        if model_type == "okapi_bm25":
            return BM25(self.docs)
        elif model_type == "tf_idf":
            return Cosiniesimilaritymodel(self.docs)
        elif model_type == "rank_svm":
            model = DocsRankSVM(self.docs, train_queries, train_relevance_ranking)
            model.train()
            return model
        else:
            raise ValueError("model_type must be one of the following: okapi_bm25, tf_idf, rank_svm")

    def corrupt(self, queries, queries_worth):
        changed_docs = {}
        for agent_ in self.agents.values():
            agent_.create_strategy(queries, queries_worth)
            changed_docs[agent_.id] = agent_.string
        self.model.update_docs(changed_docs)

    def run(self, queries: dict):
        """Runs the environment."""
        scores = {}
        for query_id, query in queries.items():
            scores[query_id] = self.model.predict(query)
        return scores
