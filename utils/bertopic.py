import json
import jieba
import jieba.posseg as pseg
from zhon import hanzi
import string
import utils.articut as A
import utils.data as D
import numpy as np
from typing import List, Tuple, Union, Callable

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix
import scipy.cluster.hierarchy as sch

from bertopic._utils import MyLogger, check_documents_type, check_embeddings_shape, check_is_fitted
from bertopic._ctfidf import ClassTFIDF
from bertopic.backend._utils import select_backend

logger = MyLogger("WARNING")

import pandas as pd

## setting dictionary for traditional chinese corpus
jieba.set_dictionary('../../data/tokenize/jieba/dict.txt.big')

## loading additional custom dictionary
with open("../../var/articut_dict.json") as f:
    keyword_dict = json.load(f)
    
## adding keyword to jieba
for keyword_list in keyword_dict.values():
    for keyword in keyword_list:
        jieba.add_word(keyword)

## for observe pos tagging results
from collections import defaultdict
pos_dict = defaultdict(set)

SPLITTER = '＄'
df_tokenization_database = D.read_df_split_comments_no_duplicate('all')

def _pass(doc):
    return doc

def topic_doc_tokenizer(text, splitter=SPLITTER, debug=False):
    """
        Tips: The topic representations can be updated with new `n_gram_range` or 
        new `vectorizer_model` by the BERT.update_topics() API
    """
#     print(text)
#     IO.print_dividing_line()

    split_text = text.split(SPLITTER)
    words = []
    
    for st in split_text:
        row = df_tokenization_database.query("split_comment == @st")
        articut_res = row['articut_wiki_lv2'].to_list()[0]

        ## remove unwanted pos and punctuation
        tokens = A.get_tokens(articut_res)
        words += tokens
        
    return words

## remove the preprocess part, other lines of code remain the same with original BERTopic package v0.11.0
def _c_tf_idf(self, documents_per_topic: pd.DataFrame, fit: bool = True) -> Tuple[csr_matrix, List[str]]:
#         documents = self._preprocess_text(documents_per_topic.Document.values)
        documents = documents_per_topic.Document.values

        if fit:
            self.vectorizer_model.fit(documents)

        words = self.vectorizer_model.get_feature_names()
        X = self.vectorizer_model.transform(documents)
        
        if self.seed_topic_list:
            seed_topic_list = [seed for seeds in self.seed_topic_list for seed in seeds]
            multiplier = np.array([1.2 if word in seed_topic_list else 1 for word in words])
        else:
            multiplier = None

        if fit:
            self.transformer = ClassTFIDF().fit(X, multiplier=multiplier)
            
        c_tf_idf = self.transformer.transform(X)

        return c_tf_idf, words

## modify the document joiner and the number of topic keywords, other lines of code remain the same with original BERTopic package v0.11.0
def custom_update_topics(self,
                      docs: List[str],
                      topics: List[int],
                      doc_joiner: str = '＄',
                      topic_keyword_num: int = 4,
                      n_gram_range: Tuple[int, int] = None,
                      vectorizer_model: CountVectorizer = None):
    
    check_is_fitted(self)
    
    if not n_gram_range:
        n_gram_range = self.n_gram_range

    self.vectorizer_model = vectorizer_model or CountVectorizer(ngram_range=n_gram_range)

    documents = pd.DataFrame({"Document": docs, "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': doc_joiner.join})
    
    self.c_tf_idf, words = _c_tf_idf(self, documents_per_topic) ## custom _c_tf_idf without text preprocessing
    self.topics = self._extract_words_per_topic(words)
    self._create_topic_vectors()
    self.topic_names = {key: f"{key}_" + "_".join([word[0] for word in values[:topic_keyword_num]])
                        for key, values in
                        self.topics.items()}
    
## modify the document joiner and the number of topic keywords, other lines of code remain the same with original BERTopic package v0.11.0
def custom_topics_per_class(self,
                         docs: List[str],
                         topics: List[int],
                         classes: Union[List[int], List[str]],
                         doc_joiner: str = '＄',
                         topic_keyword_num: int = 10,
                         global_tuning: bool = True,
                         diff_c_tf_idf: bool = False) -> pd.DataFrame:

        documents = pd.DataFrame({"Document": docs, "Topic": topics, "Class": classes})
        global_c_tf_idf = normalize(self.c_tf_idf, axis=1, norm='l1', copy=False)

        # For each unique timestamp, create topic representations
        topics_per_class = []
        documents_per_topic_dict = {}
        for index, class_ in enumerate(set(classes)):

            # Calculate c-TF-IDF representation for a specific timestamp
            selection = documents.loc[documents.Class == class_, :]
            documents_per_topic = selection.groupby(['Topic'], as_index=False).agg({'Document': doc_joiner.join,
                                                                                    "Class": "count"})
            print(class_)
            print(documents_per_topic)
            
            documents_per_topic_dict[class_] = documents_per_topic
            c_tf_idf, words = _c_tf_idf(self, documents_per_topic, fit=False)

            # Fine-tune the timestamp c-TF-IDF representation based on the global c-TF-IDF representation
            # by simply taking the average of the two
            if global_tuning:
                c_tf_idf = normalize(c_tf_idf, axis=1, norm='l1', copy=False)
                c_tf_idf = (global_c_tf_idf[documents_per_topic.Topic.values + self._outliers] + c_tf_idf) / 2.0
            elif diff_c_tf_idf:
                c_tf_idf = normalize(c_tf_idf, axis=1, norm='l1', copy=False)
                c_tf_idf = c_tf_idf - global_c_tf_idf[documents_per_topic.Topic.values + self._outliers]
#                 print(c_tf_idf)

            # Extract the words per topic
            labels = sorted(list(documents_per_topic.Topic.unique()))
            words_per_topic = self._extract_words_per_topic(words, c_tf_idf, labels)
            topic_frequency = pd.Series(documents_per_topic.Class.values,
                                        index=documents_per_topic.Topic).to_dict()

            # Fill dataframe with results
            topics_at_class = [(topic,
                                ", ".join([words[0] for words in values][:topic_keyword_num]),
                                topic_frequency[topic],
                                class_) for topic, values in words_per_topic.items()]
            topics_per_class.extend(topics_at_class)

        topics_per_class = pd.DataFrame(topics_per_class, columns=["Topic", "Words", "Frequency", "Class"])

        print("="*50)
        
        return topics_per_class, documents_per_topic_dict
    
## modify the document joiner and remove the preprocess part 
def custom_hierarchical_topics(self,
                            docs: List[int],
                            topics: List[int],
                            doc_joiner: str = '＄',
                            linkage_function: Callable[[csr_matrix], np.ndarray] = None,
                            distance_function: Callable[[csr_matrix], csr_matrix] = None) -> pd.DataFrame:
        if distance_function is None:
            distance_function = lambda x: 1 - cosine_similarity(x)

        if linkage_function is None:
            linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)

        # Calculate linkage
        embeddings = self.c_tf_idf[self._outliers:]
        X = distance_function(embeddings)
        Z = linkage_function(X)

        # Calculate basic bag-of-words to be iteratively merged later
        documents = pd.DataFrame({"Document": docs,
                                  "ID": range(len(docs)),
                                  "Topic": topics})
        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': doc_joiner.join})
        documents_per_topic = documents_per_topic.loc[documents_per_topic.Topic != -1, :]
#         documents = self._preprocess_text(documents_per_topic.Document.values)
        words = self.vectorizer_model.get_feature_names()
        bow = self.vectorizer_model.transform(documents_per_topic.Document.values)

        # Extract clusters
        hier_topics = pd.DataFrame(columns=["Parent_ID", "Parent_Name", "Topics",
                                            "Child_Left_ID", "Child_Left_Name",
                                            "Child_Right_ID", "Child_Right_Name"])
        for index in range(len(Z)):

            # Find clustered documents
            clusters = sch.fcluster(Z, t=Z[index][2], criterion='distance') - self._outliers
            cluster_df = pd.DataFrame({"Topic": range(len(clusters)), "Cluster": clusters})
            cluster_df = cluster_df.groupby("Cluster").agg({'Topic': lambda x: list(x)}).reset_index()
            nr_clusters = len(clusters)

            # Extract first topic we find to get the set of topics in a merged topic
            topic = None
            val = Z[index][0]
            while topic is None:
                if val - len(clusters) < 0:
                    topic = int(val)
                else:
                    val = Z[int(val - len(clusters))][0]
            clustered_topics = [i for i, x in enumerate(clusters) if x == clusters[topic]]

            # Group bow per cluster, calculate c-TF-IDF and extract words
            grouped = csr_matrix(bow[clustered_topics].sum(axis=0))
            c_tf_idf = self.transformer.transform(grouped)
            words_per_topic = self._extract_words_per_topic(words, c_tf_idf, labels=[0])

            # Extract parent's name and ID
            parent_id = index + len(clusters)
            parent_name = "_".join([x[0] for x in words_per_topic[0]][:5])

            # Extract child's name and ID
            Z_id = Z[index][0]
            child_left_id = Z_id if Z_id - nr_clusters < 0 else Z_id - nr_clusters

            if Z_id - nr_clusters < 0:
                child_left_name = "_".join([x[0] for x in self.get_topic(Z_id)][:5])
            else:
                child_left_name = hier_topics.iloc[int(child_left_id)].Parent_Name

            # Extract child's name and ID
            Z_id = Z[index][1]
            child_right_id = Z_id if Z_id - nr_clusters < 0 else Z_id - nr_clusters

            if Z_id - nr_clusters < 0:
                child_right_name = "_".join([x[0] for x in self.get_topic(Z_id)][:5])
            else:
                child_right_name = hier_topics.iloc[int(child_right_id)].Parent_Name

            # Save results
            hier_topics.loc[len(hier_topics), :] = [parent_id, parent_name,
                                                    clustered_topics,
                                                    int(Z[index][0]), child_left_name,
                                                    int(Z[index][1]), child_right_name]

        hier_topics["Distance"] = Z[:, 2]
        hier_topics = hier_topics.sort_values("Parent_ID", ascending=False)
        hier_topics[["Parent_ID", "Child_Left_ID", "Child_Right_ID"]] = hier_topics[["Parent_ID", "Child_Left_ID", "Child_Right_ID"]].astype(str)

        return hier_topics
    
def custom_extract_topics(self, documents: pd.DataFrame, doc_joiner: str = '＄'):
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': doc_joiner.join})
    self.c_tf_idf, words = _c_tf_idf(self, documents_per_topic)
    self.topics = self._extract_words_per_topic(words)
    self._create_topic_vectors()
    self.topic_names = {key: f"{key}_" + "_".join([word[0] for word in values[:4]])
                        for key, values in
                        self.topics.items()}
    
def custom_fit_transform(self,
                  documents: List[str],
                  embeddings: np.ndarray = None,
                  y: Union[List[int], np.ndarray] = None,
                  doc_joiner: str = '＄') -> Tuple[List[int], Union[np.ndarray, None]]:

    check_documents_type(documents)
    check_embeddings_shape(embeddings, documents)

    documents = pd.DataFrame({"Document": documents,
                              "ID": range(len(documents)),
                              "Topic": None})

    # Extract embeddings
    if embeddings is None:
        self.embedding_model = select_backend(self.embedding_model,
                                              language=self.language)
        embeddings = self._extract_embeddings(documents.Document,
                                              method="document",
                                              verbose=self.verbose)
        logger.info("Transformed documents to Embeddings")
    else:
        if self.embedding_model is not None:
            self.embedding_model = select_backend(self.embedding_model,
                                                  language=self.language)

    # Reduce dimensionality
    if self.seed_topic_list is not None and self.embedding_model is not None:
        y, embeddings = self._guided_topic_modeling(embeddings)
    umap_embeddings = self._reduce_dimensionality(embeddings, y)

    # Cluster reduced embeddings
    documents, probabilities = self._cluster_embeddings(umap_embeddings, documents)

    # Sort and Map Topic IDs by their frequency
    if not self.nr_topics:
        documents = self._sort_mappings_by_frequency(documents)

    # Extract topics by calculating c-TF-IDF
    custom_extract_topics(self, documents)

    # Reduce topics
    if self.nr_topics:
        documents = self._reduce_topics(documents)

    self._map_representative_docs(original_topics=True)
    probabilities = self._map_probabilities(probabilities, original_topics=True)
    predictions = documents.Topic.to_list()

    return predictions, probabilities