# take csv and combine review texts so that each restaurant has exactly one review text
import gensim
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk

def combine_text(csv_address,new_csv_name):
    ubst = pd.read_csv(csv_address)
    data_text = ubst[["text"]]
    business_id = ubst[["business_id"]]
    unique_business_id = business_id["business_id"].unique()
    new_text_vec = [0] * len(unique_business_id)
    for i in range(len(unique_business_id)):
        temp_list = ubst[ubst["business_id"] == unique_business_id[i]][["text"]]["text"].tolist()
        combined_string = " ".join(temp_list)
        new_text_vec[i] = combined_string

    new_df = pd.DataFrame({"business_id": unique_business_id, "text": new_text_vec})
    new_csv  = new_df.to_csv(new_csv_name)
    return(new_df)


def compute_lda(csv_address,num_topics):
    # assume csv is already processed so that business id is unique
    # return trained lda latent probability matrix for given num of topics
    # return matrix which has num_unique_restaurants x num_topics entries
    # return list of unique business ids
    #np.random.seed(2018)
    nltk.download('wordnet')
    stemmer = PorterStemmer()

    def lemmatize_stemming(text):
        return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
        return result

    ubst = pd.read_csv(csv_address)
    data_text = ubst[["text"]]
    unique_business_id = ubst[["business_id"]]
    processed_docs = data_text['text'].map(preprocess)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=2)
    lda_latent = lda_model.get_document_topics(bow_corpus)
    probability_matrix = np.zeros((len(lda_latent), num_topics))
    for i in range(len(lda_latent)):
        temp = lda_latent[i]
        for j in range(len(temp)):
            probability_matrix[i, temp[j][0]] = temp[j][1]

    new_df = pd.DataFrame(probability_matrix)
    df_singleton = pd.DataFrame({"business_id": unique_business_id["business_id"]})
    new_new_df = pd.concat([new_df, df_singleton], axis=1)
    new_new_df.to_csv('latent_probability_restaurants.csv')
    return(probability_matrix)

