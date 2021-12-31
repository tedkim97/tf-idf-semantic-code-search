import os
import pickle
import json
import gzip

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# library for TF-IDF
import train 

# Taken from the Doc2Vec Documentaiton https://radimrehurek.com/gensim/models/doc2vec.html


def train_and_save_model(corpus: list, model_path: str, vs=5, wdw=2, mc=1, epch=10):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
    # train model
    model = Doc2Vec(documents, vector_size=vs, window=wdw, min_count=mc, epochs=epch, workers=8)
    model.save(model_path)
    return model

def load_model(model_path):
    return Doc2Vec.load(model_path)

def get_top_n_query_similarities(doc2vec_model, query, top_n=10):
    sdlq = query.split(' ')
    print(sdlq)
    query_tfidf = doc2vec_model.infer_vector(sdlq)
    print(query_tfidf)
    raise NotImplementedError
    
    print(doc2vec_model.vc.shape)
    print(doc2vec_model.vc)

    return


def load_queries_and_evaluate_word2vec(output, 
                                    queries_path='evaluation_results/queries.csv', 
                                    language='python', 
                                    top_n=5, 
                                    tfilter=NOISE_TOKEN_PYTHON):
    model_name = '{}_{}_mindf{}_maxdf{}'.format(language, top_n, mindf, maxdf)
    print('---------preprocessing dataset---------')
    raw_dataset = load_language_dataset(language)
    cleaned_data = [create_doc(x, tfilter=tfilter) for x in raw_dataset]
    
    # train TFIDF model
    print('---------training model---------')
    vectorizer = TfidfVectorizer(min_df=mindf, max_df=maxdf)
    X = vectorizer.fit_transform(cleaned_data)
    
    # generate file mapping query to result - why is this the most complicated part...
    print('---------running queries---------')
    queries = list(pd.read_csv(queries_path)['query'])
    num_queries = len(queries)
    
    # avoid excessive appending
    # langs = [language] * top_n * num_queries #TODO: DELETE
    qs = [None] * top_n * num_queries
    m_urls = [None] * top_n * num_queries
    snippets = [None] * top_n * num_queries

    for ind, query in enumerate(queries):
        print('{}: executing query: {}'.format(ind, query))
        top_n_indices = get_top_n_query_similarities(vectorizer, X, query, top_n=top_n)
        
        for ind2, close_ind in enumerate(top_n_indices):
            li = (ind * top_n) + ind2
            m_urls[li] = raw_dataset[close_ind]['url']
            snippets[li] = raw_dataset[close_ind]['function']
            qs[li] = query

    pd.DataFrame.from_dict({'model_name': 'tfidf_{}_{}_{}'.format(language, mindf, maxdf), 
                            'query': qs, 
                            'language': language,
                            'function': snippets,
                            'url': m_urls}).to_csv(output, index=False)
    print('done')
    return


if __name__ == '__main__':
    token_filter = {
        'python': NOISE_TOKEN_PYTHON | LOGIC_TOKEN_PYTHON | SYNTAX_TOKEN_PYTHON,
        'go': NOISE_TOKEN_GO | LOGIC_TOKEN_GO | SYNTAX_TOKEN_GO,
        'java': NOISE_TOKEN_JAVA | LOGIC_TOKEN_JAVA | SYNTAX_TOKEN_JAVA,
        'javascript': NOISE_TOKEN_JS | LOGIC_TOKEN_JS | SYNTAX_TOKEN_JS
    }

    # TODO: Refactor below
    for mif in [0.0, 0.2, 0.4, 0.6]:
        for maf in [0.4, 0.6, 0.8, 1.0]: 
            for lang in ['python', 'go', 'java', 'javascript']:
                empty_filter_fname = 'evaluation_results/param/tfidf_cosine_mindf{}_maxdf{}_{}_nofilter.csv'.format(mif, maf, lang)
                print("generating {}".format(empty_filter_fname))
                if os.path.exists(empty_filter_fname):
                    print('skipping because {} already exists - skipping'.format(empty_filter_fname))
                    continue
                try:
                    load_queries_and_evaluate_tfidf(empty_filter_fname, mindf=mif, maxdf=maf, language=lang, top_n=100, tfilter=set())
                except ValueError as err:
                    print("skipping because skippable error occured: {}".format(err))
                except:
                    raise RuntimeError

    for mif in [0.0, 0.2, 0.4, 0.6]:
        for maf in [0.4, 0.6, 0.8, 1.0]: 
            for lang in ['python', 'go', 'java', 'javascript']:
                yes_filter_fname = 'evaluation_results/param/tfidf_cosine_mindf{}_maxdf{}_{}_yesfilter.csv'.format(mif, maf, lang)
                print("generating {}".format(yes_filter_fname))
                if os.path.exists(yes_filter_fname):
                    print('skipping because {} already exists - skipping'.format(yes_filter_fname))
                    continue
                try:
                    temp_filter = token_filter[lang]
                    load_queries_and_evaluate_tfidf(yes_filter_fname, mindf=mif, maxdf=maf, language=lang, top_n=100, tfilter=set())
                except ValueError as err:
                    print("skipping because skippable error occured: {}".format(err))
                except:
                    raise RuntimeError
                
