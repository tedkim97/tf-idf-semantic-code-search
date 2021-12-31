import os
import pickle
import json
import gzip

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel, linear_kernel
from sklearn.neighbors import NearestNeighbors

# TODO - add better documentation and be consistent with parameter typing
# TODO - refactor sloppy duplication

# -------------------------- Data Functions -----------------------------------
def train_val_test_split(data, test_size, val_size, seed=0):
    '''
    Split dataset into a train-validate-test set

    test_size and val_size should be decimals ex: test_size=0.1, val_ize=0.1
    '''
    assert(int(val_size + test_size) < 1)
    vt_sum = val_size + test_size
    vt_split = test_size / vt_sum
    print(vt_split)
    train_data, vt_data = train_test_split(data, test_size=vt_sum, random_state=seed)
    val_data, test_data = train_test_split(vt_data, test_size=vt_split, random_state=seed)
    print(len(train), len(val_data), len(test_data))
    assert((len(train_data) + len(val_data) + len(test_data)) == len(data))
    return train_data, val_data, test_data

def load_language_dataset(language:str, prepath='data', template='_dedupe_definitions_v2.pkl'):
    '''
    valid languages are determined by which zips you've unpickled: java, javascript, go, python
    '''
    with open(os.path.join(prepath, language + template), 'rb') as f:
        raw_data = pickle.load(f)
    return raw_data

# Interperability stuff
def interpret_tfidf(output_pattern, tfidf_vectorizer):
    # note - depending on the version of sklearn you may need to use get_feature_names_out() instead
    template = {'feature_name': tfidf_vectorizer.get_feature_names(), 
                'idf_weight': tfidf_vectorizer.idf_}
    pd.DataFrame.from_dict(template).to_csv('idf_weight_' + output_pattern, index=False)

    vocab_index_template = {'vocabulary': list(tfidf_vectorizer.vocabulary_.keys()),
                            'index': list(tfidf_vectorizer.vocabulary_.values())}
    pd.DataFrame.from_dict(vocab_index_template).to_csv('vocab_to_index' + output_pattern, index=False)
    
    stop_word_template = {'stop_words': list(tfidf_vectorizer.stop_words_)}
    pd.DataFrame.from_dict(stop_word_template).to_csv('stopwords_' + output_pattern, index=False)
    return

def map_K_SSE(data, min_k, max_k):
    iters = range(min_k, max_k+1, 4)
    sse = []
    for k in iters:
        res = MiniBatchKMeans(n_clusters=k, init_size=1024, batch_size=2048, random_state=20).fit(data)
        sse.append(res.inertia_)
        print('Fitting {} clusters'.format(k))
        
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Number of Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')

# Hardcoded Sets 
NOISE_TOKEN_PYTHON = set([",", ".", "(", ")", "[", "]", "{", "}"])
LOGIC_TOKEN_PYTHON = set(["and", "or", "is", "not", "!=", "==", ">=", "<=", ">", "<"])
SYNTAX_TOKEN_PYTHON = set(["#", "def", "in", "for", "while", "raise", "return", "with", "="])

NOISE_TOKEN_GO = set([",", ".", "(", ")", "[", "]", "{", "}"])
LOGIC_TOKEN_GO = set(["&&", "||", "!" , "!=", "==", ">=", "<=", ">", "<"])
SYNTAX_TOKEN_GO = set(["//", "func", "for", "=", ":=", "nil", "go", "defer"])

NOISE_TOKEN_JAVA = set([",", ".", "(", ")", "[", "]", "{", "}"])
LOGIC_TOKEN_JAVA = set(["&&", "||", "!" , "!=", "==", ">=", "<=", ">", "<"])
SYNTAX_TOKEN_JAVA = set(["//", "public", "private", "static", "for", "while", "throw", "null", "go", "defer"])

NOISE_TOKEN_JS = set([",", ".", "(", ")", "[", "]"])
LOGIC_TOKEN_JS = set(["&&", "||", "!" , "!=", "==", "!==", "===", ">=", "<=", ">", "<"])
SYNTAX_TOKEN_JS = set(["//", "function", "for", "=", "=>", "then"])

def filter_token_list(token_list, tfilter):
    '''

    '''
    # the requirement that x.isdigit() is for filtering out values that are pure numbers
    return [x.lower() for x in token_list if not (x in tfilter or x.isdigit())]

def clean_parameter(param_string):
    """
    Removes (,) tokens from an argument list in additional to default parameters
    """
    tokenless = param_string[1:-1].split(',')
    optional_paramless = [x.strip().split("=")[0] for x in tokenless]
    return optional_paramless

def parse_function(code_dict: dict, tfilter):
    # clean function_tokens, docstring_tokens,
    docstring = filter_token_list(code_dict['docstring_tokens'], tfilter)
    function = filter_token_list(code_dict['function_tokens'], tfilter)
    # params are represented in funcion body - inclusion means double counting
    # params = clean_parameter(code_dict['parameters'])
    # leaving out return statements for now
    # returns = code_dict['return_statement']
    return docstring, function

def create_doc(code_dict: dict, tfilter):
    '''
    Create a document from a value within the dataset. Filters out tokens according to tfilter (set)
    ex: cleaned_data = [create_doc(x) for x in python_dataset]
    '''
    docstring, func = parse_function(code_dict, tfilter)
    return " ".join(docstring + func)


def get_top_n_query_similarities(vectr_model, docs_tfidf, query, top_n=10):
    '''
    modified from somewhere on stack overflow, I just forgot where
    '''
    query_tfidf = vectr_model.transform([query])
    cosineSims = cosine_similarity(query_tfidf, docs_tfidf).flatten()
    indices = np.argsort(cosineSims)[::-1][:top_n]
    return indices

def get_top_n_query_similarities_RBF_KERNEL(vectr_model, docs_tfidf, query, top_n=10):
    '''
    modified from somewhere on stack overflow, I just forgot where
    '''
    query_tfidf = vectr_model.transform([query])
    rbfSims = rbf_kernel(query_tfidf, docs_tfidf).flatten()
    indices = np.argsort(rbfSims)[::-1][:top_n]
    return indices

def get_top_n_query_similarities_LINEAR_KERNEL(vectr_model, docs_tfidf, query, top_n=10):
    '''
    modified from somewhere on stack overflow, I just forgot where
    '''
    query_tfidf = vectr_model.transform([query])
    linearSims = linear_kernel(query_tfidf, docs_tfidf).flatten()
    print(linearSims)
    indices = np.argsort(linearSims)[::-1][:top_n]
    print(linearSims[np.argsort(linearSims)[::-1]])
    return indices

def load_queries_and_evaluate_tfidf(output, 
                                    mindf=1, maxdf=1.0, 
                                    queries_path='evaluation_results/queries.csv', 
                                    language='python', 
                                    top_n=5, 
                                    tfilter=NOISE_TOKEN_PYTHON):
    model_name = '{}_{}_mindf{}_maxdf{}'.format(language, top_n, mindf, maxdf)
    print('---------preprocessing dataset (cosine) ---------')
      # if you're wondering "why not read from disk, once and pass it into load_queries if we'r egoing be running multiple experiments?" 
    # note that (for some) the memory goes out of control if we do that and the program crashes
    raw_dataset = load_language_dataset(language)
    cleaned_data = [create_doc(x, tfilter) for x in raw_dataset]
    
    # train TFIDF model
    print('---------training model---------')
    vectorizer = TfidfVectorizer(min_df=mindf, max_df=maxdf)
    X = vectorizer.fit_transform(cleaned_data)
    
    # generate file mapping query to result - why is this the most complicated part...
    print('---------running queries (cosine) ---------')
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

# TODO - refactor by adding top_n_similarities function as funciton argument
def evaluate_tfidf_RBF_KERNEL(output, mindf=1, maxdf=1.0, 
                                queries_path='evaluation_results/queries.csv', 
                                language='python', 
                                top_n=5, 
                                tfilter=NOISE_TOKEN_PYTHON):
    model_name = '{}_{}_mindf{}_maxdf{}'.format(language, top_n, mindf, maxdf)
    print('---------preprocessing dataset (rbf)---------')
    raw_dataset = load_language_dataset(language) 
    cleaned_data = [create_doc(x, tfilter) for x in raw_dataset] 
    
    # train TFIDF model
    print('---------training model---------')
    vectorizer = TfidfVectorizer(min_df=mindf, max_df=maxdf)
    X = vectorizer.fit_transform(cleaned_data)
    
    # generate file mapping query to result - why is this the most complicated part...
    print('---------running queries (rbf) ---------')
    queries = list(pd.read_csv(queries_path)['query'])
    num_queries = len(queries)
    
    # avoid excessive appending
    qs = [None] * top_n * num_queries
    m_urls = [None] * top_n * num_queries
    snippets = [None] * top_n * num_queries

    for ind, query in enumerate(queries):
        print('{}: executing query: {}'.format(ind, query))
        top_n_indices = get_top_n_query_similarities_RBF_KERNEL(vectorizer, X, query, top_n=top_n)
        
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

def evaluate_tfidf_LINEAR_KERNEL(output, mindf=1, maxdf=1.0, 
                                queries_path='evaluation_results/queries.csv', 
                                language='python', 
                                top_n=5, 
                                tfilter=NOISE_TOKEN_PYTHON):
    model_name = '{}_{}_mindf{}_maxdf{}'.format(language, top_n, mindf, maxdf)
    print('---------preprocessing dataset (linear)---------')
    raw_dataset = load_language_dataset(language) 
    cleaned_data = [create_doc(x, tfilter) for x in raw_dataset] 
    
    # train TFIDF model
    print('---------training model---------')
    vectorizer = TfidfVectorizer(min_df=mindf, max_df=maxdf)
    X = vectorizer.fit_transform(cleaned_data)
    
    # generate file mapping query to result - why is this the most complicated part...
    print('---------running queries (linear)---------')
    queries = list(pd.read_csv(queries_path)['query'])
    num_queries = len(queries)
    
    # avoid excessive appending
    qs = [None] * top_n * num_queries
    m_urls = [None] * top_n * num_queries
    snippets = [None] * top_n * num_queries

    for ind, query in enumerate(queries):
        print('{}: executing query: {}'.format(ind, query))
        top_n_indices = get_top_n_query_similarities_LINEAR_KERNEL(vectorizer, X, query, top_n=top_n)
        
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

    # evaluate_tfidf_RBF_KERNEL('evaluation_results/rbf_qualitative1.csv', mindf=100, maxdf=1.0, language='python', top_n=20, tfilter=set())
    # evaluate_tfidf_RBF_KERNEL('evaluation_results/rbf_qualitative2.csv', mindf=100, maxdf=1.0, language='python', top_n=20, tfilter=token_filter['python'])
    # evaluate_tfidf_RBF_KERNEL('evaluation_results/param/tfidf_rbf_mindf100_maxdf1.0_python_nofilter_v1.csv', mindf=100, maxdf=1.0, language='python', top_n=100, tfilter=set())
    # evaluate_tfidf_RBF_KERNEL('evaluation_results/param/tfidf_rbf_mindf100_maxdf1.0_python_nofilter_v2.csv', mindf=100, maxdf=1.0, language='python', top_n=100, tfilter=token_filter['python'])

    # evaluate_tfidf_LINEAR_KERNEL('evaluation_results/linear_qualitative1.csv', mindf=100, maxdf=1.0, language='python', top_n=20, tfilter=set())
    # evaluate_tfidf_LINEAR_KERNEL('evaluation_results/linear_qualitative2.csv', mindf=100, maxdf=1.0, language='python', top_n=20, tfilter=token_filter['python'])
    # evaluate_tfidf_LINEAR_KERNEL('evaluation_results/param/tfidf_linear_mindf100_maxdf1.0_python_nofilter_v1.csv', mindf=100, maxdf=1.0, language='python', top_n=100, tfilter=set())
    # evaluate_tfidf_LINEAR_KERNEL('evaluation_results/param/tfidf_linear_mindf100_maxdf1.0_python_nofilter_v2.csv', mindf=100, maxdf=1.0, language='python', top_n=100, tfilter=token_filter['python'])

    # load_queries_and_evaluate_tfidf('evaluation_results/cosine_qualitative1.csv', mindf=100, maxdf=1.0, language='python', top_n=20, tfilter=set())
    # load_queries_and_evaluate_tfidf('evaluation_results/cosine_qualitative2.csv', mindf=100, maxdf=1.0, language='python', top_n=20, tfilter=token_filter['python'])
    

    # TODO: Refactor below
    for mif in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        for maf in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: 
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

    for mif in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        for maf in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: 
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
                
    for mif in [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
        for maf in [0.8, 0.9, 1.0]: 
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