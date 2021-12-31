import os
import pandas as pd
import referenceeval # reference script provided by codesearchnet challenge 

from collections import defaultdict
import numpy as np

def load_relevances(filepath: str):
    relevance_annotations = pd.read_csv(filepath)
    per_query_language = relevance_annotations.pivot_table(
        index=['Query', 'Language', 'GitHubUrl'], values='Relevance', aggfunc=np.mean)

    # Map language -> query -> url -> float
    relevances = defaultdict(lambda: defaultdict(dict))  # type: Dict[str, Dict[str, Dict[str, float]]]
    for (query, language, url), relevance in per_query_language['Relevance'].items():
        relevances[language.lower()][query.lower()][url] = relevance
    return relevances

def load_predictions(filepath: str, max_urls_per_language: int=300):
    prediction_data = pd.read_csv(filepath)

    # Map language -> query -> Ranked List of URL
    predictions = defaultdict(lambda: defaultdict(list))
    for _, row in prediction_data.iterrows():
        predictions[row['language'].lower()][row['query'].lower()].append(row['url'])
    for query_data in predictions.values():
        for query, ranked_urls in query_data.items():
            query_data[query] = ranked_urls[:max_urls_per_language]

    return predictions

def coverage_per_language(predictions, relevance_scores, with_positive_relevance=False):
    """
    Compute the % of annotated URLs that appear in the algorithm's predictions.
    """
    num_annotations = 0
    num_covered = 0
    for query, url_data in relevance_scores.items():
        urls_in_predictions = set(predictions[query])
        for url, relevance in url_data.items():
            if not with_positive_relevance or relevance > 0:
                num_annotations += 1
                if url in urls_in_predictions:
                    num_covered += 1
    return num_covered / num_annotations

def ndcg(predictions, relevance_scores, ignore_rank_of_non_annotated_urls=True):
    num_results = 0
    ndcg_sum = 0

    for query, query_relevance_annotations in relevance_scores.items():
        current_rank = 1
        query_dcg = 0
        for url in predictions[query]:
            if url in query_relevance_annotations:
                query_dcg += (2**query_relevance_annotations[url] - 1) / np.log2(current_rank + 1)
                current_rank += 1
            elif not ignore_rank_of_non_annotated_urls:
                current_rank += 1

        query_idcg = 0
        for i, ideal_relevance in enumerate(sorted(query_relevance_annotations.values(), reverse=True), start=1):
            query_idcg += (2 ** ideal_relevance - 1) / np.log2(i + 1)
        if query_idcg == 0:
            # We have no positive annotations for the given query, so we should probably not penalize anyone about this.
            continue
        num_results += 1
        ndcg_sum += query_dcg / query_idcg
    return ndcg_sum / num_results

def extract_info_from_filename(fname: str, true_lbls: str):
    if ('.csv' not in fname):
        raise ValueError('{} does not seem to be a .csv'.format(fname))

    print(fname)
    prepath_cleaned = fname.split('/')[-1]
    vals = prepath_cleaned[:-4].split('_')
    language = vals[4]
    
    acc_naive, acc_relevant, ndcg_naive, ndcg_full_ranking = referenceeval.compare(true_lbls, fname, language)
    print(acc_naive, acc_relevant, ndcg_naive, ndcg_full_ranking)

    filter_employed = 'yes' in vals[5] 
    return vals[0], vals[1], vals[2].split('mindf')[1], vals[3].split('maxdf')[1], language, filter_employed, acc_naive, acc_relevant, ndcg_naive, ndcg_full_ranking

def fill_df(fnames: list, col_lbl:list):
    vals = [None] * len(fnames)
    for ind, fname in enumerate(fnames):
        vals[ind] = extract_info_from_filename(fname, 'truth.csv')
    
    return pd.DataFrame(data=vals, columns=col_lbl)            

if __name__ == '__main__':

    COLUMN_LABELS = columns=['vectorizer', 'similarity_metric', 'min_df_param', 'max_df_param', 'language', 'langfilter', 'accuracy (%)', 'accuracy (average relevancy > 0) (%)', 'ndcg', 'ndcg (full rank)']
    source = 'param' 
    predictions = [os.path.join(source, x) for x in os.listdir(source)]
    print('number of files =', len(predictions))
    df = fill_df(predictions, COLUMN_LABELS)
    df.to_csv('tfidf_params_final.csv', index=False)


