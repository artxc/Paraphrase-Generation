"""Этот файл содержит код для ранжирования примеров парафраза."""

import gc
import tqdm

from collections import defaultdict
from data import *


def scoring(n_paras=30,
            n_sentences=200000,
            cls_file='classification_tree_bag_acc74_f72.pkl',
            reg_file='regression_forest_maxdepth_7_rsq_35_mae_076_tree_10.pkl'):
    wmds = load_pkl('stats/wmd_with_stopwords_all.pkl')
    stats = load_pkl('stats/stats.pkl')

    classifier = load_pkl(cls_file)
    regressor = load_pkl(reg_file)

    scores = defaultdict(list)

    quarter = n_sentences // 4
    for i in range(4):
        left, right = quarter * i, quarter * (i + 1)

        quality = load_pkl(f'stats/quality_{right}.pkl')

        for i in tqdm(range(left, right)):
            if (any(para_wmd == float('inf') for para_wmd in wmds[i]) or
                    any(len(para_quality.keys()) == 0 for para_quality in quality[i - left])):
                continue

            for j in range(n_paras):
                if 18 <= j <= 23:  # удалить парафразы, полученные с помощью top-p
                    continue

                bleu = stats['bleu'][i][j]
                rouge_l = stats['rouge-l'][i][j]
                wmd = wmds[i][j]
                pos = stats['pos'][i][j] if stats['pos'][i][j] is not None else 0
                rouge_1 = stats['rouge-1'][i][j]
                rouge_2 = stats['rouge-2'][i][j]
                levenstein = quality[i - left][j]['Levenshtein similarity']
                additions = quality[i - left][j]['Additions proportion']
                deletions = quality[i - left][j]['Deletions proportion']

                x = [[bleu, rouge_l, wmd, pos, rouge_1, rouge_2, levenstein, additions, deletions]]
                scores[i].append((classifier.predict(x)[0], regressor.predict(x)[0], j))

        del quality
        gc.collect()

        save_pkl(scores, f'stats/scores_{right}.pkl')
