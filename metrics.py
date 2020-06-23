"""Этот файл содержит код для вычисления метрик. """

import gc
import heapq
import langid
import torch
import torchtext.vocab as torch_vocab
import gensim.downloader as api

from tqdm import tqdm
from nltk import pos_tag, word_tokenize, RegexpTokenizer
from data import save_pkl, load_pkl
from rouge_score import rouge_scorer
from easse.quality_estimation import corpus_quality_estimation


def rouge12(directory='cleaned_paraphrases',
            sentences_file='sentences_200k.pkl',
            paras_file='paraphrases_all.pkl'):
    sentences = load_pkl(f'{directory}/{sentences_file}')
    all_para = load_pkl(f'{directory}/{paras_file}')

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2'])

    rouge1_scores, rouge2_scores = [], []
    for sent, paras in tqdm(zip(sentences, all_para)):
        rouge1_scores.append([])
        rouge2_scores.append([])
        for para in paras:
            score = scorer.score(sent, para)
            rouge1_scores[-1].append(score['rouge1'].recall)
            rouge2_scores[-1].append(score['rouge2'].recall)

    save_pkl(rouge1_scores, 'stats/rouge1.pkl')
    save_pkl(rouge2_scores, 'stats/rouge2.pkl')


def pos(directory='cleaned_paraphrases',
        sents_filename='sentences_200k.pkl',
        paras_filename='paraphrases_all.pkl'):
    sents = load_pkl(f'{directory}/{sents_filename}')
    paras = load_pkl(f'{directory}/{paras_filename}')

    glove_vocab = torch_vocab.GloVe(name='twitter.27B', dim=100)

    stats_pos = []
    for i, (sent, sent_paras) in tqdm(enumerate(zip(sents, paras), 1)):
        cleaned_sent = pos_tag(word_tokenize(sent))

        stats_pos.append([pos_distance(cleaned_sent, sent_para, glove_vocab) for sent_para in sent_paras])

        if i % 25000 == 0:
            save_pkl(stats_pos, f'stats/pos_{i}.pkl')
            stats_pos = []
            gc.collect()


def pos_distance(temp_res_ori, sentence_gen, glove_vocab):
    temp_res_gen = pos_tag(word_tokenize(sentence_gen))

    temp_nn_vector_ori = []
    temp_nn_vector_gen = []

    for tube in temp_res_ori:
        if tube[1] == 'NN':
            try:
                temp_nn_vector_ori.append(glove_vocab.vectors[glove_vocab.stoi[tube[0]]])
            except KeyError:
                pass

    for tube in temp_res_gen:
        if tube[1] == 'NN':
            try:
                temp_nn_vector_gen.append(glove_vocab.vectors[glove_vocab.stoi[tube[0]]])
            except KeyError:
                pass

    if temp_nn_vector_ori and temp_nn_vector_gen:
        loss_list = []
        heap_size = min(len(temp_nn_vector_ori), len(temp_nn_vector_gen))
        for vector_target in temp_nn_vector_ori:
            for vector_gen in temp_nn_vector_gen:
                loss_list.append(torch.dist(torch.FloatTensor(vector_gen), torch.FloatTensor(vector_target)).item())
        loss_list = heapq.nsmallest(heap_size, loss_list)
        loss_nn = (sum(loss_list) / heap_size *
                   (1 + abs(len(temp_nn_vector_ori) - len(temp_nn_vector_gen)) / len(temp_nn_vector_ori)))
        return loss_nn

    return 0


def wmd(directory='cleaned_paraphrases',
        sentences_file='sentences_200k.pkl',
        paras_file='paraphrases_all.pkl'):
    model = api.load('word2vec-google-news-300')
    tokenizer = RegexpTokenizer(r'\w+')

    def clean(sentence):
        return [word for word in tokenizer.tokenize(sentence.lower())]

    sentences = load_pkl(f'{directory}/{sentences_file}')
    paras = load_pkl(f'{directory}/{paras_file}')

    stats_wmd = []
    for i, (sent, sent_paras) in tqdm(enumerate(zip(sentences, paras), 1)):
        cleaned_sent = clean(sent)
        stats_wmd.append([model.wmdistance(cleaned_sent, clean(sent_para)) for sent_para in sent_paras])
        if i % 50000 == 0:
            save_pkl(stats_wmd, f'stats/wmd_with_stopwords_{i}.pkl')
            stats_wmd = []
            gc.collect()


def quality_estimation(directory='cleaned_paraphrases',
                       sentences_file='sentences_200k.pkl',
                       paras_file='paraphrases_all.pkl'):
    sents = load_pkl(f'{directory}/{sentences_file}')
    paras = load_pkl(f'{directory}/{paras_file}')

    quality = []
    for i, (sent, sent_paras) in tqdm(enumerate(zip(sents, paras), 1)):
        estimate = []
        for sent_para in sent_paras:
            if len(sent) == 0:
                estimate.append(dict())
            else:
                estimate.append(corpus_quality_estimation([sent], [sent_para]))
        quality.append(estimate)

        if i % 50000 == 0:
            save_pkl(quality, f'stats/quality_{i}.pkl')
            quality = []
            gc.collect()


def eng_log_prob(directory='cleaned_paraphrases',
                 sentences_file='sentences_200k.pkl',
                 paras_file='paraphrases_all.pkl'):
    sentences = load_pkl(f'{directory}/{sentences_file}')
    all_para = load_pkl(f'{directory}/{paras_file}')

    langid.set_languages(['en'])

    eng_log_prob_sents, eng_log_prob_paras = [], []
    for sent, paras in tqdm(zip(sentences, all_para)):
        eng_log_prob_sents.append(langid.classify(sent)[1])
        eng_log_prob_paras.append([langid.classify(para)[1] for para in paras])

    save_pkl(eng_log_prob_sents, 'stats/eng_log_prob_sents.pkl')
    save_pkl(eng_log_prob_paras, 'stats/eng_log_prob_paras.pkl')
