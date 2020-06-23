"""Этот файл содержит код для считывания, записи и обработки данных."""

import gc
import random

from os import listdir
from pickle import dump, load
from tqdm import tqdm
from nltk import sent_tokenize
from transformers import BartTokenizer


def load_doc(filename):
    with open(filename, encoding='utf-8') as file:
        text = file.read()
    return text


def remove_highlight(doc):
    return doc[:doc.find('@highlight')]


def load_stories(directory):
    return [remove_highlight(load_doc(f'{directory}/{name}')) for name in tqdm(listdir(directory), desc='Load docs')]


def clean_line(line):
    index = line.find('CNN) -- ')
    return line if index == -1 else line[index + len('CNN) -- '):]


def load_pkl(filename='sentences_max_len_2500.pkl'):
    return load(open(filename, 'rb'))


def save_pkl(sentences, filename):
    dump(sentences, open(filename, 'wb'))


def make_sentences(directory='cnn_stories/', sentences_filename='sentences_max_len_2500.pkl'):
    stories = load_stories(directory)

    sentences = []
    for story in tqdm(stories, desc='Create sentences'):
        for line in story.splitlines():
            sentences.extend(filter(lambda sent: len(sent) < 2500, sent_tokenize(clean_line(line))))

    save_pkl(sentences, sentences_filename)


def merge_tokens(directory):
    all_tokens = []
    for name in tqdm(sorted(listdir(directory), key=lambda name: int(name[name.rfind('_') + 1:-4]))):
        print(name)
        all_tokens.extend(load_pkl(f'{directory}/{name}'))
    gc.collect()
    save_pkl(all_tokens, f"{directory}/{name[:name.rfind('_') + 1]}all.pkl")


def write_txt(list_, file_name):
    with open(file_name, 'w') as file_handler:
        file_handler.write('\n'.join(list_))


def encode_data(directory, filename):
    tokenizer = BartTokenizer.from_pretrained('bart-large')
    pairs = load_pkl(f'{directory}/{filename}')

    sents = [pair[0] for pair in pairs]
    paras = [pair[1] for pair in pairs]

    sents_masks = tokenizer.batch_encode_plus(sents, max_length=152, pad_to_max_length=True)
    sents, masks = sents_masks['input_ids'], sents_masks['attention_mask']

    paras = tokenizer.batch_encode_plus(paras, max_length=152, pad_to_max_length=True)['input_ids']

    save_pkl(sents, f'{directory}/sents_encoded.pkl')
    save_pkl(masks, f'{directory}/attention_masks.pkl')
    save_pkl(paras, f'{directory}/paras_encoded.pkl')


def create_labels(labels, real, all_para, stats, n_iter, n_sentences=199999):
    """ Функция для ручной разметки. """
    for i in range(n_iter):
        print(f'Итерация:{i}')

        sentence_index = random.randint(0, n_sentences)
        while (sentence_index in labels or
               any(para_wmd == float('inf') for para_wmd in stats['wmd'][sentence_index]) or
               any(para_pos is None for para_pos in stats['pos'][sentence_index]) or
               len(real[sentence_index]) < 5):
            sentence_index = random.randint(0, n_sentences)

        paras = [k for j in range(0, 30, 6) for k in random.sample(range(j, j + 6), 1)]
        paras = [para for para in paras if para < 18 or para >= 24]  # удалить парафразы, полученные с помощью top-p
        paras = list({all_para[sentence_index][para]: para for para in paras}.values())

        for k, para in enumerate(paras):
            print(f'{k}::REAL::{real[sentence_index]}')
            print(f'{k}::PARA::{all_para[sentence_index][para]}', end='\n\n')

        for k in range(len(paras)):
            print(k, end=' ')
        print()

        sent_labels = map(float, input().strip().split())

        for sent_label, para in zip(sent_labels, paras):
            labels[sentence_index].append((para, sent_label))

    return labels
