'''
generate_co_occurrences_matrices.py
Created on 2023 02 23 13:30:21
Description: This script generates two csv files, being them:

- co_occurrency_emotions.csv
    A file containing the word and the amount of times it happened with each emotion of the dataset
    Example: lorem,17,20,1,0
    Lorem appeared 17 times for emotion 1, 20 times for emotion 2 and 1 time for emotion 3.

- co_occurrency_words.csv
    A file containing the co-occurrency between words in the caption.
    Example: lorem,ipsum
    When Lorem appeared, ipsum appeared one time.

Author: Will <wlc2@cin.ufpe.br>
'''

import csv
import yaml
import os
import numpy as np
import json
from utils.emotic import get_cat_array, cat_to_one_hot


def generate_co_occurrence_words(captions, dataset=None, label=None, state=None):
    """Generates the matrices for co-occurrence based on captions (words).

    Arguments:
        captions -- caption for each sample.

    Keyword Arguments:
        dataset -- dataset name (default: {None})
        label -- split (default: {None})
        state -- if raw captions or processed captions (default: {None})
    """

    co_occurrence_words = dict()

    for caption in captions:
        caption = caption.split()
        for i in range(len(caption)-cfg['co_occ']['window_size']+1):
            key, *values = caption[i:cfg['co_occ']['window_size']+i]
            for value in values:
                if not key in co_occurrence_words:
                    co_occurrence_words[key] = {'length_': 0}
                if value in co_occurrence_words[key]:
                    co_occurrence_words[key][value] += 1
                else:
                    co_occurrence_words[key][value] = 1
                co_occurrence_words[key]['length_'] += 1

    json_ = json.dumps(co_occurrence_words, indent=2)
    with open(f"dataset/{cfg['dataset']['name']+cfg['dataset']['output_dir']}/{dataset}_{label}_{state}_com_words.json", "w") as f:
        f.write(json_)


def generate_co_occurrence_emotions(captions, cats, dataset=None, label=None, state=None):
    """Generates the matrices for co-occurrence based on emotions (labels)

    Arguments:
        captions -- list of captions
        cats -- list of categorical emotions

    Keyword Arguments:
        dataset -- dataset name (default: {None})
        label -- split (default: {None})
        state -- raw or processed captions (default: {None})
    """
    assert cats.shape[0] == captions.shape[0], "Annotations for captions and categorical emotions have different shapes!"
    co_occurrence_emotions = dict()

    for caption, cat in zip(captions, cats):
        for valid_word in caption.split():
            if not valid_word in co_occurrence_emotions.keys():
                co_occurrence_emotions[valid_word] = len(get_cat_array())*[0]
            co_occurrence_emotions[valid_word] = [
                x+int(y) for x, y in zip(co_occurrence_emotions[valid_word], cat)]

    json_ = json.dumps(co_occurrence_emotions)
    with open(f"dataset/{cfg['dataset']['name']+cfg['dataset']['output_dir']}/{dataset}_{label}_{state}_com_emotions.json", "w") as f:
        f.write(json_)


def main(caption_files, caption_path):
    for file in caption_files:
        dataset, label, _, state, _ = file.split('_')
        captions = np.load(os.path.join(caption_path, file))

        if state not in cfg['co_occ']['inputs']:
            print(f'[INFO] Skiping {file}.')
            continue

        generate_co_occurrence_words(captions, dataset, label, state)
        cats = np.load(os.path.join(
            caption_path, f"{dataset}_{label}_cat.npy"))
        generate_co_occurrence_emotions(captions, cats, dataset, label, state)


if __name__ == '__main__':
    cfg = yaml.safe_load(open('config.yaml', 'r'))
    dataset_save_dir = f"dataset/{cfg['dataset']['name']+cfg['dataset']['output_dir']}"

    caption_files = [x for x in os.listdir(dataset_save_dir) if 'caption' in x]
    cat_files = [x for x in os.listdir(
        dataset_save_dir) if x.endswith('cat.npy')]
    main(caption_files, dataset_save_dir)
