'''
generate_dataset_annotations.py
Created on 2023 02 22 11:27:08
Description: This script will iterate through EMOTIC dataset and generate captions for each image.
It will output the following files:

- data
-- emotic_[train/test/val]_[face/context/captions/body].pth
-- co_occurrency_emotions.csv
-- co_occurrency_words.csv

Author: Will <wlc2@cin.ufpe.br>
'''

import yaml
import os
import cv2
import sys
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat
from build_expansionnet_model import build_expansionnet_model, fetch_image_caption
from utils.mediapipe_utils import get_face_locations_mediapipe
from utils.emotic import EmoticTrain, EmoticTest, cat_to_one_hot

# [0: simple tqdm (faster); 1: medium information logged; 2: all information logged]
DEBUG_LEVEL = 2


def generate_annotations(label, data_mat):
    print(f"[INFO] Working on {label}")
    # Create empty lists to store the dataset through processing
    # These will later be saved to npy files
    face_arr = list()
    context_image_arr = list()
    context_raw_caption_arr = list()
    context_nlp_caption_arr = list()
    body_arr = list()
    cat_arr = list()    # Categorical emotions
    cont_arr = list()   # Continuous emotions

    files_missing = 0
    raw_caption_cache = None
    nlp_caption_cache = None
    faces_detected = 0
    faces_not_detected = 0

    expansionnet, coco_tokens = build_expansionnet_model()

    for example in (pbar := tqdm(data_mat[label][0])):
        pbar.set_description(f'Current label: {label}')
        number_of_persons = len(example[4][0])
        for person_id in (pbar_ := tqdm(range(number_of_persons), position=1, leave=False)):
            pbar_.set_description(f'Person {person_id+1}/{number_of_persons}')
            if label == 'train':
                sample = EmoticTrain(
                    example[0][0], example[1][0], example[2], example[4][0][person_id])
            else:
                sample = EmoticTest(
                    example[0][0], example[1][0], example[2], example[4][0][person_id])

            image_path = os.path.join(
                cfg['dataset']['root_folder'], sample.folder, sample.filename)
            if not os.path.exists(image_path):
                files_missing += 1
                if files_missing >= cfg['dataset']['maximum_files_missing']:
                    raise RuntimeError(
                        f'Too many files missing! Is {cfg["dataset"]["root_folder"]} the correct dataset path?.')
                continue

            # Extract cues from images
            if raw_caption_cache is None or nlp_caption_cache is None:
                # TODO: pass image as an argument to avoid loading two times (PIL and cv2)
                raw_caption, nlp_caption = fetch_image_caption(
                    image_path, expansionnet, coco_tokens)

            else:
                raw_caption = raw_caption_cache
                nlp_caption = nlp_caption_cache

            context = cv2.imread(image_path)
            context = cv2.cvtColor(context, cv2.COLOR_BGR2RGB)

            body = context[sample.bbox[1]:sample.bbox[3],
                           sample.bbox[0]:sample.bbox[2]].copy()
            if 0 in body.shape:
                continue
            try:
                face_locations = get_face_locations_mediapipe(body)
                if face_locations is not None and None not in face_locations:
                    (a, b), (c, d) = face_locations
                    face = body[b:d, a:c, :].copy()
                    faces_detected += 1
                else:
                    face = None
                    faces_not_detected += 1
            except:
                face = None
                faces_not_detected += 1

            face_cv = cv2.resize(
                face, (224, 224)) if face is not None else None
            context_cv = cv2.resize(context, (224, 224))
            try:
                body_cv = cv2.resize(body, (224, 224))
            except cv2.error:
                continue

            if sample.cat_annotators == 0 or sample.cont_annotators == 0:
                # Checks if sample was not annotation (possible)
                continue

            face_arr.append(face_cv)
            context_image_arr.append(context_cv)
            context_raw_caption_arr.append(raw_caption)
            context_nlp_caption_arr.append(nlp_caption)
            body_arr.append(body_cv)

            if label == 'train':
                cat_arr.append(cat_to_one_hot(sample.cat))
                cont_arr.append(sample.cont)
            else:
                cat_arr.append(cat_to_one_hot(sample.comb_cat))
                cont_arr.append(sample.comb_cont)

            if raw_caption_cache is None or nlp_caption_cache is None:
                raw_caption_cache = raw_caption
                nlp_caption_cache = nlp_caption
        raw_caption_cache = None
        nlp_caption_cache = None

    face_arr = np.array(face_arr)
    context_image_arr = np.array(context_image_arr)
    context_raw_caption_arr = np.array(context_raw_caption_arr)
    context_nlp_caption_arr = np.array(context_nlp_caption_arr)
    body_arr = np.array(body_arr)

    cont_arr = np.array(cont_arr)
    cat_arr = np.array(cat_arr)

    if not os.path.exists(cfg['dataset']['name']+cfg['dataset']['output_dir']):
        os.makedirs('dataset/'+cfg['dataset']['name'] +
                    cfg['dataset']['output_dir'], exist_ok=True)

    np.save(
        os.path.join(
            'dataset',
            cfg['dataset']['name']+cfg['dataset']['output_dir'],
            f"{cfg['dataset']['name']}_{label}_face.npy"
        ), face_arr
    )
    np.save(
        os.path.join(
            'dataset',
            cfg['dataset']['name']+cfg['dataset']['output_dir'],
            f"{cfg['dataset']['name']}_{label}_context_image.npy"
        ), context_image_arr
    )
    np.save(
        os.path.join(
            'dataset',
            cfg['dataset']['name']+cfg['dataset']['output_dir'],
            f"{cfg['dataset']['name']}_{label}_context_raw_caption.npy"
        ), context_raw_caption_arr
    )
    np.save(
        os.path.join(
            'dataset',
            cfg['dataset']['name']+cfg['dataset']['output_dir'],
            f"{cfg['dataset']['name']}_{label}_context_nlp_caption.npy"
        ), context_nlp_caption_arr
    )
    np.save(
        os.path.join(
            'dataset',
            cfg['dataset']['name']+cfg['dataset']['output_dir'],
            f"{cfg['dataset']['name']}_{label}_body.npy"
        ), body_arr
    )
    np.save(
        os.path.join(
            'dataset',
            cfg['dataset']['name']+cfg['dataset']['output_dir'],
            f"{cfg['dataset']['name']}_{label}_cat.npy"
        ), cat_arr
    )
    np.save(
        os.path.join(
            'dataset',
            cfg['dataset']['name']+cfg['dataset']['output_dir'],
            f"{cfg['dataset']['name']}_{label}_cont.npy"
        ), cont_arr
    )


if __name__ == '__main__':
    cfg = yaml.safe_load(open('config.yaml', 'r'))
    annotations_path = os.path.join(
        cfg['dataset']['root_folder'], 'Annotations', 'Annotations.mat')
    print(f"[INFO] Loading annotations from {annotations_path}")
    mat = loadmat(annotations_path)
    print(
        f"[INFO] Dataset will be saved to dataset/{cfg['dataset']['name']+cfg['dataset']['output_dir']}")

    for label in ['train', 'test', 'val']:
        generate_annotations(label, mat)

    print(annotations_path)
