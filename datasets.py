'''
datasets.py
Created on 2023 02 25 10:04:53
Description: This file manages the loading and management of preprocessed dataset files.

Author: Will <wlc2@cin.ufpe.br>
'''

import dgl
import json
import nltk
from nltk.corpus import wordnet
from senticnet5 import senticnet as stn
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from utils.emotic import get_cat_array


def load_glove():
    """Loads glove's embeddings

    Returns:
        A dictionary with glove embeddings.
    """
    glove_dict = {}
    with open('glove/glove.6B.50d.txt', encoding="utf-8") as f:
        for line in f:
            word_, *vector = line.split()
            glove_dict[word_] = [float(x) for x in vector]

    return glove_dict


glove_object = load_glove()


def fetch_glove_embeddings(word, memory=True):
    """Fetches a representation from glove given an input word

    Arguments:
        word -- word to seek representations

    Keyword Arguments:
        memory -- keep a cache of previous representations (default: {True})

    Returns:
        embedding of given word
    """

    word = word.lower()
    if word == 'doubt/confusion':
        word = 'confusion'
    if memory:
        try:
            return glove_object[word]
        except KeyError:
            with open('glove/glove.6B.50d_mod.txt', 'a') as f:
                string = ''
                string = string+word
                vector = list(np.random.uniform(low=-1, high=1, size=(50,)))
                for val in vector:
                    string = string + ' ' + str("%.6f" % val)
                f.write('\n'+string)
                glove_object[word] = vector
                return glove_object[word]
    if not memory:
        with open('glove/glove.6B.50d.txt', encoding="utf-8") as f:
            # Search the word in glove file
            for line in f:
                word_, *vector = line.split()
                if word == word_:
                    # If word is found, return vector representation
                    return [float(x) for x in vector]
            # Otherwise, generate a new random vector and save it to file
            with open('glove/glove.6B.50d_mod.txt', 'a') as f:
                string = ''
                string = string+word
                vector = list(np.random.uniform(low=-1, high=1, size=(50,)))
                for val in vector:
                    string = string + ' ' + str("%.6f" % val)
                f.write('\n'+string)
                return vector


def get_com(com_e=None, com_w=None):
    """Loads co-occurrence matrices to memory

    Keyword Arguments:
        com_e -- Path to co-occurrence matrices of emotion categories (default: {None})
        com_w -- Path to co-occurrence matrices of word pairings (default: {None})

    Raises:
        RuntimeError: if paths are not found

    Returns:
        Co-occurrence matrices for both emotion and word pairings
    """

    try:
        with open(com_e, 'r') as f:
            com_e = json.load(f)
        with open(com_w, 'r') as f:
            com_w = json.load(f)
    except:
        raise RuntimeError(
            f"Unable to load co-occurence matrices from {com_e} and {com_w}. Please check the paths.")
    return com_e, com_w


def get_wordnet_synonyms(word):
    """Finds synonyms on WordNet

    Arguments:
        word -- word to search synonyms

    Returns:
        Synonym of given word
    """
    synonyms = {}
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms[l.name()] = l.synset().wup_similarity(
                wordnet.synsets(word)[0])

    sorted_synonms = sorted(synonyms.items(), key=lambda x: x[1], reverse=True)
    return sorted_synonms


def generate_graph(caption, com_e, com_w):
    """Generates knowledge graph based on the given caption.

    Arguments:
        caption -- input caption
        com_e -- co-occurrence of emotions
        com_w -- co-occurrence of words

    Returns:
        dgl graph
    """
    nodes = {}
    caption = caption.split()
    # iterate over co-occurrence matrix to find matches
    # Generates a knowledge dictionary to feed our graph
    for id, c in enumerate(caption):
        try:
            com_e_scores = ([int(float(x)) for x in com_e[c.lower()]])
        except KeyError:
            com_e_scores = (list(np.zeros(26,)))
        nodes[c] = {}
        nodes[c]['id'] = id
        nodes[c]['com_emotion'] = com_e_scores
        nodes[c]['glove'] = fetch_glove_embeddings(c)
        try:
            pleasantness_value, _, _, _, primary_mood, secondary_mood, _, \
                polarity_value, semantics1, semantics2, semantics3, semantics4, semantics5 = stn[
                    c]
        except KeyError:
            # print(f'The following word is not available on SenticNet: {c}.')
            # print(f'Searching for synonyms in WordNet.')
            synonyms = get_wordnet_synonyms(c)
            caption_ = " ".join(caption)
            remove_word = True
            for s, _ in synonyms:
                if s in stn.keys():
                    # print(f'Replacing {c} with synonym {s}.')
                    # caption_ = caption_.replace(c, s)
                    # print(f'New caption: {caption_}')
                    pleasantness_value, _, _, _, primary_mood, secondary_mood, _, \
                        polarity_value, semantics1, semantics2, semantics3, semantics4, semantics5 = stn[
                            s]
                    remove_word = False
                    break
                    # return generate_graph(caption_, com_e, com_w)
            if remove_word:
                # print(f"Removing {c} from {caption_}: {caption_.replace(c, '')}")
                caption_ = caption_.replace(c, '')
                if caption_ == '':
                    return generate_graph('None', com_e, com_w)
                return generate_graph(caption_, com_e, com_w)
        nodes[c]['sentic_descriptions'] = {
            'pleasantness_value': pleasantness_value,
            'primary_mood': primary_mood.replace('#', ''),
            'primary_mood_glove': fetch_glove_embeddings(primary_mood.replace('#', '')),
            'secondary_mood': secondary_mood.replace('#', ''),
            'secondary_mood_glove': fetch_glove_embeddings(secondary_mood.replace('#', '')),
            'polarity_value': polarity_value,
            'semantics1': {
                'root_word': semantics1,
                'root_word_glove': fetch_glove_embeddings(semantics1),
                'word_semantics': stn[semantics1][7:],
                'word_semantics_glove': [fetch_glove_embeddings(x) for x in stn[semantics1][8:]]
            },
            'semantics2': {
                'root_word': semantics2,
                'root_word_glove': fetch_glove_embeddings(semantics2),
                'word_semantics': stn[semantics2][7:],
                'word_semantics_glove': [fetch_glove_embeddings(x) for x in stn[semantics2][8:]]
            },
            'semantics3': {
                'root_word': semantics3,
                'root_word_glove': fetch_glove_embeddings(semantics3),
                'word_semantics': stn[semantics3][7:],
                'word_semantics_glove': [fetch_glove_embeddings(x) for x in stn[semantics3][8:]]
            },
            'semantics4': {
                'root_word': semantics4,
                'root_word_glove': fetch_glove_embeddings(semantics4),
                'word_semantics': stn[semantics4][7:],
                'word_semantics_glove': [fetch_glove_embeddings(x) for x in stn[semantics4][8:]]
            },
            'semantics5': {
                'root_word': semantics5,
                'root_word_glove': fetch_glove_embeddings(semantics5),
                'word_semantics': stn[semantics5][7:],
                'word_semantics_glove': [fetch_glove_embeddings(x) for x in stn[semantics5][8:]]
            },
        }
        try:  # Checks if there is another word in the caption description
            nodes[c]['connection_weight'] = {
                # If yes, then the next word is stored in the knowledge graph
                'word': caption[id+1],
                # And we extract the weight from the co_occurrence words matrix
                'weight': com_w[c][caption[id+1]]
            }
        except KeyError:  # If there is another word, but there are no weights, we set it to zero
            nodes[c]['connection_weight'] = {
                'word': caption[id+1],
                'weight': 0
            }
        except IndexError:  # If there is no other word, we just set it to None
            nodes[c]['connection_weight'] = None

    src_ids = []
    dst_ids = []
    dst_ids_ = None
    edge_weights = []
    node_embeddings = []
    # node_readable_desc = []

    for valid_word in list(nodes.keys()):
        node_embeddings_ = []
        # First, we define an id for the valid_word node
        # For the first valid_word, its id is one
        valid_word_id = 1 if len(src_ids) == 0 else dst_ids[-1]

        # Second, we create a temporary dest_ids list
        dst_ids_ = []
        # First set of destinations is emotion
        dst_ids_emotion = [(valid_word_id+id+1)
                           for id, _ in enumerate(get_cat_array())]

        # Second set of destinations is level-1 sentic descriptions
        dst_ids_sentic_level_1 = [x+dst_ids_emotion[-1] for x in range(1, 8)]
        # Third set of destinations is level-2 sentic descriptions
        dst_ids_sentic_level_2 = [
            x+dst_ids_sentic_level_1[-1] for x in range(1, 26)]
        dst_ids_.extend(dst_ids_emotion)
        dst_ids_.extend(dst_ids_sentic_level_1)
        # dst_ids_.extend(dst_ids_sentic_level_2)

        # Now we create a temporary source_ids list, which needs to be the same length as dest_ids lists
        # The first source_id is the valid_word node
        # The valid_word node is connected to the emotions and the first level of sentic descriptions
        src_ids_ = len(dst_ids_)*[valid_word_id]

        # The next source_ids will be the connections between the first level and second level sentic descriptions
        src_ids_.extend([item for sublist in [5*[x] for x in dst_ids_sentic_level_1[2:]]
                        for item in sublist])  # We reuse the same ids from the destination ids
        # And we also expand for the second-level descriptions
        dst_ids_.extend(dst_ids_sentic_level_2)

        # Finally, we create a temporary edge_weights list, which also needs to be the same length
        # The first set of weights is valid_word -> emotion
        edge_weights_ = []
        edge_weights_.extend(
            list(nodes[valid_word]['com_emotion']/np.sum(nodes[valid_word]['com_emotion'])))

        # The second set of weights is valid_word -> sentic descriptions
        # For this, we use the pleasantness and polarity values
        edge_weights_.extend([item for sublist in [
                             2*[float(nodes[valid_word]['sentic_descriptions']['pleasantness_value'])]] for item in sublist])
        edge_weights_.extend([item for sublist in [
                             5*[float(nodes[valid_word]['sentic_descriptions']['polarity_value'])]] for item in sublist])

        # The third set of weights is sentic_descriptions_level1 -> sentic_descriptions_level2
        # For this, we use the polarity of each word on the second node
        for semantics_ in ['semantics1', 'semantics2', 'semantics3', 'semantics4', 'semantics5']:
            edge_weights_.extend([item for sublist in [
                                 5*[float(nodes[valid_word]['sentic_descriptions'][semantics_]['word_semantics'][0])]] for item in sublist])

        # Finally, we extract the embeddings of each word and store it into a list
        node_embeddings_.append(nodes[valid_word]['glove'])
        # node_readable_desc.append('Glove embedding of valid word')

        node_embeddings_.extend([fetch_glove_embeddings(x)
                                for x in get_cat_array()])
        # node_readable_desc.extend([f'Embedding for {x}' for x in get_cat_array()])

        node_embeddings_.append(
            nodes[valid_word]['sentic_descriptions']['primary_mood_glove'])
        # node_readable_desc.append(nodes[valid_word]['sentic_descriptions']['primary_mood'])

        node_embeddings_.append(
            nodes[valid_word]['sentic_descriptions']['secondary_mood_glove'])
        # node_readable_desc.append(nodes[valid_word]['sentic_descriptions']['secondary_mood'])

        for semantics_ in ['semantics1', 'semantics2', 'semantics3', 'semantics4', 'semantics5']:
            node_embeddings_.append(
                nodes[valid_word]['sentic_descriptions'][semantics_]['root_word_glove'])
            # node_readable_desc.append(nodes[valid_word]['sentic_descriptions'][semantics_]['root_word'])

        for semantics_ in ['semantics1', 'semantics2', 'semantics3', 'semantics4', 'semantics5']:
            node_embeddings_.extend(
                nodes[valid_word]['sentic_descriptions'][semantics_]['word_semantics_glove'])
            # node_readable_desc.extend(nodes[valid_word]['sentic_descriptions'][semantics_]['word_semantics'][1:])

        assert len(dst_ids_) == len(src_ids_) == len(
            edge_weights_), "Length of nodes are not matching!"

        # Check if there is another node to connect
        if nodes[valid_word]['connection_weight'] is not None:
            src_ids_.append(valid_word_id)
            dst_ids_.append(dst_ids_[-1]+1)
            try:
                edge_weights_.append(
                    nodes[valid_word]['connection_weight']['weight']/com_w[valid_word]['length_'])
            except KeyError:
                edge_weights_.append(0)
            # node_readable_desc.append("Next valid word")

        # for s, d, e in zip(src_ids_, dst_ids_, edge_weights_):
            # print(f'Connecting {s} ({node_readable_desc[s-1]}) -> {d} ({node_readable_desc[d-1]}) with weight {e}')

        src_ids.extend(src_ids_)
        dst_ids.extend(dst_ids_)
        edge_weights.extend(edge_weights_)
        node_embeddings.extend(node_embeddings_)
    # Now, to build the graph and return it

    try:
        g = dgl.graph(
            (torch.tensor([x-1 for x in src_ids]), torch.tensor([x-1 for x in dst_ids])))
    except dgl.DGLError:
        print(f'Stopped with {caption}')
        import sys
        sys.exit(0)
    g.edata['w'] = torch.tensor(edge_weights)
    g.ndata['x'] = torch.tensor(node_embeddings, dtype=torch.double)
    return g


def main():
    com_e, com_w = get_com()
    g = generate_graph('baseball player swinge bat game', com_e, com_w)
    # print(g.device)
    '''G = dgl.to_networkx(g)
    plt.figure(figsize=[15*3,7*3])
    draw_options = {
        'node_color': 'red',
        'node_size': 20,
        'width': 1,
    }
    nx.draw_networkx(G, **draw_options)
    plt.show()'''


if __name__ == '__main__':
    main()
