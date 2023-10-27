'''
generate_graphs_to_disk.py
Created on 2023 02 27 18:41:50
Description: This file generates the graph binary files to disk.

Author: Will <wlc2@cin.ufpe.br>
'''

from datasets import generate_graph, get_com
import yaml
import os
import numpy as np
import dgl
from tqdm import tqdm


def main():
    com_e, com_w = get_com()
    cfg = yaml.safe_load(open('config.yaml', 'r'))
    dataset_save_dir = f"dataset/{cfg['dataset']['name']+cfg['dataset']['output_dir']}"

    caption_files = [x for x in os.listdir(dataset_save_dir) if 'caption' in x]
    for file in tqdm(caption_files, desc="Caption files"):
        dataset, label, _, state, _ = file.split('_')
        if state not in cfg['co_occ']['inputs']:
            print(f'[INFO] Skiping {file}.')
            continue
        captions = np.load(os.path.join(dataset_save_dir, file))
        graphs = []
        for caption in (pbar := tqdm(captions, position=1, leave=True)):
            pbar.set_description(f'{caption}')
            if caption == '':
                g = generate_graph('None', com_e, com_w)
            else:
                g = generate_graph(caption, com_e, com_w)
            g = dgl.add_self_loop(g)
            graphs.append(g)

        dgl.data.utils.save_graphs(
            f'{dataset_save_dir}/{dataset}_{label}_graph_{state}.bin', graphs)


if __name__ == '__main__':
    main()
