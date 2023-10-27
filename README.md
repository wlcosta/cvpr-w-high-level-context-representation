## High-Level Context Representation for Emotion Recognition in Images

This repository contains the PyTorch implementation of our paper [High-Level Context Representation for Emotion Recognition in Images](https://openaccess.thecvf.com/content/CVPR2023W/LatinX/html/de_Lima_Costa_High-Level_Context_Representation_for_Emotion_Recognition_in_Images_CVPRW_2023_paper.html).

    @inproceedings{costa2023high,
      title={High-Level Context Representation for Emotion Recognition in Images},
      author={de Lima Costa, Willams and Talavera, Estefania and Figueiredo, Lucas Silva and Teichrieb, Veronica},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
      pages={326--334},
      year={2023}
    }

#### Dependencies

- PyTorch (Tested with 1.13.1 on Windows)
- dgl
- NLTK
- Mediapipe (for facial region processing)
- OpenCV, PyYAML and tqdm

### Preparation

- Download the [EMOTIC dataset](https://s3.sunai.uoc.edu/emotic/index.html) (2019 version)
- Download [GloVe](https://nlp.stanford.edu/projects/glove/) and [SenticNet](http://sentic.net/senticnet-5.0.zip), placing them into their respectie folders.
- Download the `rf-model.pth` file from [ExpansionNet](https://github.com/jchenghu/ExpansionNet_v2) and place it into the `model` folder.

1. Run `generate_dataset_annotations.py` to generate npy files with the captioning of each image.
2. Run `generate_co_occurrences_matrices.py` to generate the co-occurrency of emotions and words for each caption generated on (1).
3. Run `generate_graphs_to_disk.py` to generate the graphs and save them as a binary file.

#### Training

Use `python train.py` to train and test the model on the processed dataset.