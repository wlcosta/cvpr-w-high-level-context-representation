import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import csv
import dgl
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

class EmoticDataset(Dataset):
    def __init__(self, cat, cont):
        super(EmoticDataset, self).__init__()
        self.cat = cat
        self.cont = cont

    def __len__(self):
        return len(self.cat)
    
    def __getitem__(self, index):
        cat_label = self.cat[index]
        cont_label = self.cont[index]
        
        return torch.tensor(cat_label, dtype=torch.float32), torch.tensor(cont_label, dtype=torch.float32)/10.0, index


class EmoticTrain:
    def __init__(self, filename, folder, image_size, person):
        self.filename = filename
        self.folder = folder
        self.im_size = []
        self.bbox = []
        self.cat = []
        self.cont = []
        self.gender = person[3][0]
        self.age = person[4][0]
        self.cat_annotators = 0
        self.cont_annotators = 0
        self.set_imsize(image_size)
        self.set_bbox(person[0])
        self.set_cat(person[1])
        self.set_cont(person[2])
        self.check_cont()

    def set_imsize(self, image_size):
        image_size = np.array(image_size).flatten().tolist()[0]
        row = np.array(image_size[0]).flatten().tolist()[0]
        col = np.array(image_size[1]).flatten().tolist()[0]
        self.im_size.append(row)
        self.im_size.append(col)

    def validate_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        x1 = min(self.im_size[0], max(0, x1))
        x2 = min(self.im_size[0], max(0, x2))
        y1 = min(self.im_size[1], max(0, y1))
        y2 = min(self.im_size[1], max(0, y2))
        return [int(x1), int(y1), int(x2), int(y2)]

    def set_bbox(self, person_bbox):
        self.bbox = self.validate_bbox(np.array(person_bbox).flatten().tolist())

    def set_cat(self, person_cat):
        cat = np.array(person_cat).flatten().tolist()
        cat = np.array(cat[0]).flatten().tolist()
        self.cat = [np.array(c).flatten().tolist()[0] for c in cat]
        self.cat_annotators = 1

    def set_cont(self, person_cont):
        cont = np.array(person_cont).flatten().tolist()[0]
        self.cont = [np.array(c).flatten().tolist()[0] for c in cont]
        self.cont_annotators = 1

    def check_cont(self):
        for c in self.cont:
            if np.isnan(c):
                self.cont_annotators = 0
                break

class EmoticTest:
    def __init__(self, filename, folder, image_size, person):
        self.filename = filename
        self.folder = folder
        self.im_size = []
        self.bbox = []
        self.cat = []
        self.cat_annotators = 0
        self.comb_cat = []
        self.cont_annotators = 0
        self.cont = []
        self.comb_cont = []
        self.gender = person[5][0]
        self.age = person[6][0]

        self.set_imsize(image_size)
        self.set_bbox(person[0])
        self.set_cat(person[1])
        self.set_comb_cat(person[2])
        self.set_cont(person[3])
        self.set_comb_cont(person[4])
        self.check_cont()

    def set_imsize(self, image_size):
        image_size = np.array(image_size).flatten().tolist()[0]
        row = np.array(image_size[0]).flatten().tolist()[0]
        col = np.array(image_size[1]).flatten().tolist()[0]
        self.im_size.append(row)
        self.im_size.append(col)

    def validate_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        x1 = min(self.im_size[0], max(0, x1))
        x2 = min(self.im_size[0], max(0, x2))
        y1 = min(self.im_size[1], max(0, y1))
        y2 = min(self.im_size[1], max(0, y2))
        return [int(x1), int(y1), int(x2), int(y2)]

    def set_bbox(self, person_bbox):
        self.bbox = self.validate_bbox(np.array(person_bbox).flatten().tolist())

    def set_cat(self, person_cat):
        self.cat_annotators = len(person_cat[0])
        for ann in range(self.cat_annotators):
            ann_cat = person_cat[0][ann]
            ann_cat = np.array(ann_cat).flatten().tolist()
            ann_cat = np.array(ann_cat[0]).flatten().tolist()
            ann_cat = [np.array(c).flatten().tolist()[0] for c in ann_cat]
            self.cat.append(ann_cat)

    def set_comb_cat(self, person_comb_cat):
        if self.cat_annotators != 0:
            self.comb_cat = [np.array(c).flatten().tolist()[0] for c in person_comb_cat[0]]
        else:
            self.comb_cat = []

    def set_comb_cont(self, person_comb_cont):
        if self.cont_annotators != 0:
            comb_cont = [np.array(c).flatten().tolist()[0] for c in person_comb_cont[0]]
            self.comb_cont = [np.array(c).flatten().tolist()[0] for c in comb_cont[0]]
        else:
            self.comb_cont = []

    def set_cont(self, person_cont):
        self.cont_annotators = len(person_cont[0])
        for ann in range(self.cont_annotators):
            ann_cont = person_cont[0][ann]
            ann_cont = np.array(ann_cont).flatten().tolist()
            ann_cont = np.array(ann_cont[0]).flatten().tolist()
            ann_cont = [np.array(c).flatten().tolist()[0] for c in ann_cont]
            self.cont.append(ann_cont)

    def check_cont(self):
        for c in self.comb_cont:
            if np.isnan(c):
                self.cont_annotators = 0
                break

def test_scikit_ap(cat_preds, cat_labels, ind2cat):
  ''' Calculate average precision per emotion category using sklearn library.
  :param cat_preds: Categorical emotion predictions. 
  :param cat_labels: Categorical emotion labels. 
  :param ind2cat: Dictionary converting integer index to categorical emotion.
  :return: Numpy array containing average precision per emotion category.
  '''
  ap = np.zeros(26, dtype=np.float32)
  for i in range(26):
    ap[i] = average_precision_score(cat_labels[i, :], cat_preds[i, :])
    #print ('Category %16s %.5f' %(ind2cat[i], ap[i]))
  #print ('Mean AP %.5f' %(ap.mean()))
  return ap.mean()

def test_vad(cont_preds, cont_labels, ind2vad):
  ''' Calcaulate VAD (valence, arousal, dominance) errors. 
  :param cont_preds: Continuous emotion predictions. 
  :param cont_labels: Continuous emotion labels. 
  :param ind2vad: Dictionary converting integer index to continuous emotion dimension (Valence, Arousal and Dominance).
  :return: Numpy array containing mean absolute error per continuous emotion dimension. 
  '''
  vad = np.zeros(3, dtype=np.float32)
  for i in range(3):
    vad[i] = np.mean(np.abs(cont_preds[i, :] - cont_labels[i, :]))
    #print ('Continuous %10s %.5f' %(ind2vad[i], vad[i]))
  #print ('Mean VAD Error %.5f' %(vad.mean()))
  return vad.mean()

def get_cat_array():
    return ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
       'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
       'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']

cat2ind = {}
ind2cat = {}
for idx, emotion in enumerate(get_cat_array()):
    cat2ind[emotion] = idx
    ind2cat[idx] = emotion

vad = ['Valence', 'Arousal', 'Dominance']
ind2vad = {}
for idx, continuous in enumerate(vad):
    ind2vad[idx] = continuous

def cat_to_one_hot(y_cat):
    '''
    One hot encode a categorical label. 
    :param y_cat: Categorical label.
    :return: One hot encoded categorical label. 
    '''
    cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection',
       'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear',
       'Happiness', 'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
    cat2ind = {}
    ind2cat = {}
    for idx, emotion in enumerate(cat):
        cat2ind[emotion] = idx
        ind2cat[idx] = emotion
    one_hot_cat = np.zeros(26)
    for em in y_cat:
        one_hot_cat[cat2ind[em]] = 1
    return one_hot_cat