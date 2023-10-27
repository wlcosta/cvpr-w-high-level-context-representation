'''
build_expansionnet_model.py
Created on 2023 02 22 12:00:45
Description: Builds the ExpansionNetV2 model, returning the calls for the main loop.

Author: Will <wlc2@cin.ufpe.br>
'''

import torch
from argparse import Namespace
import pickle
from PIL import Image as PIL_Image
import torchvision
import spacy

en = spacy.load('en_core_web_sm')
sw_spacy = en.Defaults.stop_words

from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.language_utils import convert_vector_idx2word

def build_expansionnet_model():
    model_dim = 512
    N_enc = 3
    N_dec = 3
    max_seq_len = 74
    load_path = 'models/rf_model.pth'
    beam_size = 5

    drop_args = Namespace(enc=0.0,
                            dec=0.0,
                            enc_input=0.0,
                            dec_input=0.0,
                            other=0.0)
    model_args = Namespace(model_dim=model_dim,
                            N_enc=N_enc,
                            N_dec=N_dec,
                            dropout=0.0,
                            drop_args=drop_args)
    with open('models/demo_coco_tokens.pickle', 'rb') as f:
            coco_tokens = pickle.load(f)
    print("Dictionary loaded ...")

    img_size = 384
    model = End_ExpansionNet_v2(swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                                swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
                                swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                                swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                                swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
                                swin_use_checkpoint=False,
                                final_swin_dim=1536,

                                d_model=model_args.model_dim, N_enc=model_args.N_enc,
                                N_dec=model_args.N_dec, num_heads=8, ff=2048,
                                num_exp_enc_list=[32, 64, 128, 256, 512],
                                num_exp_dec=16,
                                output_word2idx=coco_tokens['word2idx_dict'],
                                output_idx2word=coco_tokens['idx2word_list'],
                                max_seq_len=max_seq_len, drop_args=model_args.drop_args,
                                rank='cuda:0')
    model.to('cuda:0')
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded ...")

    return model, coco_tokens

def fetch_image_caption(impath, model, coco_tokens, img_size=384):
  transf_1 = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size))])
  transf_2 = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                              std=[0.229, 0.224, 0.225])])
  remove_words = ['man', 'woman', 'people', 'girl', 'boy']
  pil_image = PIL_Image.open(impath)
  if pil_image.mode != 'RGB':
      pil_image = PIL_Image.new("RGB", pil_image.size)
  preprocess_pil_image = transf_1(pil_image)
  tens_image_1 = torchvision.transforms.ToTensor()(preprocess_pil_image)
  tens_image_2 = transf_2(tens_image_1)
  img = tens_image_2.unsqueeze(0)
  img.to('cuda:0')

  beam_search_kwargs = {'beam_size': 5,
                              'beam_max_seq_len': 74,
                              'sample_or_max': 'max',
                              'how_many_outputs': 1,
                              'sos_idx': coco_tokens['word2idx_dict'][coco_tokens['sos_str']],
                              'eos_idx': coco_tokens['word2idx_dict'][coco_tokens['eos_str']]}
  with torch.no_grad():
      pred, _ = model(enc_x=img.to('cuda:0'),
                      enc_x_num_pads=[0],
                      mode='beam_search', **beam_search_kwargs)
  pred = convert_vector_idx2word(pred[0][0], coco_tokens['idx2word_list'])[1:-1]
  pred[-1] = pred[-1] + '.'
  pred = ' '.join(pred).capitalize()

  words = [word for word in pred.replace('.', '').split() if word.lower() not in sw_spacy]
  words = [word for word in words if word.lower() not in remove_words]
  new_text = " ".join(words)
  new_text = " ".join([token.lemma_ for token in en(new_text)])

  return pred, new_text

if __name__ == '__main__':
    model, coco_tokens = build_expansionnet_model()
    raw, caption = fetch_image_caption("/mnt/642E087C15C26D71/emoticram/emotic19/framesdb/images/frame_6m8xv1wil22n0dwz.jpg", model, coco_tokens)
    print(raw, caption)