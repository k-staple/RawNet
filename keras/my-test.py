#should be run from keras folder in RawNet
#use cosine similarity on embeddings

#mine
import collections

import os
import numpy as np
np.random.seed(1016)
import yaml
import queue
import struct
import pickle as pk
from multiprocessing import Process
from threading import Thread
from tqdm import tqdm
from time import sleep
from keras.utils import multi_gpu_model, plot_model, to_categorical
from keras.optimizers import *
from keras.models import Model
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from model_RawNet_pre_train import get_model as get_model_pretrn
from model_RawNet import get_model

#for pickle part
#in keras folder of RawNet
fileA = '../../VGG-Speaker-Recognition/media/weidi/2TB-2/datasets/voxceleb1/wav/id10270/x6uYqmx31kE/00001.wav'
fileB = '../../VGG-Speaker-Recognition/media/weidi/2TB-2/datasets/voxceleb1/wav/id10270/8jEAjG6SegY/00008.wav'
testFile = '../../VGG-Speaker-Recognition/meta/testing_short_voxceleb1_veri_test.txt'

dataA = open('data/speaker_embeddings_RawNet_4.8eer', 'rb')


#RawNet lines 01-trn_RawNet.py functions, evaluate! part, and varaibles necessary for evaluate!
def cos_sim(a,b):
  return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def simple_loss(y_true, y_pred):
  return K.mean(y_pred)

def zero_loss(y_true, y_pred):
  return 0.5 * K.sum(y_pred, axis=0)

def compose_spkFeat_dic(lines, model, f_desc_dic, base_dir):
  '''
  Extracts speaker embeddings from a given model
  =====
  lines: (list) A list of strings that indicate each utterance
  model: (keras model) DNN that extracts speaker embeddings,
  output layer should be rmoved(model_pred)
  f_desc_dic: (dictionary) A dictionary of file objects
  '''
  dic_spkFeat = {}
  for line in tqdm(lines, desc='extracting spk feats'):
#########
    print(line)
########
    k, f, p = line.strip().split(' ')
    p = int(p)
    if f not in f_desc_dic:
      # f_tmp = '/'.join([base_dir, f])
      f_tmp = f
      f_desc_dic[f] = open(f_tmp, 'rb')

    f_desc_dic[f].seek(p)
    l = struct.unpack('i', f_desc_dic[f].read(4))[0]# number of samples of each utterance
    utt = np.asarray(struct.unpack('%df'%l, f_desc_dic[f].read(l * 4)), dtype=np.float32)# read binary utterance 
    spkFeat = model.predict(utt.reshape(1,-1,1))[0]# extract speaker embedding from utt
    dic_spkFeat[k] = spkFeat

  return dic_spkFeat

def make_spkdic(lines):
  '''
  Returns a dictionary where
  key: (str) speaker name
  value: (int) unique integer for each speaker
  '''
  idx = 0
  dic_spk = {}
  list_spk = []
  for line in lines:
    k, f, p = line.strip().split(' ')
    spk = k.split('/')[0]
    if spk not in dic_spk:
      dic_spk[spk] = idx
      list_spk.append(spk)
      idx += 1
  return (dic_spk, list_spk)

def compose_batch(lines, f_desc_dic, dic_spk, nb_samp, base_dir):
  '''
  Compose one mini-batch using utterances in `lines'
  nb_samp: (int) duration of utterance at train phase.
    Fixed for each mini-batch for mini-batch training.
  '''
  batch = []
  ans = []
  for line in lines:
    k, f, p = line.strip().split(' ')
    ans.append(dic_spk[k.split('/')[0]])
    p = int(p)
    if f not in f_desc_dic:
      f_tmp = '/'.join([base_dir, f])
      f_desc_dic[f] = open(f_tmp, 'rb')

    f_desc_dic[f].seek(p)
    l = struct.unpack('i', f_desc_dic[f].read(4))[0]
    utt = struct.unpack('%df'%l, f_desc_dic[f].read(l * 4))
    _nb_samp = len(utt)
		#need to verify this part later!!!!!!
    assert _nb_samp >= nb_samp
    cut = np.random.randint(low = 0, high = _nb_samp - nb_samp)
    utt = utt[cut:cut+nb_samp]
    batch.append(utt)

  return (np.asarray(batch, dtype=np.float32).reshape(len(lines), -1, 1), np.asarray(ans))

def process_epoch(lines, q, batch_size, nb_samp, dic_spk, base_dir): 
  '''
  Wrapper function for processing mini-batches for the train set once.
  '''
  f_desc_dic = {}
  nb_batch = int(len(lines) / batch_size)
  for i in range(nb_batch):
    while True:
      if q.full():
        sleep(0.1)
      else:
        q.put(compose_batch(lines = lines[i*batch_size: (i+1)*batch_size],
          f_desc_dic = f_desc_dic,
          dic_spk = dic_spk,
          nb_samp = nb_samp,
          base_dir = base_dir))
        break

  for k in f_desc_dic.keys():
    f_desc_dic[k].close()

  return
		

#======================================================================#
#======================================================================#
if __name__ == '__main__':


###########
  #include parts form the beginning
  #hardcode __file__ to a file in same directory (keras) since colab doesn't load file path variable
  sameDirFile = "01-trn_RawNet.py"
  _abspath = os.path.abspath(sameDirFile)
  dir_yaml = os.path.splitext(_abspath)[0] + '.yaml'
  with open(dir_yaml, 'r') as f_yaml:
    parser = yaml.load(f_yaml)


  
  save_dir = ''
  with open(testFile, 'r') as f_dev_scp:
    dev_lines = f_dev_scp.readlines()
  dic_spk, list_spk = make_spkdic(dev_lines)
  parser['model']['nb_spk'] = len(list_spk)
 
  f_eer = open(save_dir + 'eers_pretrn.txt', 'w', buffering=1)
##########



  #make speaker embeddings and compute cos_sim
  #include variables needed for lower section I copied from RawNet
###########
  eval_lines = open(parser['eval_scp'], 'r').readlines() #NOT testFile with bool audio1.wav audio2.wav; should be from processed wav from step 2 recreate environment in README
  trials = open(testFile, 'r').readlines() 
  
  #runs the file model_RawNet_pre_train.py since func is imported from there
  model, m_name = get_model_pretrn(argDic = parser['model'])
  print('m_name')
  print(m_name)
  print('after m_name')
  model_pred = Model(inputs=model.get_layer('input_pretrn').input, outputs=model.get_layer('code_pretrn').output)

#PPPPP parser['base_dir'] = '' #base dir is here

  epoch = 1
  #have to define?
  #?? dic_eval[spkMd], dic_eval[utt]
  #f_err defined & opened above
###########


  ####to deal for now with 2 indents of code copied from 01-trn_RawNet.py
  loopVar = 0
  while(loopVar < 1):
    loopVar = 1 
  #####
    #evaluate! #print('{score} {target}\n'.format(score=cos_score,target=target))
    dic_eval = compose_spkFeat_dic(lines = eval_lines, model = model_pred, f_desc_dic = {}, base_dir = parser['base_dir'])
	
    #f_res = open(save_dir + 'results_pretrn/epoch%s.txt'%(epoch), 'w') 
    y = []
    y_score = []
    for smpl in trials:
      target, spkMd, utt = smpl.strip().split(' ')
      target = int(target)
      cos_score = cos_sim(dic_eval[spkMd], dic_eval[utt])
      y.append(target)
      y_score.append(cos_score)
      print('{score} {target}\n'.format(score=cos_score,target=target))
      #f_res.write('{score} {target}\n'.format(score=cos_score,target=target))
    #f_res.close()
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    print('\nepoch: %d, eer: %f'%(int(epoch), eer))
    f_eer.write('%f\n'%(eer))

    if not bool(parser['save_best_only']):
      model.save_weights(save_dir +  'models_pretrn/%d-%.4f.h5'%(epoch, eer))
  f_eer.close()  
