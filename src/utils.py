from smart_open import smart_open
import numpy as np
import torch
import random
import torch.nn.functional as F
from nltk import Tree
from collections import defaultdict
import h5py
from tqdm import tqdm
import json
import torch.nn as nn
from abc import ABC
import os
from queue import PriorityQueue
import yaml
import copy
import shutil
from itertools import count
from transition import ArcSwift, ArcEagerReduce, ArcEagerShift, ArcStandard, ArcHybrid
import transition
import yaml
import architectures
from scipy.stats import spearmanr, pearsonr

ignored_tags = ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]

MODEL_DATA = {'gpt2': {'layer_count': 13, 'feature_count': 768},
              'gpt2-medium': {'layer_count': 25, 'feature_count': 1024},
              'gpt2-large': {'layer_count': 37, 'feature_count': 1280},
              'gpt2-xl': {'layer_count': 49, 'feature_count': 1600},
              'bert-base-cased': {'layer_count': 13, 'feature_count': 768},
              'bert-large-cased': {'layer_count': 25, 'feature_count': 1024}}


def generate_continuous_mask(action_ids, token_pad):
    mask = []
    #i think we we missing the last embedding before
    #shouuldnt be word indec should be number of words
    wrd_indx = 1
    for indx,i in enumerate(action_ids):
        if i == 0: wrd_indx+=1
        mask.append([1]*wrd_indx + [0]*(token_pad-wrd_indx))
    return mask


def generate_lines_for_sent(lines):
    '''Yields batches of lines describing a sentence in conllx.
    Args:
        lines: Each line of a conllx file.
    Yields:
        a list of lines describing a single sentence in conllx.
    '''
    buf = []
    for line in lines:
        if line.startswith('#'):
            continue
        if not line.strip():
            if buf:
                yield buf
                buf = []
            else:
                continue
        else:
            buf.append(line.strip())
    if buf:
        yield buf

def clean_number(w):
    new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', w)
    return new_w

def conv_padded_ngrams(probe_vocab, 
                       action_ids, 
                       action_ngram_pad=30, 
                       token_pad=30, 
                       pad_token = -1):
    '''
    input:
      converts unpadded array of action id to padded array of padded action ngrams
      probe_vocab(dict): probe.a2i
      action_ids (array, cpu tensor): (len(action_ids),)
      action_ngram_pad (int): pad
      token_pad (int): pad
      pad_token (int): what int to pad with (should be probe.a2i[PAD])
    retuns:
      padded_action_ngrams (nparray): (token_pad x action_ngram_pad)
    '''
    #convert to numpy array
    arr_action_ids = np.array(action_ids)

    #boolean array is this action a shift?
    shift_bin = (arr_action_ids == probe_vocab['SHIFT'])

    #idxs of where shifts should happen - adds a shift at the end
    shift_ids = np.concatenate((np.nonzero(shift_bin)[0], [len(arr_action_ids)]))

    #action ngrams
    split_actions = np.split(arr_action_ids,shift_ids+1,0)[:-1]

    #remove trailing pad token
    split_actions[-1] = split_actions[-1][np.where(split_actions[-1] != probe_vocab['PAD'])]

    #pad ngrams and add special tokens
    padded_ngrams = np.array([np.concatenate(([probe_vocab['BOS']], i,[probe_vocab['EOS']] ,[probe_vocab['PAD']]*(action_ngram_pad-len(i)-2))) for i in split_actions])

    #pad ngram batch to token_pad
    padded_ngrams = np.concatenate((padded_ngrams, np.zeros((token_pad-len(padded_ngrams), action_ngram_pad)) + probe_vocab['PAD']),0)
    return padded_ngrams


def update_log(s):
    with open(args.logpath, 'a') as f:
        f.write(s + '\n')

def flatten_list(lst): return [j for sub in lst for j in sub]

def head_indxs_to_states(head_indxs,oracle):
    goldrels = [dict() for i in range(len(head_indxs)+1)]
    for tok, head in enumerate(head_indxs): goldrels[head][tok+1] = -1

    state = transition.ParserState_dec(["<ROOT>"] + head_indxs, transsys=oracle, goldrels=goldrels)
    full_states = []#[state.clone()]
    while len(state.transitionset()) > 0:

        goldtransition =oracle.goldtransition(state)
        state.action_tuples.append(list(oracle.goldtransition(state, return_tuple=True)))
        oracle.advance(state, goldtransition)
        full_states.append(state.clone())
        
    return full_states
    
def prune_queue(queue, k):
    pruned_queue = PriorityQueue()
    for i in range(k):
        if queue.qsize():
            g = queue.get() #g is a tuple (score, node)
            pruned_queue.put(g) 
    return pruned_queue

def clean_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)

def mkdir_ex(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
        
MODEL_DATA = {'gpt2': {'layer_count': 13, 'feature_count': 768},
              'gpt2-medium': {'layer_count': 25, 'feature_count': 1024},
              'gpt2-large': {'layer_count': 37, 'feature_count': 1280},
              'gpt2-xl': {'layer_count': 49, 'feature_count': 1600},
              'bert-base': {'layer_count': 13, 'feature_count': 768},
              'bert-large': {'layer_count': 25, 'feature_count': 1024}}

def oracle_lookup(k):
    lookup = {"ASw": ArcSwift,
              "AER": ArcEagerReduce,
              "AES": ArcEagerShift,
              "ASd": ArcStandard,
              "AH" : ArcHybrid,}
    return lookup[k]

class obs(object):
    def __init__(self, head_indices): self.head_indices = head_indices
    def __getitem__(self,index): return self.head_indices

class UnionFind:
  '''
  Naive UnionFind implementation for (slow) Prim's MST algorithm
  Used to compute minimum spanning trees for distance matrices
  '''
  def __init__(self, n):
    self.parents = list(range(n))
  def union(self, i,j):
    if self.find(i) != self.find(j):
      i_parent = self.find(i)
      self.parents[i_parent] = j
  def find(self, i):
    i_parent = i
    while True:
      if i_parent != self.parents[i_parent]:
        i_parent = self.parents[i_parent]
      else:
        break
    return i_parent

def prims_matrix_to_edges(matrix, poses):
  '''
  Constructs a minimum spanning tree from the pairwise weights in matrix;
  returns the edges.
  Never lets punctuation-tagged words be part of the tree.
  '''
  pairs_to_distances = {}
  uf = UnionFind(len(matrix))
  for i_index, line in enumerate(matrix):
    for j_index, dist in enumerate(line):
      if IDX2XPOS[poses[i_index].item()] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]:
        continue
      if IDX2XPOS[poses[j_index].item()] in ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]:
        continue
      pairs_to_distances[(i_index, j_index)] = dist
  edges = []
  for (i_index, j_index), distance in sorted(pairs_to_distances.items(), key = lambda x: x[1]):
    if uf.find(i_index) != uf.find(j_index):
      uf.union(i_index, j_index)
      edges.append((i_index, j_index))
  return edges

def get_nopunct_argmin(prediction, poses):
  '''
  Gets the argmin of predictions, but filters out all punctuation-POS-tagged words
  '''
  puncts = ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]
  original_argmin = np.argmin(prediction)
  for i in range(len(poses)):
    argmin = np.argmin(prediction)
    if IDX2XPOS[poses[argmin].item()] not in puncts:
      return argmin
    else:
      prediction[argmin] = np.inf
  return original_argmin

def heads_to_displacy(sentence, heads):
    displacy_format = {
        "words": [
            {"text": token, "tag": ' '} for token in sentence.split()
        ],
        "arcs": [
            {"start": dep, "end": head[0], "label": ' ', "dir": "right"} if dep < head[0] else {"start": head[0], "end": dep, "label": ' ', "dir": "left"} for dep, head in heads.items() if head[0] != -1
        ]
    }
    displacy_format["words"].insert(0, {"text": 'ROOT', "tag": ' '})
    return displacy_format

def load_lit_checkpoint(purpose,mod,probe_name,l):
    if os.path.isfile(f"./experiment_checkpoints/{purpose}/{mod}/{probe_name}/layer_{str(l)}/checkpoints/last.ckpt"):
        with open(f"./experiment_checkpoints/{purpose}/{mod}/{probe_name}/layer_{str(l)}/config.yaml", 'r') as file: l_args = yaml.safe_load(file)
        l_args['probe_params']['pretrained_model'] = l_args['pretrained_model']
        p_ckpt = experiment.IncrementalParseProbeExperiment.load_from_checkpoint(f"./experiment_checkpoints/{purpose}/{mod}/{probe_name}/layer_{str(l)}/checkpoints/last.ckpt").probe
        p = getattr(architectures, l_args['probe_params']['probe_type'])(l_args['probe_params']).to('cuda')
        p.load_state_dict(p_ckpt.state_dict())
        p.eval()
        p.oracle = transition.ArcStandard(l_args['probe_params']['oracle_params']['mappings_file'])
        return l_args, p
    else: return None, None

def berkeley_unk_conv(ws):
  """This is a simplified version of unknown token conversion in BerkeleyParser.

  The full version is berkely_unk_conv2.
  """
  uk = "unk"
  sz = len(ws) - 1
  if ws[0].isupper():
    uk = "c" + uk
  if ws[0].isdigit() and ws[sz].isdigit():
    uk = uk + "n"
  elif sz <= 2:
    pass
  elif ws[sz-2:sz+1] == "ing":
    uk = uk + "ing"
  elif ws[sz-1:sz+1] == "ed":
    uk = uk + "ed"
  elif ws[sz-1:sz+1] == "ly":
    uk = uk + "ly"
  elif ws[sz] == "s":
    uk = uk + "s"
  elif ws[sz-2:sz+1] == "est":
    uk = uk + "est"
  elif ws[sz-1:sz+1] == "er":
    uk = uk + 'ER'
  elif ws[sz-2:sz+1] == "ion":
    uk = uk + "ion"
  elif ws[sz-2:sz+1] == "ory":
    uk = uk + "ory"
  elif ws[0:2] == "un":
    uk = "un" + uk
  elif ws[sz-1:sz+1] == "al":
    uk = uk + "al"
  else:
    for i in range(sz):
      if ws[i] == '-':
        uk = uk + "-"
        break
      elif ws[i] == '.':
        uk = uk + "."
        break
  return "<" + uk + ">"

def berkeley_unk_conv2(token):
  numCaps = 0
  hasDigit = False
  hasDash = False
  hasLower = False
  for char in token:
    if char.isdigit():
      hasDigit = True
    elif char == '-':
      hasDash = True
    elif char.isalpha():
      if char.islower():
        hasLower = True
      elif char.isupper():
        numCaps += 1
  result = 'UNK'
  lower = token.rstrip().lower()
  ch0 = token.rstrip()[0]
  if ch0.isupper():
    if numCaps == 1:
      result = result + '-INITC'
      # Remove this because it relies on a vocabulary, not given to this funciton (HN).
      # if lower in words_dict:
      #   result = result + '-KNOWNLC'
    else:
      result = result + '-CAPS'
  elif not(ch0.isalpha()) and numCaps > 0:
    result = result + '-CAPS'
  elif hasLower:
    result = result + '-LC'
  if hasDigit:
    result = result + '-NUM'
  if hasDash:
    result = result + '-DASH'
  if lower[-1] == 's' and len(lower) >= 3:
    ch2 = lower[-2]
    if not(ch2 == 's') and not(ch2 == 'i') and not(ch2 == 'u'):
      result = result + '-s'
  elif len(lower) >= 5 and not(hasDash) and not(hasDigit and numCaps > 0):
    if lower[-2:] == 'ed':
      result = result + '-ed'
    elif lower[-3:] == 'ing':
      result = result + '-ing'
    elif lower[-3:] == 'ion':
      result = result + '-ion'
    elif lower[-2:] == 'er':
      result = result + '-er'
    elif lower[-3:] == 'est':
      result = result + '-est'
    elif lower[-2:] == 'ly':
      result = result + '-ly'
    elif lower[-3:] == 'ity':
      result = result + '-ity'
    elif lower[-1] == 'y':
      result = result + '-y'
    elif lower[-2:] == 'al':
      result = result + '-al'
  return result

import logging
import os.path as op
from smart_open import smart_open
# import cPickle as pickle
import pickle
from transition import ArcSwift, ArcEagerReduce, ArcEagerShift, ArcStandard, ArcHybrid
import numpy as np

from copy import copy

class ParserState:
    def __init__(self, sentence, transsys=None, goldrels=None):
#         print(sentence)
        self.stack = [0]
        # sentences should already have a <ROOT> symbol as the first token
#         print([i+1 for i in range(len(sentence)-1)])
        self.buf = [i+1 for i in range(len(sentence)-1)]
        # head and relation labels
        self.head = [[-1, -1] for _ in range(len(sentence))]

        self.pos = [-1 for _ in range(len(sentence))]

        self.goldrels = goldrels

        self.transsys = transsys
        if self.transsys is not None:
            self.transsys._preparetransitionset(self)

    def transitionset(self):
        return self._transitionset

    def clone(self):
        res = ParserState([])
        res.stack = copy(self.stack)
        res.buf = copy(self.buf)
        res.head = copy(self.head)
        res.pos = copy(self.pos)
        res.goldrels = copy(self.goldrels)
        res.transsys = self.transsys
        if hasattr(self, '_transitionset'):
            res._transitionset = copy(self._transitionset)
        return res


transition_dims = ['action', 'n', 'rel', 'pos', 'fpos']
transition_pos = {v:i for i, v in enumerate(transition_dims)}
floatX = np.float32

def transsys_lookup(k):
    lookup = {"ASw": ArcSwift,
              "AER": ArcEagerReduce,
              "AES": ArcEagerShift,
              "ASd": ArcStandard,
              "AH" : ArcHybrid,}
    return lookup[k]

def process_example(conll_lines, seq_lines, vocab, mappings, transsys, fpos=False, log=None):
    if fpos:
        res = [[] for _ in range(4)]
    else:
        res = [[] for _ in range(3)]
    res[0] = [vocab[u'<ROOT>']] + [vocab[u'<UNK>'] if line.split()[1] not in vocab else vocab[line.split()[1]] for line in conll_lines]
    for line in seq_lines:
        line = line.split()
        try:
            fields = transsys.trans_from_line(line)
        except ValueError as e:
            log.error('Encountered unknown transition type "%s" in sequences file, ignoring...' % (str(e)))
            return None

        vector_form = []
        for k in transition_dims:
            if k in fields:
                if k in mappings:
                    fields[k] = mappings[k][fields[k]]
                vector_form += [fields[k]]
            else:
                vector_form += [-1] # this should never be used

        res[1] += [vector_form]

    # gold POS
    res[2] = [len(mappings['pos'])] + [mappings['pos'][line.split()[3]] for line in conll_lines]
    if fpos:
        # fine-grained POS
        res[3] = [len(mappings['fpos'])] + [mappings['fpos'][line.split()[4]] for line in conll_lines]

    return tuple(res)

def read_mappings(mappings_file, transsys, log=None):
    i = 0
    res = dict()
    res2 = dict()
    with smart_open(mappings_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("::"):
                currentkey = line[2:]
                res[currentkey] = dict()
                res2[currentkey] = []
                i = 0
            else:
                res[currentkey][line] = i
                res2[currentkey] += [line]
                i += 1

    res['action'] = {k: i for i, k in enumerate(transsys.actions_list())}
    res2['action'] = transsys.actions_list()

    return res, res2

def read_gold_parserstates(fin, transsys, fpos=False):
    def processlines(lines):
        arcs = [dict() for i in range(len(lines)+1)]

        pos = ["" for i in range(len(lines)+1)]
        fpos = ["" for i in range(len(lines)+1)]

        for i, line in enumerate(lines):
            pos[i+1] = line[3] # fine-grained
            fpos[i+1] = line[4]
            parent = int(line[6])
            relation = line[7]
            arcs[parent][i+1] = transsys.mappings['rel'][relation]
#         print(ParserState(["<ROOT>"] + lines))

        res = [ParserState(["<ROOT>"] + lines, transsys=transsys, goldrels=arcs), pos]
        if fpos:
            res += [fpos]
        else:
            res == [None]
        return res
    res = []

    lines = []
    line = fin.readline()#.decode('utf-8')
    while line:
        line = line.strip().split()

        if len(line) == 0:
            res += [processlines(lines)]
            
            lines = []
        else:
            lines += [line]

        line = fin.readline()#.decode('utf-8')

    if len(lines) > 0:
        res += [processlines(lines)]
#         print(res[0][0].buf)

    return res

def write_gold_trans(tpl, fout):
    state, pos, fpos = tpl
    transsys = state.transsys
    while len(state.transitionset()) > 0:
        t = transsys.goldtransition(state)

        fout.write("%s\n" % transsys.trans_to_str(t, state, pos, fpos))

        transsys.advance(state, t)

    fout.write("\n")

def multi_argmin(lst):
    minval = 1e10
    res = []
    for i, v in enumerate(lst):
        if v < minval:
            minval = v
            res = [i]
        elif v == minval:
            res += [i]

    return res

XPOS2IDX = {'$': 0,'PRP$': 1,'VBZ': 2,'CD': 3,'JJS': 4,'VBG': 5,'IN': 6,'VB': 7,',': 8,'RB': 9,'JJ': 10,'LS': 11,'TO': 12,'UH': 13,'EX': 14,'``': 15,'SYM': 16,'NNP': 17,'WP': 18,'.': 19,"''": 20,'VBP': 21,'WP$': 22,'-RRB-': 23,'-LRB-': 24,'PDT': 25,'PRP': 26,'NNS': 27,':': 28,'WDT': 29,'POS': 30,'MD': 31,'RBS': 32,'RP': 33,'VBN': 34,'CC': 35,'NNPS': 36,'JJR': 37,'RBR': 38,'DT': 39,'WRB': 40,'NN': 41,'FW': 42,'VBD': 43,'#': 44}
IDX2XPOS = {v: k for k, v in XPOS2IDX.items()}