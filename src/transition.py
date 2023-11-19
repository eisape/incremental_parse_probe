"""
Implementation of transition systems.

The TransitionSystem class is an "interface" for all of the
subclasses that are being used, but isn't really used anywhere
explicitly itself.
source: https://github.com/qipeng/arc-swift/blob/master/src/transition.py
"""
from smart_open import smart_open
import torch  
import random
import copy
import torch.nn as nn
from collections import defaultdict
import numpy as np

class ParserState_dec:
    def __init__(self, sentence = [None], transsys=None, goldrels=None):
        self.history = []
        self.action_tuples = []
        self.model_embeddings = torch.tensor([])
        self.log_prob = 0
        self.num_shifts = 0
        self.action_log_probs = []
        self.conditional_likelihood= []
        self.word_log_probs = []
        self.words = []
        self.expanded = False

        self.stack = [0]
        self.buf = [i+1 for i in range(len(sentence)-1)]
        # head and relation labels
        self.head = defaultdict(list) #[[-1, -1] for _ in range(len(sentence))]

        self.goldrels = goldrels

        self.transsys = transsys
        if self.transsys is not None:
            self.transsys._preparetransitionset(self)

        self.terminated = False
    
    def to_batch(self, probe):
        device = next(probe.parameters()).device
        gold_tuples = torch.tensor([t+[-1] for t in self.action_tuples]).unsqueeze(0).to(device)
        model_embeddings = self.model_embeddings.detach().clone().to(device)

        action_ids = [t[0] for t in self.action_tuples]

        if 'continuous_action_masks' in probe.args['data_sources']:
            mask =generate_continuous_mask(action_ids, model_embeddings.shape[2])#self.num_shifts+1)
            cont_mask = mask
            # cont_mask = np.pad(mask,
            #                     ((0, 400 - len(mask)),(0,0)),
            #                     'constant', constant_values=-1)
        else: cont_mask = torch.tensor([-1])

        return {'gold_tuples':gold_tuples,
                'padded_embeddings': model_embeddings,
                'action_ids':torch.tensor(action_ids).unsqueeze(0).to(device),
                'continuous_action_masks':torch.tensor(cont_mask).unsqueeze(0).to(device)} #tuples

    def heads_idxs(self): return [self.head[i][0] for i in sorted(self.head.keys())]

    def incremental_distance_matrix(self):
        sentence_length = len(self.heads_idxs()) #All observation fields must be of same length
        distances = torch.zeros((sentence_length, sentence_length))
        relative_depths = torch.zeros((sentence_length, sentence_length))
        for i in range(sentence_length):
            for j in range(i,sentence_length):
                # print(self.incremental_distance(i, j))
                i_j_distance,i_j_relative_depth = self.incremental_distance(i, j)
                distances[i][j] = i_j_distance
                distances[j][i] = i_j_distance

                relative_depths[i][j] = i_j_relative_depth
                relative_depths[j][i] = -i_j_relative_depth

        return distances, relative_depths

    def incremental_distance(self, i, j,unconnected_pad = 1):
        if i == j:
            return 0, 0
        # if observation:
        head_indices = []
        number_of_underscores = 0
        for elt in self.heads_idxs():
            # print(elt)
            if elt == '_':
                head_indices.append(0)
                number_of_underscores += 1
            else:
                head_indices.append(int(elt) + number_of_underscores)
        i_path = [i+1]
        j_path = [j+1]
        i_head = i+1
        j_head = j+1
        while True:
            if not (i_head == 0 and (i_path == [i+1] or i_path[-1] == 0)):
                i_head = head_indices[i_head - 1]
                i_path.append(i_head)
            if not (j_head == 0 and (j_path == [j+1] or j_path[-1] == 0)):
                j_head = head_indices[j_head - 1]
                j_path.append(j_head)
            if i_head in j_path:
                j_path_length = j_path.index(i_head)
                i_path_length = len(i_path) - 1
                
                break
            elif j_head in i_path:
                i_path_length = i_path.index(j_head)
                j_path_length = len(j_path) - 1
                break
            elif i_head == j_head:
                i_path_length = len(i_path) - 1
                j_path_length = len(j_path) - 1
                break
        
        total_length = j_path_length + i_path_length
        nodes_along_path = j_path[:j_path_length+1] + i_path[:i_path_length+1]

        if -1 in nodes_along_path:
            if unconnected_pad: total_length += unconnected_pad
            else: total_length = -1

        # if return_rel_depth:
        if -1 in nodes_along_path:
            return -1, float('inf')

        rel_depth = -(i_path_length - j_path_length) if not j_path_length == i_path_length else 0
        return total_length, rel_depth

    def transitionset(self):
        return self._transitionset

    def clone(self, clone_embeddings=True):
        res = ParserState_dec([])
        res.stack = copy.copy(self.stack)
        res.buf = copy.copy(self.buf)
        res.head = copy.copy(self.head)
        # res.pos = copy.copy(self.pos)
        res.goldrels = copy.copy(self.goldrels)
        res.transsys = self.transsys
        res.terminated = self.terminated
        res.action_tuples = copy.copy(self.action_tuples)
        res.log_prob = self.log_prob
        res.num_shifts = self.num_shifts
        res.action_log_probs = copy.copy(self.action_log_probs)
        res.conditional_likelihood = copy.deepcopy(self.conditional_likelihood)
        if clone_embeddings: res.model_embeddings = copy.deepcopy(self.model_embeddings)
        else: res.model_embeddings = []
        res.word_log_probs = copy.copy(self.word_log_probs)
        res.words = copy.copy(self.words)
        res.expanded = self.expanded
        res.history = copy.copy(self.history)
        
        if hasattr(self, '_transitionset'):
            res._transitionset = copy.copy(self._transitionset)
        return res

class ParserState:
    def __init__(self, sentence, transsys=None, goldrels=None):

        self.stack = [0]
        self.buf = [i+1 for i in range(len(sentence)-1)]
        self.head = [[-1, -1] for _ in range(len(sentence))]
        self.pos = [-1 for _ in range(len(sentence))]
        self.goldrels = goldrels
        self.transsys = transsys
        if self.transsys is not None: self.transsys._preparetransitionset(self)

    def transitionset(self): return self._transitionset

    def clone(self):
        res = ParserState([])
        res.stack = copy.copy(self.stack)
        res.buf = copy.copy(self.buf)
        res.head = copy.copy(self.head)
        res.pos = copy.copy(self.pos)
        res.goldrels = copy.copy(self.goldrels)
        res.transsys = self.transsys
        if hasattr(self, '_transitionset'):
            res._transitionset = copy.copy(self._transitionset)
        return res

class TransitionSystem(object):
    def __init__(self, mappings_file):
        self.mappings, self.invmappings = read_mappings(mappings_file, self.actions_list(), log=None)

    def _preparetransitionset(self, parserstate):
        """ Prepares the set of gold transitions given a parser state """
        raise NotImplementedError()

    def advance(self, parserstate, action):
        """ Advances a parser state given an action """
        raise NotImplementedError()

    def goldtransition(self, parserstate, goldrels):
        """ Returns the next gold transition given the set of gold arcs """
        raise NotImplementedError()

    def trans_to_str(self, transition, state, pos, fpos=None):
        raise NotImplementedError()

    @classmethod
    def trans_from_line(self, line):
        raise NotImplementedError()

    @classmethod
    def actions_list(self):
        raise NotImplementedError()

class ArcSwift(TransitionSystem):
    def __init__(self, mappings_file):
        self.mappings, self.invmappings = read_mappings(mappings_file, self.actions_list(), log=None)
        self.name='ASw'
    @classmethod
    def actions_list(self):
        return ['SHIFT', 'Left-Arc', 'Right-Arc']

    def _preparetransitionset(self, parserstate):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        stack, buf, head = parserstate.stack, parserstate.buf, parserstate.head

        t = []

        if len(buf) > 1:
            t += [(SHIFT, -1)]

        left_possible = False
        if len(buf) > 0:
            for si in range(len(stack) - 1):
                if head[stack[si]][0] < 0:
                    t += [(LEFTARC, si)]
                    left_possible = True
                    break
        if len(buf) > 1 or (len(buf) == 1 and not left_possible):
            for si in range(len(stack)):
                t += [(RIGHTARC, si)]
                if head[stack[si]][0] < 0:
                    break

        parserstate._transitionset = t

    def advance(self, parserstate, action):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        RELS = len(self.mappings['rel'])
        cand = parserstate.transitionset()

        if isinstance(action, int):
            a, rel = self.tuple_trans_from_int(cand, action)
        else:
            rel = action[-1]
            a = action[:-1]

        stack = parserstate.stack
        buf = parserstate.buf

        if a[0] == SHIFT:
            parserstate.stack = [buf[0]] + stack
            parserstate.buf = buf[1:]
        elif a[0] == LEFTARC:
            si = a[1]
            parserstate.head[stack[si]] = [buf[0], rel]
            parserstate.stack = stack[(si+1):]
        elif a[0] == RIGHTARC:
            si = a[1]
            parserstate.head[buf[0]] = [stack[si], rel]
            parserstate.stack = [buf[0]] + stack[si:]
            parserstate.buf = buf[1:]

        self._preparetransitionset(parserstate)

    def goldtransition(self, parserstate, goldrels=None, return_tuple=False):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        goldrels = goldrels or parserstate.goldrels
        stack = parserstate.stack
        buf = parserstate.buf
        head = parserstate.head

        j = buf[0]
        addedArc = False
        for n in range(len(stack)):
            if stack[n] in goldrels[j]:
                rel = goldrels[j][stack[n]]
                a = (LEFTARC, n, rel)
                addedArc = True
                
                break
            elif j in goldrels[stack[n]]:
                rel = goldrels[stack[n]][j]
                a = (RIGHTARC, n, rel)
                addedArc = True
                break
            if head[stack[n]][0] < 0: break

        if not addedArc:
            a = (SHIFT, -1, -1)
            if return_tuple:
            #this means we did |stack| comparisions and non of them succeeded
            #choice point, if we want to optimize for implicit action return full stack
                return a[0], buf[0], stack
            
        if return_tuple:
            #this means we did n comparisions and only the last on succeeded
            return a[0], buf[0], stack[:n]
        return a

    def trans_to_str(self, t, state, pos, fpos=None):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        if t[0] == SHIFT:
            if fpos is None:
                return "SHIFT\t%s" % (pos[state.buf[0]])
            else:
                return "SHIFT\t%s\t%s" % (pos[state.buf[0]], fpos[state.buf[0]])
        elif t[0] == LEFTARC:
            return "Left-Arc\t%d\t%s" % (t[1]+1, self.invmappings['rel'][t[2]])
        elif t[0] == RIGHTARC:
            if fpos is None:
                return "Right-Arc\t%d\t%s\t%s" % (t[1]+1, self.invmappings['rel'][t[2]], pos[state.buf[0]])
            else:
                return "Right-Arc\t%d\t%s\t%s\t%s" % (t[1]+1, self.invmappings['rel'][t[2]], pos[state.buf[0]], fpos[state.buf[0]])

    @classmethod
    def trans_from_line(self, line):
        if line[0] == 'Left-Arc':
            fields = { 'action':line[0], 'n':int(line[1]), 'rel':line[2] }
        elif line[0] == 'Right-Arc':
            fields = { 'action':line[0], 'n':int(line[1]), 'rel':line[2], 'pos':line[3] }
            if len(line) > 4:
                fields['fpos'] = line[4]
        elif line[0] == 'SHIFT':
            fields = { 'action':line[0], 'pos':line[1] }
            if len(line) > 2:
                fields['fpos'] = line[2]
        else:
            raise ValueError(line[0])
        return fields

    def tuple_trans_to_int(self, cand, t):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        RELS = len(self.mappings['rel'])

        base = 0
        if t[0] == SHIFT:
            return 0

        if cand[0][0] == SHIFT:
            base = 1

        if t[0] == LEFTARC:
            return base + t[2]

        if len(cand) > 1 and cand[1][0] == LEFTARC:
            base += RELS

        if t[0] == RIGHTARC:
            return base + t[1]*RELS + t[2]

    def tuple_trans_from_int(self, cand, action):
        SHIFT = self.mappings['action']['SHIFT']
        RELS = len(self.mappings['rel'])
        rel = -1

        if cand[0][0] == SHIFT:
            if action == 0:
                a = cand[0]
            else:
                a = cand[(action - 1) / RELS + 1]
                rel = (action - 1) % RELS
        else:
            a = cand[action / RELS]
            rel = action % RELS

        return a, rel

class ArcEagerReduce(TransitionSystem):
    def __init__(self, mappings_file):
        self.mappings, self.invmappings = read_mappings(mappings_file, self.actions_list(), log=None)
        self.name='AER'
        
    @classmethod
    def actions_list(self):
        return ['SHIFT', 'Left-Arc', 'Right-Arc', 'Reduce']

    def _preparetransitionset(self, parserstate):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        REDUCE = self.mappings['action']['Reduce']

        stack, buf, head = parserstate.stack, parserstate.buf, parserstate.head

        t = []

        if len(buf) > 1:
            t += [(SHIFT,)]

        if len(buf) > 0 and len(stack) > 1:
            t += [(REDUCE,)]

        left_possible = False
        if len(buf) > 0 and len(stack) > 1:
            if head[stack[0]][0] < 0:
                t += [(LEFTARC,)]
                left_possible = True

        if len(buf) > 1 or (len(buf) == 1 and not left_possible):
            t += [(RIGHTARC,)]

        parserstate._transitionset = t

    def advance(self, parserstate, action):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        REDUCE = self.mappings['action']['Reduce']

        RELS = len(self.mappings['rel'])
        cand = parserstate.transitionset()

        if isinstance(action, int):
            a, rel = self.tuple_trans_from_int(cand, action)
        else:
            rel = action[-1]
            a = action[:-1]

        stack = parserstate.stack
        buf = parserstate.buf

        if a[0] == SHIFT:
            parserstate.stack = [buf[0]] + stack
            parserstate.buf = buf[1:]
        elif a[0] == LEFTARC:
            parserstate.head[stack[0]] = [buf[0], rel]
            parserstate.stack = stack[1:]
        elif a[0] == RIGHTARC:
            parserstate.head[buf[0]] = [stack[0], rel]
            parserstate.stack = [buf[0]] + stack
            parserstate.buf = buf[1:]
        elif a[0] == REDUCE:
            parserstate.stack = stack[1:]

        self._preparetransitionset(parserstate)

    def goldtransition(self, parserstate, goldrels=None, return_tuple=False):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        REDUCE = self.mappings['action']['Reduce']

        goldrels = goldrels or parserstate.goldrels
        stack = parserstate.stack
        buf = parserstate.buf
        head = parserstate.head

        POS = len(self.mappings['pos'])

        j = buf[0]

        norightchildren = True
        for x in buf:
            if x in goldrels[stack[0]]:
                norightchildren = False
                break

        if stack[0] in goldrels[j]:
            rel = goldrels[j][stack[0]]
            a = (LEFTARC, rel)

            if return_tuple:
                return a[0], buf[0], stack[0]

        elif j in goldrels[stack[0]]:
            rel = goldrels[stack[0]][j]
            a = (RIGHTARC, rel)

            if return_tuple:
                return a[0], buf[0], stack[0]

        elif head[stack[0]][0] >= 0 and norightchildren:
            a = (REDUCE, -1)
            if return_tuple:
                return a[0], buf[0], stack[0]

        else:
            a = (SHIFT, -1)
            if return_tuple:
                return a[0], buf[0], stack[0]

        return a

    def trans_to_str(self, t, state, pos, fpos=None):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        REDUCE = self.mappings['action']['Reduce']
        if t[0] == SHIFT:
            if fpos is None:
                return "SHIFT\t%s" % (pos[state.buf[0]])
            else:
                return "SHIFT\t%s\t%s" % (pos[state.buf[0]], fpos[state.buf[0]])
        elif t[0] == LEFTARC:
            return "Left-Arc\t%s" % (self.invmappings['rel'][t[1]])
        elif t[0] == RIGHTARC:
            if fpos is None:
                return "Right-Arc\t%s\t%s" % (self.invmappings['rel'][t[1]], pos[state.buf[0]])
            else:
                return "Right-Arc\t%s\t%s\t%s" % (self.invmappings['rel'][t[1]], pos[state.buf[0]], fpos[state.buf[0]])
        elif t[0] == REDUCE:
            return "Reduce"

    @classmethod
    def trans_from_line(self, line):
        if line[0] == 'Left-Arc':
            fields = { 'action':line[0], 'rel':line[1] }
        elif line[0] == 'Right-Arc':
            fields = { 'action':line[0], 'rel':line[1], 'pos':line[2] }
            if len(line) > 3:
                fields['fpos'] = line[3]
        elif line[0] == 'SHIFT':
            fields = { 'action':line[0], 'pos':line[1] }
            if len(line) > 2:
                fields['fpos'] = line[2]
        elif line[0] == 'Reduce':
            fields = { 'action':line[0] }
        else:
            raise ValueError(line[0])
        return fields

    def tuple_trans_to_int(self, cand, t):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        REDUCE = self.mappings['action']['Reduce']

        RELS = len(self.mappings['rel'])

        base = 0
        if t[0] == SHIFT:
            return base

        base += 1

        if t[0] == REDUCE:
            return base

        base += 1

        if t[0] == LEFTARC:
            return base + t[1]

        base += RELS

        if t[0] == RIGHTARC:
            return base + t[1]

    def tuple_trans_from_int(self, cand, action):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        REDUCE = self.mappings['action']['Reduce']
        RELS = len(self.mappings['rel'])
        rel = -1

        base = 0
        if action == base:
            a = (SHIFT,)
        base += 1

        if action == base:
            a = (REDUCE,)
        base += 1

        if base <= action < base + RELS:
            a = (LEFTARC,)
            rel = action - base
        base += RELS

        if base <= action < base + RELS:
            a = (RIGHTARC,)
            rel = action - base

        return a, rel

class ArcEagerShift(ArcEagerReduce):
    def __init__(self, mappings_file):
        self.mappings, self.invmappings = read_mappings(mappings_file, self.actions_list(), log=None)
        self.name='AES'
        
    def goldtransition(self, parserstate, goldrels=None, return_tuple=False):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        REDUCE = self.mappings['action']['Reduce']

        goldrels = goldrels or parserstate.goldrels
        stack = parserstate.stack
        buf = parserstate.buf
        head = parserstate.head

        POS = len(self.mappings['pos'])

        j = buf[0]

        has_right_children = False
        for i in buf:
            if i in goldrels[stack[0]]:
                has_right_children = True
                break

        must_reduce = False
        for i in stack:
            if i in goldrels[j] or j in goldrels[i]:
                must_reduce = True
                break
            if head[i][0] < 0:
                break

        if stack[0] in goldrels[j]:
            rel = goldrels[j][stack[0]]
            a = (LEFTARC, rel)

            if return_tuple:
                return a[0], buf[0], stack[0]

        elif j in goldrels[stack[0]]:
            rel = goldrels[stack[0]][j]
            a = (RIGHTARC, rel)

            if return_tuple:
                return a[0], buf[0], stack[0]

        elif not must_reduce or head[stack[0]][0] < 0 or has_right_children:
            a = (SHIFT, -1)
            if return_tuple:
                #you can only be here if the comparisons failed (and of course someother things failed as well)
                return a[0], buf[0], stack[0]
        else:
            a = (REDUCE, -1)
            if return_tuple:
                #you can only be here if the comparisons failed (and of course someother things failed as well)
                return a[0], buf[0], stack[0]
        return a

class ArcStandard(TransitionSystem):
    def __init__(self, mappings_file):
        self.mappings, self.invmappings = read_mappings(mappings_file, self.actions_list(), log=None)
        self.name='ASd'
        self.num_actions = 3

        self.i2a = self.actions_list()
        self.i2a.extend(['BOS', 'EOS', 'PAD'])
        self.a2i = {i:self.i2a.index(i) for i in self.i2a}
    

    def action_dists(self, p_shift, marginal_p_reduce):
        p_reduce = (1-p_shift).unsqueeze(-1).log()+torch.concat((1-marginal_p_reduce.unsqueeze(-1), marginal_p_reduce.unsqueeze(-1)), -1).log()
        dists = torch.cat(((p_shift).unsqueeze(-1).log(), p_reduce), -1)
        return dists

    def initial_state(self):
        '''returns the initial state for beam search parsing
        blank parser state after one shift
        '''
        init_parserstate = ParserState_dec()

        init_parserstate.buf = [init_parserstate.num_shifts+1]
        self._preparetransitionset(init_parserstate)
        self.advance(init_parserstate, self.a2i['SHIFT'])
        init_parserstate.action_log_probs.append(0)
        init_parserstate.action_tuples = [[self.a2i['SHIFT'], -1, -1]]

        init_parserstate.buf = [init_parserstate.num_shifts+1]
        self._preparetransitionset(init_parserstate)
        return init_parserstate
    
    def targets_idxs(self, batch):
        '''
        Returns 2 np arrays of the form [[indx in batch],
                                       [index of first embedding],
                                       [index of second embedding]],

                                      [[imdex of the target action]]]
        
        indices of the gold actions in the batch'''

        tuples = batch['gold_tuples'].clone()

        tuples = tuples.roll(1, -1)
        tuples[:,:,0] = torch.arange(tuples.shape[0]).unsqueeze(1).repeat(1,tuples.shape[1])

        vector_comparisons = tuples[:,:,3] != -1
        
        oracle_action_idxs = tuples[vector_comparisons][:,[0,2,3,1]].transpose(1,0).cpu().numpy()

        return oracle_action_idxs[:-1], oracle_action_idxs[-1]

    @classmethod
    def actions_list(self):
        return ['SHIFT', 'Left-Arc', 'Right-Arc']

    def _preparetransitionset(self, parserstate):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        stack, buf, head = parserstate.stack, parserstate.buf, parserstate.head

        t = []

        if len(buf) > 0:
            t += [(SHIFT,)]

        if len(stack) > 2:
            t += [(LEFTARC,)]

        if len(stack) > 1:
            t += [(RIGHTARC,)]

        parserstate._transitionset = t

    def advance(self, parserstate, action):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        RELS = len(self.mappings['rel'])
        cand = parserstate.transitionset()

        if isinstance(action, int):
            a, rel = self.tuple_trans_from_int(cand, action)
        else:
            rel = action[-1]
            a = action[:-1]

        stack = parserstate.stack
        buf = parserstate.buf

        if a[0] == SHIFT:
            parserstate.stack = [buf[0]] + stack
            #new
            parserstate.head[buf[0]] = [-1, -1]
            parserstate.num_shifts += 1
            #
            parserstate.buf = buf[1:]
        elif a[0] == LEFTARC:
            parserstate.head[stack[1]] = [stack[0], rel]
            parserstate.stack = [stack[0]] + stack[2:]
        elif a[0] == RIGHTARC:
            parserstate.head[stack[0]] = [stack[1], rel]
            parserstate.stack = stack[1:]

        self._preparetransitionset(parserstate)

    def goldtransition(self, parserstate, goldrels=None, return_tuple=False):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        goldrels = goldrels or parserstate.goldrels
        stack = parserstate.stack
        buf = parserstate.buf
        head = parserstate.head

        POS = len(self.mappings['pos'])

        #this is a double check to make sure we dont reduce node that still have children in the future
        #hopefully this is just a hack and we dont need it    
        stack0_done = True
        for x in buf:
            if x in goldrels[stack[0]]:
                stack0_done = False
                break

        if len(stack) > 2 and stack[1] in goldrels[stack[0]]:
            rel = goldrels[stack[0]][stack[1]]
            a = (LEFTARC, rel)

            if return_tuple:
                return a[0], stack[0], stack[1]

        elif len(stack) > 1 and stack[0] in goldrels[stack[1]] and stack0_done:
            rel = goldrels[stack[1]][stack[0]]
            a = (RIGHTARC, rel)

            if return_tuple:
                return a[0], stack[0], stack[1]
                # return a[0], stack[1], stack[0]
        else:
            a = (SHIFT, -1)

            if return_tuple:
                #look at the non distance comparison triggers if neither ('or' statement) triggered it means its the distance comparisons fault
                if not len(stack) > 1:
                    return a[0], -1, -1

                else: 
                    return a[0], stack[0], stack[1]

                    # if random.randint(0, 1):
                    #     return a[0], stack[0], stack[1]
                    # else:
                    #     return a[0], stack[1], stack[0]
                #means we didnt actually compare anything (one of the disqualifies triggered), doesnt matter what the stack looks like 

        return a

    def trans_to_str(self, t, state, pos, fpos=None):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        if t[0] == SHIFT:
            if fpos is None:
                return "SHIFT\t%s" % (pos[state.buf[0]])
            else:
                try: 
                    return "SHIFT\t%s\t%s" % (pos[state.buf[0]], fpos[state.buf[0]])
                except: 
                    None

        elif t[0] == LEFTARC:
            return "Left-Arc\t%s" % (self.invmappings['rel'][t[1]])
        elif t[0] == RIGHTARC:
            return "Right-Arc\t%s" % (self.invmappings['rel'][t[1]])
            

    @classmethod
    def trans_from_line(self, line):
        if line[0] == 'Left-Arc':
            fields = { 'action':line[0], 'rel':line[1] }
        elif line[0] == 'Right-Arc':
            fields = { 'action':line[0], 'rel':line[1] }
        elif line[0] == 'SHIFT':
            fields = { 'action':line[0], 'pos':line[1] }
            if len(line) > 2:
                fields['fpos'] = line[2]
        else:
            raise ValueError(line[0])
        return fields

    def tuple_trans_to_int(self, cand, t):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        RELS = len(self.mappings['rel'])

        base = 0
        if t[0] == SHIFT:
            return base

        base += 1

        if t[0] == LEFTARC:
            return base + t[1]

        base += RELS

        if t[0] == RIGHTARC:
            return base + t[1]

    def tuple_trans_from_int(self, cand, action):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']
        RELS = len(self.mappings['rel'])
        rel = -1

        base = 0
        if action == base:
            a = (SHIFT,)
        base += 1

        if base <= action < base + RELS:
            a = (LEFTARC,)
            rel = action - base
        base += RELS

        if base <= action < base + RELS:
            a = (RIGHTARC,)
            rel = action - base

        return a, rel

class ArcHybrid(ArcStandard):
    def __init__(self, mappings_file):
        self.mappings, self.invmappings = read_mappings(mappings_file, self.actions_list(), log=None)
        self.name='AH'
        
    def _preparetransitionset(self, parserstate):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        stack, buf, head = parserstate.stack, parserstate.buf, parserstate.head

        t = []

        if len(buf) > 0:
            t += [(SHIFT,)]

        if len(buf) > 0 and len(stack) > 1 and head[stack[0]][0] < 0:
            t += [(LEFTARC,)]

        if len(stack) > 1 and head[stack[0]][0] < 0:
            t += [(RIGHTARC,)]

        parserstate._transitionset = t

    def advance(self, parserstate, action):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        RELS = len(self.mappings['rel'])
        cand = parserstate.transitionset()

        if isinstance(action, int):
            a, rel = self.tuple_trans_from_int(cand, action)
        else:
            rel = action[-1]
            a = action[:-1]

        stack = parserstate.stack
        buf = parserstate.buf

        if a[0] == SHIFT:
            parserstate.stack = [buf[0]] + stack
            parserstate.buf = buf[1:]
        elif a[0] == LEFTARC:
            parserstate.head[stack[0]] = [buf[0], rel]
            parserstate.stack = stack[1:]
        elif a[0] == RIGHTARC:
            parserstate.head[stack[0]] = [stack[1], rel]
            parserstate.stack = stack[1:]

        self._preparetransitionset(parserstate)

    def goldtransition(self, parserstate, goldrels=None, return_tuple=False):
        SHIFT = self.mappings['action']['SHIFT']
        LEFTARC = self.mappings['action']['Left-Arc']
        RIGHTARC = self.mappings['action']['Right-Arc']

        goldrels = goldrels or parserstate.goldrels
        stack = parserstate.stack
        buf = parserstate.buf
        head = parserstate.head

        POS = len(self.mappings['pos'])

        stack0_done = True
        for x in buf:
            if x in goldrels[stack[0]]:
                stack0_done = False
                break

        if len(buf) > 0 and stack[0] in goldrels[buf[0]]:
            rel = goldrels[buf[0]][stack[0]]
            a = (LEFTARC, rel)

            if return_tuple:
                #for LEFTARC, only buf[0] and stack[0] are used
                return a[0], buf[0], stack[0]

        elif len(stack) > 1 and stack[0] in goldrels[stack[1]] and stack0_done:
            rel = goldrels[stack[1]][stack[0]]
            a = (RIGHTARC, rel)

            if return_tuple:
                #for RIGHTARC, only stack[0] and stack[1] are used
                return a[0], stack[0], stack[1] 
        else:
            a = (SHIFT, -1)
            if return_tuple:
                if not (not (len(stack) > 1) or not (stack0_done)):
                    #for SHIFT all three are used (implictly)
                    return a[0], buf[0], stack[0], stack[1] 
                else: 
                #means we didnt actually compare anything (one of the disqualifies triggered), doesnt matter what the stack looks like
                    return a[0], -1, -1, -1

        return a

def read_mappings(mappings_file, actions_list, log=None):
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

    res['action'] = {k: i for i, k in enumerate(actions_list)}
    res2['action'] = actions_list

    return res, res2
    
def generate_continuous_mask(action_ids, token_pad):
    mask = []
    #i think we we missing the last embedding before
    #shouuldnt be word indec should be number of words
    wrd_indx = 1
    for indx,i in enumerate(action_ids):
        if i == 0: wrd_indx+=1
        mask.append([1]*wrd_indx + [0]*(token_pad-wrd_indx))
    return mask