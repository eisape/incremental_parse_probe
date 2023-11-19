#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create data files
source: https://github.com/aistairc/rnng-pytorch/blob/master/preprocess.py
"""

import os
import sys
import argparse
import itertools
from collections import defaultdict
import utils as utils
import re
import shutil
import json
from multiprocessing import Pool
import itertools

from transition import (
    ArcSwift,
    ArcEagerReduce,
    ArcEagerShift,
    ArcStandard,
    ArcHybrid,
    ParserState_dec,
)

import json

pad = "<pad>"
unk = "<unk>"


class Vocabulary(object):
    """
    This vocabulary prohibits registering a new token during lookup.
    Vocabulary should be constructed from a set of tokens with counts (w2c), a dictionary
    from a word to its count in the training data. (or anything)
    """

    def __init__(
        self, w2c_list, pad="<pad>", unkmethod="unk", unktoken="<unk>", specials=[]
    ):
        self.pad = pad
        self.padding_idx = 0
        self.specials = specials
        self.unkmethod = unkmethod
        self.unktoken = unktoken
        if self.unkmethod == "unk":
            if unktoken not in specials:
                specials.append(unktoken)

        assert isinstance(w2c_list, list)
        self.i2w = [self.pad] + specials + [w for w, _ in w2c_list]
        self.w2i = dict([(w, i) for i, w in enumerate(self.i2w)])
        self.w2c = dict(w2c_list)
        self.i2c = dict([(self.w2i[w], c) for w, c in self.w2c.items()])

        if self.unkmethod == "unk":
            self.unk_id = self.w2i[self.unktoken]

    def id_to_word(self, i):
        return self.i2w[i]

    def to_unk(self, w):
        if self.unkmethod == "unk":
            return self.unktoken
        elif self.unkmethod == "berkeleyrule":
            return utils.berkeley_unk_conv(w)
        elif self.unkmethod == "berkeleyrule2":
            return utils.berkeley_unk_conv2(w)

    def to_unk_id(self, w_id):
        if self.unkmethod == "unk":
            return self.unk_id
        else:
            if 1 <= w_id < 1 + len(self.specials):
                return w_id
            else:
                return self.get_id(utils.berkeley_unk_conv(self.i2w[w_id]))

    def size(self):
        return len(self.i2w)

    def get_id(self, w):
        if w not in self.w2i:
            w = self.to_unk(w)
            if w not in self.w2i:
                # Back-off to a general unk token when converted unk-token is not registered in the
                # vocabulary (which happens when an unseen unk token is generated at test time).
                w = self.unktoken
        return self.w2i[w]

    def get_count_from_id(self, w_id):
        if w_id not in self.i2c:
            return 0
        else:
            return self.i2c[w_id]

    def get_count(self, w):
        if w not in self.w2c:
            return 0
        else:
            return self.w2c[w]

    # for serialization
    def list_w2c(self):
        return [(w, self.get_count(w)) for w in self.i2w[1 + len(self.specials) :]]

    def dump(self, fn):
        with open(fn, "wt") as o:
            o.write(self.pad + "\n")
            o.write(self.unkmethod + "\n")
            o.write(self.unktoken + "\n")
            o.write(" ".join(self.specials) + "\n")
            for w, c in self.list_w2c():
                o.write("{}\t{}\n".format(w, c))

    def to_json_dict(self):
        return {
            "pad": self.pad,
            "unkmethod": self.unkmethod,
            "unktoken": self.unktoken,
            "specials": self.specials,
            "word_count": self.list_w2c(),
        }

    @staticmethod
    def load(self, fn):
        with open(fn) as f:
            lines = [line for line in f]
        pad, unkmethod, unktoken, specials = [l.strip() for l in line[:4]]
        specials = [w for w in specials]

        def parse_line(line):
            w, c = line[:-1].split()
            return w, int(c)

        w2c_list = [parse_line(line) for line in lines[4:]]
        return Vocabulary(w2c_list, pad, unkmethod, unktoken, specials)

    @staticmethod
    def from_data_json(data):
        d = data["vocab"]
        return Vocabulary(
            d["word_count"], d["pad"], d["unkmethod"], d["unktoken"], d["specials"]
        )


def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1) :]:
        if char == "(":
            return True
        elif char == ")":
            return False
    raise IndexError(
        "Bracket possibly not balanced, open bracket not followed by closed bracket"
    )


def get_next_bracket_index(line, start_idx):
    for i in range(start_idx + 1, len(line)):
        char = line[i]
        if char == "(" or char == ")":
            return i
    raise IndexError(
        "Bracket possibly not balanced, open bracket not followed by closed bracket"
    )


def get_between_brackets(line, start_idx):
    output = []
    for char in line[(start_idx + 1) :]:
        if char == ")":
            break
        assert not (char == "(")
        output.append(char)
    return "".join(output)


def get_tags_tokens_lowercase(line):
    output = []
    line = line.rstrip()
    for i in range(len(line)):
        if i == 0:
            assert line[i] == "("
        if line[i] == "(" and not (
            is_next_open_bracket(line, i)
        ):  # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line, i))
    # print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        # print(terminal, terminal_split)
        assert len(terminal_split) == 2  # each terminal contains a POS tag and word
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]


def transform_to_subword_tree(line, sp):
    line = line.rstrip()
    tags, tokens, _ = get_tags_tokens_lowercase(line)
    pieces = sp.encode(" ".join(tokens), out_type=str)
    end_idxs = [i + 1 for i, p in enumerate(pieces) if "▁" in p]
    begin_idxs = [0] + end_idxs[:-1]
    spans = list(
        zip(begin_idxs, end_idxs)
    )  # map from original token idx to piece span idxs.

    def get_piece_preterms(tok_i):
        tag = tags[tok_i]
        b, e = spans[tok_i]
        span_pieces = pieces[b:e]
        return " ".join(["({} {})".format(tag, p) for p in span_pieces])

    new_preterms = [get_piece_preterms(i) for i in range(len(tokens))]
    orig_token_spans = []
    for i in range(len(line)):
        if line[i] == "(":
            next_bracket_idx = get_next_bracket_index(line, i)
            found_bracket = line[next_bracket_idx]
            if found_bracket == "(":
                continue  # not terminal -> skip
            orig_token_spans.append((i, next_bracket_idx + 1))
    assert len(new_preterms) == len(orig_token_spans)
    ex_span_ends = [span[0] for span in orig_token_spans] + [len(line)]
    ex_span_begins = [0] + [span[1] for span in orig_token_spans]
    parts = []
    for i in range(len(new_preterms)):
        parts.append(line[ex_span_begins[i] : ex_span_ends[i]])
        parts.append(new_preterms[i])
    parts.append(line[ex_span_begins[i + 1] : ex_span_ends[i + 1]])
    return "".join(parts)


def get_nonterminal(line, start_idx):
    assert line[start_idx] == "("  # make sure it's an open bracket
    output = []
    for char in line[(start_idx + 1) :]:
        if char == " ":
            break
        assert not (char == "(") and not (char == ")")
        output.append(char)
    return "".join(output)


def get_actions(line):
    output_actions = []
    line_strip = line.rstrip()
    i = 0
    max_idx = len(line_strip) - 1
    while i <= max_idx:
        assert line_strip[i] == "(" or line_strip[i] == ")"
        if line_strip[i] == "(":
            if is_next_open_bracket(line_strip, i):  # open non-terminal
                curr_NT = get_nonterminal(line_strip, i)
                output_actions.append("NT(" + curr_NT + ")")
                i += 1
                while (
                    line_strip[i] != "("
                ):  # get the next open bracket, which may be a terminal or another non-terminal
                    i += 1
            else:  # it's a terminal symbol
                output_actions.append("SHIFT")
                while line_strip[i] != ")":
                    i += 1
                i += 1
                while line_strip[i] != ")" and line_strip[i] != "(":
                    i += 1
        else:
            output_actions.append("REDUCE")
            if i == max_idx:
                break
            i += 1
            while line_strip[i] != ")" and line_strip[i] != "(":
                i += 1
    assert i == max_idx
    return output_actions


def find_nts_in_tree(tree):
    tree = tree.strip()
    return re.findall(r"(?=\(([^\s]+)\s\()", tree)


def get_sent_info(arg):
    tree, setting = arg
    tree = tree.strip()
    lowercase, replace_num, vocab, sp = setting
    if sp is not None:
        # use sentencepiece
        tree = transform_to_subword_tree(tree, sp)
    subword_tokenized = sp is not None
    tags, tokens, tokens_lower = get_tags_tokens_lowercase(tree)
    tags, tokens, tokens_lower = get_tags_tokens_lowercase(tree)
    orig_tokens = tokens[:]
    if sp is None:
        # these are not applied with sentencepiece
        if lowercase:
            tokens = tokens_lower
        if replace_num:
            tokens = [utils.clean_number(w) for w in tokens]

        token_ids = [vocab.get_id(t) for t in tokens]
        conved_tokens = [vocab.i2w[w_i] for w_i in token_ids]
    else:
        token_ids = sp.piece_to_id(tokens)
        conved_tokens = tokens

    return {
        "orig_tokens": orig_tokens,
        "tokens": conved_tokens,
        "token_ids": token_ids,
        "tags": tags,
        "tree_str": tree,
    }


def make_vocab(
    textfile,
    seqlength,
    minseqlength,
    lowercase,
    replace_num,
    vocabsize,
    vocabminfreq,
    unkmethod,
    jobs,
    apply_length_filter=True,
):
    w2c = defaultdict(int)
    with open(textfile, "r") as f:
        trees = [tree.strip() for tree in f]
    with Pool(jobs) as pool:
        for tags, sent, sent_lower in pool.map(get_tags_tokens_lowercase, trees):
            assert len(tags) == len(sent)
            if lowercase:
                sent = sent_lower
            if replace_num:
                sent = [utils.clean_number(w) for w in sent]
            if (len(sent) > seqlength and apply_length_filter) or len(
                sent
            ) < minseqlength:
                continue

            for word in sent:
                w2c[word] += 1
    if unkmethod == "berkeleyrule" or unkmethod == "berkeleyrule2":
        conv_method = (
            utils.berkeley_unk_conv
            if unkmethod == "berkeleyrule"
            else utils.berkeley_unk_conv2
        )
        berkeley_unks = set([conv_method(w) for w, c in w2c.items()])
        specials = list(berkeley_unks)
    else:
        specials = [unk]
    if vocabminfreq:
        w2c = dict([(w, c) for w, c in w2c.items() if c >= vocabminfreq])
    elif vocabsize > 0 and len(w2c) > vocabsize:
        sorted_wc = sorted(list(w2c.items()), key=lambda x: x[1], reverse=True)
        w2c = dict(sorted_wc[:vocabsize])
    return Vocabulary(list(w2c.items()), pad, unkmethod, unk, specials)


def get_data(args):
    def get_nonterminals(textfiles, jobs=-1):
        nts = set()
        for fn in textfiles:
            with open(fn, "r") as f:
                lines = [line for line in f]
            with Pool(jobs) as pool:
                local_nts = pool.map(find_nts_in_tree, lines)
                nts.update(list(itertools.chain.from_iterable(local_nts)))
        nts = sorted(list(nts))
        print("Found nonterminals: {}".format(nts))
        return nts

    def convert(
        textfile,
        lowercase,
        replace_num,
        seqlength,
        minseqlength,
        outfile,
        vocab,
        sp,
        apply_length_filter=True,
        jobs=-1,
    ):
        dropped = 0
        num_sents = 0
        conv_setting = (lowercase, replace_num, vocab, sp)

        def process_block(tree_with_settings, f):
            _dropped = 0
            with Pool(jobs) as pool:
                for sent_info in pool.map(get_sent_info, tree_with_settings):
                    tokens = sent_info["tokens"]
                    if apply_length_filter and (
                        len(tokens) > seqlength or len(tokens) < minseqlength
                    ):
                        _dropped += 1
                        continue
                    sent_info["key"] = "sentence"
                    f.write(json.dumps(sent_info) + "\n")
            return _dropped

        with open(outfile, "w") as f, open(textfile, "r") as in_f:
            block_size = 100000
            tree_with_settings = []
            for tree in in_f:
                tree_with_settings.append((tree, conv_setting))
                if len(tree_with_settings) >= block_size:
                    dropped += process_block(tree_with_settings, f)
                    num_sents += len(tree_with_settings)
                    tree_with_settings = []
                    print(num_sents)
            if len(tree_with_settings) > 0:
                process_block(tree_with_settings, f)
                num_sents += len(tree_with_settings)

            others = {
                "vocab": vocab.to_json_dict() if vocab is not None else None,
                "nonterminals": nonterminals,
                "pad_token": pad,
                "unk_token": unk,
                "args": args.__dict__,
            }
            for k, v in others.items():
                print("Saving {} to {}".format(k, outfile + "." + k))
                f.write(json.dumps({"key": k, "value": v}) + "\n")

        print(
            "Saved {} sentences (dropped {} due to length/unk filter)".format(
                num_sents, dropped
            )
        )

    print("First pass through data to get nonterminals...")
    nonterminals = get_nonterminals(
        [args.trainfile, args.valfile, args.testfile], args.jobs
    )

    if args.unkmethod == "subword":
        if args.vocabfile != "":
            print(
                "Loading pre-trained sentencepiece model from {}".format(args.vocabfile)
            )
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor(model_file=args.vocabfile)
            sp_model_path = "{}-spm.model".format(args.outputpath)
            print("Copy sentencepiece model to {}".format(sp_model_path))
            shutil.copyfile(args.vocabfile, sp_model_path)
        else:
            print(
                "unkmethod subword is selected. Running sentencepiece on the training data..."
            )
            sp = learn_sentencepiece(
                args.trainfile, args.outputpath + "/" + "-spm", args
            )
        vocab = None
    else:
        if args.vocabfile != "":
            print("Loading pre-specified source vocab from " + args.vocabfile)
            vocab = Vocabulary.load(args.vocabfile)
        else:
            print("Second pass through data to get vocab...")
            vocab = make_vocab(
                args.trainfile,
                args.seqlength,
                args.minseqlength,
                args.lowercase,
                args.replace_num,
                args.vocabsize,
                args.vocabminfreq,
                args.unkmethod,
                args.jobs,
            )
        vocab.dump(args.outputpath + "/" + ".vocab")
        print("Vocab size: {}".format(len(vocab.i2w)))
        sp = None

    convert(
        args.testfile,
        args.lowercase,
        args.replace_num,
        0,
        args.minseqlength,
        args.outputpath + "/" + "test.json",
        vocab,
        sp,
        0,
        args.jobs,
    )
    convert(
        args.valfile,
        args.lowercase,
        args.replace_num,
        args.seqlength,
        args.minseqlength,
        args.outputpath + "/" + "valid.json",
        vocab,
        sp,
        0,
        args.jobs,
    )
    convert(
        args.trainfile,
        args.lowercase,
        args.replace_num,
        args.seqlength,
        args.minseqlength,
        args.outputpath + "/" + "train.json",
        vocab,
        sp,
        1,
        args.jobs,
    )


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--vocabsize", type=int, default=100000)
    parser.add_argument("--vocabminfreq", type=int, default=1)
    parser.add_argument(
        "--unkmethod",
        choices=["unk", "berkeleyrule", "berkeleyrule2", "subword"],
        default="berkeleyrule",
    )
    parser.add_argument("--subword_type", choices=["bpe", "unigram"], default="bpe")
    parser.add_argument("--keep_ptb_bracket", action="store_true")
    parser.add_argument("--subword_user_defined_symbols", nargs="*")
    parser.add_argument("--lowercase", help="Lower case", action="store_true")
    parser.add_argument(
        "--replace_num", help="Replace numbers with N", action="store_true"
    )
    # parser.add_argument('--trainfile', help="Path to training data.",default='/data/cl/user/eisape/docker-home/incremental_parse_probe/data_large/train.txt')
    # parser.add_argument('--valfile', help="Path to validation data.",default='/data/cl/user/eisape/docker-home/incremental_parse_probe/data_large/valid.txt')
    # parser.add_argument('--testfile', help="Path to test validation data.",default='/data/cl/user/eisape/docker-home/incremental_parse_probe/data_large/test.txt')
    parser.add_argument(
        "--seqlength",
        help="Maximum sequence length. Sequences longer than this are dropped.",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--minseqlength",
        help="Minimum sequence length. Sequences shorter than this are dropped.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--data_dir", help="Prefix of the output file names. ", type=str, default="data"
    )
    parser.add_argument("--vocabfile", type=str, default="")
    parser.add_argument("--jobs", type=int, default=10)
    # for example here is the command line to run the script
    # python3 preprocess.py --trainfile data/train.txt --valfile data/valid.txt --testfile data/test.txt --outputfile ./data/ --jobs 10 --vocabminfreq 1 --lowercase
    # comand to copy ./*.json to /data/cl/user/eisape/drive/ptb1/
    # cp ./*.json /data/cl/user/eisape/drive/ptb1/
    args = parser.parse_args(arguments)
    if args.jobs == -1:
        args.jobs = len(os.sched_getaffinity(0))
    # set file pats by hand
    args.trainfile = args.data_dir + "/train.txt"
    args.valfile = args.data_dir + "/valid.txt"
    args.testfile = args.data_dir + "/test.txt"
    args.outputpath = args.data_dir

    # np.random.seed(3435)
    get_data(args)

    def transsys_lookup(k):
        lookup = {
            "ASw": ArcSwift,
            "AER": ArcEagerReduce,
            "AES": ArcEagerShift,
            "ASd": ArcStandard,
            "AH": ArcHybrid,
        }
        return lookup[k]

    def is_projective(lines):
        projective = True

        # find decendents
        words = ["ROOT"]
        for line in lines:
            words += [line[1]]

        children = [[] for i in range(len(words))]
        for i, line in enumerate(lines):
            try:
                parent = int(line[6])
                relation = line[7]
                children[parent] += [(relation, i + 1)]
            except Exception:
                print(line)

        decendents = [
            set([child[1] for child in children[i]]) for i in range(len(words))
        ]

        change = True
        while change:
            change = False
            for i in range(len(decendents)):
                update = []
                for d in decendents[i]:
                    for d1 in decendents[d]:
                        if d1 not in decendents[i]:
                            update += [d1]
                if len(update) > 0:
                    decendents[i].update(update)
                    change = True

        for i, node in enumerate(children):
            for child in node:
                childid = child[1]
                for j in range(min(childid, i) + 1, max(childid, i)):
                    if j not in decendents[i]:
                        projective = False

        return projective

    def processlines(lines, transsys):
        arcs = [dict() for i in range(len(lines) + 1)]

        pos = ["" for i in range(len(lines) + 1)]
        fpos = ["" for i in range(len(lines) + 1)]

        for i, line in enumerate(lines):
            pos[i + 1] = line[3]  # fine-grained
            fpos[i + 1] = line[4]
            parent = int(line[6])
            relation = line[7]
            arcs[parent][i + 1] = transsys.mappings["rel"][relation]

        res = [
            ParserState_dec(["<ROOT>"] + lines, transsys=transsys, goldrels=arcs),
            pos,
        ]
        if fpos:
            res += [fpos]
        else:
            res == [None]
        return res

    for dataset in ["valid", "train", "test"]:
        sents = []
        ret_sents = []
        ds = dataset
        # if dataset == 'valid': ds='val'
        with open(args.outputpath + "/" + ds + ".json", "r") as f:
            for line in f:
                o = json.loads(line)
                if o["key"] == "sentence":
                    sents.append(o)

        count, nonproj, lines = 0, 0, []

        with open(args.outputpath + "/" + ds + ".conllx", "r") as fin:
            line = fin.readline()
            while line:
                if line.startswith("#"):
                    line = fin.readline()
                    continue
                line = line.strip().split()
                if len(line) > 0 and "-" in line[0]:
                    line = fin.readline()
                    continue

                if len(line) == 0:
                    if is_projective(lines):
                        sents[count]["projective"] = True
                        for tsys in ["ASd"]:
                            sents[count][tsys] = {}
                            transsys = transsys_lookup(tsys)("./data/mappings-ptb.txt")
                            stck, buf, actions, tuples = [], [], [], []

                            state, pos, fpos = processlines(lines, transsys)
                            transsys = state.transsys

                            while len(state.transitionset()) > 0:
                                t = transsys.goldtransition(state)
                                actions.append(t)
                                stck.append(state.stack)
                                buf.append(state.buf)
                                tup = transsys.goldtransition(state, return_tuple=True)
                                tuples.append(list(tup))
                                transsys.advance(state, t)

                            stck.append(state.stack)
                            buf.append(state.buf)

                            sents[count][tsys]["gold_stacks"] = stck
                            sents[count][tsys]["gold_buffers"] = buf
                            sents[count][tsys]["actions"] = actions
                            sents[count][tsys]["action_tuples"] = tuples
                            ret_sents.append(sents[count])
                    else:
                        # Remove non-projective sentences from the dataset
                        sents[count]["projective"] = False
                        ret_sents.append(sents[count])
                    count += 1
                    lines = []
                else:
                    lines += [line]
                line = fin.readline()
            if len(lines) > 0:
                None
        with open(
            args.outputpath + "/" + ds + ".json", "w", encoding="utf8"
        ) as json_file:
            print(f"Writing {ds} to {args.outputpath+'/'+ds+'.json'}", os.getcwd())
            for s in ret_sents:
                json_file.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
