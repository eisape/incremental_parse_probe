import os
import yaml
import argparse
from collections import defaultdict
from itertools import count
from tqdm import tqdm
import torch
import pandas as pd

from experiment import IncrementalParseProbeExperiment
from task import ParseDepthTask
from datasets import PTB_Dataset
from utils import ignored_tags

from transformers import AutoTokenizer, GPT2LMHeadModel
from transition import *
from utils import *
from gpt2 import GPT2_extended
import json

args = argparse.ArgumentParser()
args.add_argument(
    "--experiment_path",
    type=str,
    default="experiment_checkpoints/eval/gpt2/AttentiveProbe/layer_6/",
)

args = args.parse_args()

with open(args.experiment_path + "config.yaml") as file:
    l_args = yaml.safe_load(file)

print("loading probe...")
l_args["probe_params"]["pretrained_model"] = l_args["pretrained_model"]
l_args["probe_params"]["checkpoint_path"] = None
exp = IncrementalParseProbeExperiment.load_from_checkpoint(
    args.experiment_path + "checkpoints/last.ckpt"
)
p = exp.probe.eval()

print("loading gpt2...")
device = "cuda"
gpt2 = GPT2LMHeadModel.from_pretrained(
    l_args["pretrained_model"], local_files_only=True
)
gpt2_tokenizer = AutoTokenizer.from_pretrained(
    l_args["pretrained_model"], local_files_only=True
)

for param in gpt2.parameters():
    param.requires_grad = False

gpt2_ext = GPT2_extended(model=gpt2, tokenizer=gpt2_tokenizer, tail=None)

results = pd.DataFrame(
    columns=[
        "model",
        "probe_name",
        "layer",
        "loss",
        "distance_mse",
        "depth_mse",
        "oracle_action_nll",
        "f1",
        "perplexity",
        "accuracy",
        "uuas_beamsearch",
        "root_accuracy_beamsearch",
        "root_accuracy_spanning_tree",
        "uuas_spanning_tree",
    ]
)

l_args["data_params"]["test"]["shuffle"] = False
l_args["data_params"]["train"]["dry_run"] = 2
l_args["data_params"]["valid"]["dry_run"] = 2
l_args["data_params"]["test"]["dry_run"] = False

l_args["probe_params"]["data_sources"].extend(
    ["gold_distances", "gold_depths", "xpos", "gold_tuples"]
)

distance_depth_data = PTB_Dataset(config=l_args, probe=p)
distance_depth_data.setup()

with open(distance_depth_data.test_dataset.data_path) as f:
    (
        total_sents,
        correct_root_predictions,
        uspan_correct,
        uspan_total,
        uas_correct,
        uas_total,
        uuas_w_head_total,
        uuas_w_head_correct,
    ) = (0, 0, 0, 0, 0, 0, 0, 0)
    incr = count()
    for idx, line in tqdm(enumerate(f), desc=f"beamsearch decoding"):
        o = json.loads(line)
        if o["key"] == "sentence" and o["projective"]:
            inc = next(incr)
            if len(o["tokens"]) > 1:
                topk, ncont, parses = 10, 10, []

                while not parses:
                    if topk > 100:
                        print("max beamsize exceeded, breaking")
                        break
                    print("topk:", topk, " /ncont:", ncont)
                    parses = gpt2_ext.parse_beamsearch(
                        probe=p,
                        sentence=" ".join(o["orig_tokens"]),
                        generative=False,
                        topk=topk,
                        ncont=ncont,
                    )
                    topk, ncont = topk * 2, ncont * 2

                if not parses:
                    print("no parses found")
                    continue

                batch = exp.format_batch(
                    [
                        torch.tensor(i)
                        for i in distance_depth_data.test_dataset.__getitem__(inc)
                    ]
                )

                top_parse = parses[0][1]
                test_batch = exp.format_batch(
                    [
                        torch.tensor(i)
                        for i in distance_depth_data.test_dataset.__getitem__(inc)
                    ]
                )

                vparse = parses[0][1]

                gold_depths = batch["gold_depths"][: batch["lengths"]]
                gold_distances = batch["gold_distances"][
                    : batch["lengths"], : batch["lengths"]
                ]

                pred_depths = ParseDepthTask.labels(obs(top_parse.heads_idxs()))
                correct_root_predictions += (
                    (gold_depths == 0).nonzero(as_tuple=True)[0]
                ).item() == get_nopunct_argmin(vparse.heads_idxs(), batch["xpos"])

                gold_edges = prims_matrix_to_edges(gold_distances, test_batch["xpos"])
                pred_edges = [
                    tuple(sorted((tup[0] - 1, tup[1][0] - 1)))
                    for tup, tag in zip(vparse.head.items(), o["tags"])
                    if not tag in ignored_tags
                ]

                total_sents += 1
                top_parse_head_invetred = {}

                gold_heads = [
                    i
                    for i, tag in zip(
                        distance_depth_data.test_dataset.observations[idx].head_indices,
                        o["tags"],
                    )
                    if not tag in ignored_tags
                ]
                pred_heads = [
                    i
                    for i, tag in zip(vparse.heads_idxs(), o["tags"])
                    if not tag in ignored_tags
                ]

                invert_heads = defaultdict(list)
                for x, y in vparse.head.items():
                    invert_heads[int(y[0])].append(int(x))

                overlap = [
                    h for i, h in enumerate(pred_heads) if gold_heads[i] == str(h)
                ]
                undir_overlap = [
                    h
                    for i, h in enumerate(pred_heads)
                    if gold_heads[i] == str(h) or i in invert_heads[h]
                ]
                undir_overlap_no_root = [h for h in undir_overlap if h != 0]

                uuas_w_head_correct += len(undir_overlap)

                uuas_w_head_total += len(gold_heads)
                uspan_correct += len(undir_overlap_no_root)
                uspan_total += len(gold_heads) - 1
                uas_correct += len(overlap)
                uas_total += len(gold_heads)

                root_acc = correct_root_predictions / float(total_sents)
                uuas = uspan_correct / float(uspan_total)
                uas = uas_correct / float(uas_total)
                uuas_w_head = uuas_w_head_correct / float(uuas_w_head_total)

                print(
                    "root_acc:",
                    root_acc,
                    "uas:",
                    uas,
                    "uuas:",
                    uuas,
                    "uuas_w_head:",
                    uuas_w_head,
                    "inc:",
                    inc,
                )

    results = results.append(
        {
            "model": l_args["pretrained_model"],
            "probe_name": l_args["probe_params"]["probe_name"],
            "layer": l_args["probe_params"]["layer"],
            "uuas_beamsearch": uuas,
            "uas_beamsearch": uas,
            "uuas_beamsearch_w_head": uuas_w_head,
            "root_accuracy_beamsearch": root_acc,
        },
        ignore_index=True,
    )

    results = results.melt(
        id_vars=["model", "probe_name", "layer"], var_name="metric", value_name="value"
    ).dropna()

results_path = f'./results/results_{l_args["pretrained_model"]}_layer_{str(l_args["probe_params"]["layer"])}_{l_args["probe_params"]["probe_name"]}_beamsearch.csv'
if os.path.exists(results_path):
    net_res = pd.read_csv(results_path)
    pd.concat([results, net_res]).drop_duplicates(
        subset=["model", "probe_name", "layer", "metric"]
    ).to_csv(results_path, index=False)
else:
    os.makedirs(results_path.rsplit("/", 1)[0], exist_ok=True)
    results.to_csv(results_path, mode="a", header=True, index=False)
