import torch
from pathlib import Path
from typing import List, Optional
import numpy as np
from utils import *
from gpt2 import GPT2_extended

from tqdm import tqdm
import json

from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer, GPT2LMHeadModel
import task
from collections import namedtuple


class PTB_Split(Dataset):
    def __init__(self, split=None, probe=None, config=None, gpt=None):
        with torch.no_grad():
            self.root_dir = config["data_params"]["root_dir"]
            self.data_path = f"{self.root_dir}/{split}.json"
            self.config = config
            self.oracle = probe.oracle
            self.probe = probe
            self.items, self.embs = [], []
            self.gpt = gpt

            if config["probe_params"]["layer"] == "all":
                start, end = 0, MODEL_DATA[config["pretrained_model"]]["layer_count"]
            else:
                start, end = (
                    config["probe_params"]["layer"],
                    config["probe_params"]["layer"] + 1,
                )

            self.observations = self.load_conll_dataset(
                f"{self.root_dir}/{split}.conllx"
            )

            device = "cuda"

            (
                self.token_ids,
                self.stacks,
                self.bufs,
                self.action_ids,
                self.padded_action_ngrams,
                self.embs,
                self.gold_distances,
                self.gold_depths,
                self.lengths,
                self.gold_tuples,
                self.cont_mask,
                self.xpos,
            ) = [[] for _ in range(12)]

            with open(self.data_path) as f:
                num_lines = len(f.readlines())

            with open(self.data_path) as f:
                batch_embs, batch_maps, batch_toks, count = [], [], [], 0
                for idx, line in tqdm(
                    enumerate(f), desc=f"loading {split} data", total=num_lines
                ):
                    o = json.loads(line)
                    if o["key"] == "sentence":
                        if o["projective"]:
                            sent = " ".join(o["orig_tokens"])
                            line = sent.strip()  # Remove trailing characters
                            line = (
                                self.gpt.tokenizer.bos_token
                                + line
                                + self.gpt.tokenizer.eos_token
                            )
                            tokenized_text = self.gpt.tokenizer.tokenize(line)
                            untok_tok_mapping = self.gpt.match_tokenized_to_untokenized(
                                tokenized_text, line
                            )
                            batch_maps.append(untok_tok_mapping)
                            batch_toks.append(tokenized_text)
                            count += 1

                            if count > 100 or idx == num_lines - 1:
                                lens = [len(x) for x in batch_toks]
                                max_len = max(lens)

                                for i, tok in enumerate(batch_toks):
                                    if len(tok) < max_len:
                                        batch_toks[i].extend(
                                            [self.gpt.tokenizer.eos_token]
                                            * (max_len - len(batch_toks[i]))
                                        )
                                batch_embs = [
                                    torch.tensor(
                                        [
                                            self.gpt.tokenizer.convert_tokens_to_ids(
                                                tokenized_text
                                            )
                                        ]
                                    ).to(device)
                                    for tokenized_text in batch_toks
                                ]
                                with torch.no_grad():
                                    encoded_layers = self.gpt.model(
                                        torch.cat(batch_embs, dim=0),
                                        output_hidden_states=True,
                                    )["hidden_states"][start]

                                for ind2, untok_tok_mapping in enumerate(batch_maps):
                                    model_embeddings = encoded_layers[ind2].unsqueeze(0)
                                    aligned_model_embeddings = torch.cat(
                                        [
                                            torch.mean(
                                                model_embeddings[
                                                    :,
                                                    untok_tok_mapping[i][
                                                        0
                                                    ] : untok_tok_mapping[i][-1]
                                                    + 1,
                                                    :,
                                                ],
                                                dim=1,
                                            )
                                            for i, tok in enumerate(
                                                untok_tok_mapping.keys()
                                            )
                                        ]
                                    ).unsqueeze(0)
                                    aligned_model_embeddings = torch.cat(
                                        (
                                            model_embeddings[:, 0:1, :],
                                            aligned_model_embeddings,
                                            model_embeddings[:, -1:, :].repeat(
                                                1,
                                                self.config["data_params"]["token_pad"]
                                                - aligned_model_embeddings.shape[1]
                                                - 1,
                                                1,
                                            ),
                                        ),
                                        dim=1,
                                    ).unsqueeze(0)
                                    assert (
                                        aligned_model_embeddings.shape[2]
                                        == self.config["data_params"]["token_pad"]
                                    )  # model_embeddings.shape[1]#len(untok_tok_mapping.keys())+2

                                    # model_embeddings = align(encoded_layers[ind2].unsqueeze(0), b)
                                    self.embs.append(
                                        aligned_model_embeddings[:, 0, :, :].to("cpu")
                                    )

                                batch_embs, batch_maps, batch_toks, count = (
                                    [],
                                    [],
                                    [],
                                    0,
                                )

                            if self.oracle:
                                action_ids = [
                                    i[0] for i in o[self.oracle.name]["actions"]
                                ]
                                action_ids = np.pad(
                                    action_ids,
                                    (
                                        0,
                                        self.config["data_params"]["action_pad"]
                                        - len(action_ids),
                                    ),
                                    "constant",
                                    constant_values=self.probe.oracle.a2i["PAD"],
                                )
                            else:
                                action_ids = torch.tensor([-1])

                            if (
                                "padded_action_ngrams"
                                in config["probe_params"]["data_sources"]
                            ):
                                padded_action_ngrams = conv_padded_ngrams(
                                    self.probe.oracle.a2i,
                                    action_ids,
                                    action_ngram_pad=self.config["data_params"][
                                        "action_ngram_pad"
                                    ],
                                    token_pad=self.config["data_params"]["token_pad"],
                                )
                            else:
                                padded_action_ngrams = torch.tensor([-1])

                            if (
                                "continuous_action_masks"
                                in config["probe_params"]["data_sources"]
                            ):
                                mask = generate_continuous_mask(
                                    action_ids, self.config["data_params"]["token_pad"]
                                )
                                cont_mask = np.pad(
                                    mask,
                                    (
                                        (
                                            0,
                                            self.config["data_params"]["action_pad"]
                                            - len(mask),
                                        ),
                                        (0, 0),
                                    ),
                                    "constant",
                                    constant_values=-1,
                                )
                            else:
                                cont_mask = torch.tensor([-1])

                            if "gold_stacks" in config["probe_params"]["data_sources"]:
                                stacks = o[self.oracle.name]["gold_stacks"]
                                stacks.extend(
                                    [[0]]
                                    * (
                                        self.config["data_params"]["action_pad"]
                                        - len(stacks)
                                    )
                                )
                                stacks = np.array(
                                    [
                                        i
                                        + [0]
                                        * (
                                            self.config["data_params"]["token_pad"]
                                            - len(i)
                                        )
                                        for i in stacks
                                    ]
                                )
                            else:
                                stacks = torch.tensor([-1])

                            if "gold_buffers" in config["probe_params"]["data_sources"]:
                                bufs = o[self.oracle.name]["gold_buffers"]
                                bufs.extend(
                                    [[0]]
                                    * (
                                        self.config["data_params"]["action_pad"]
                                        - len(bufs)
                                    )
                                )
                                bufs = np.array(
                                    [
                                        i
                                        + [0]
                                        * (
                                            self.config["data_params"]["token_pad"]
                                            - len(i)
                                        )
                                        for i in bufs
                                    ]
                                )
                            else:
                                bufs = torch.tensor([-1])

                            if "gold_tuples" in config["probe_params"]["data_sources"]:
                                gold_tuples = o[self.oracle.name]["action_tuples"]
                                gold_tuples.extend(
                                    [[-1]]
                                    * (
                                        self.config["data_params"]["action_pad"]
                                        - len(gold_tuples)
                                    )
                                )
                                gold_tuples = np.array(
                                    [
                                        i
                                        + [-1]
                                        * (
                                            self.config["data_params"]["token_pad"]
                                            - len(i)
                                        )
                                        for i in gold_tuples
                                    ]
                                )
                            else:
                                gold_tuples = torch.tensor([-1])

                            if (
                                "gold_distances"
                                in config["probe_params"]["data_sources"]
                            ):
                                gold_distances = task.ParseDistanceTask.labels(
                                    self.observations[idx]
                                )
                                gold_distances = np.pad(
                                    gold_distances,
                                    (
                                        (
                                            0,
                                            config["data_params"]["token_pad"]
                                            - len(gold_distances),
                                        ),
                                        (
                                            0,
                                            config["data_params"]["token_pad"]
                                            - len(gold_distances),
                                        ),
                                    ),
                                    "constant",
                                    constant_values=-1,
                                )
                            else:
                                gold_distances = torch.tensor([-1])

                            if "gold_depths" in config["probe_params"]["data_sources"]:
                                gold_depths = task.ParseDepthTask.labels(
                                    self.observations[idx]
                                )
                                gold_depths = np.pad(
                                    gold_depths,
                                    (
                                        0,
                                        config["data_params"]["token_pad"]
                                        - len(gold_depths),
                                    ),
                                    "constant",
                                    constant_values=-1,
                                )
                            else:
                                gold_depths = torch.tensor([-1])

                            if "token_ids" in config["probe_params"]["data_sources"]:
                                token_ids = np.pad(
                                    o["token_ids"],
                                    (
                                        0,
                                        self.config["data_params"]["token_pad"]
                                        - len(o["token_ids"]),
                                    ),
                                    "constant",
                                    constant_values=0,
                                )
                            else:
                                token_ids = torch.tensor([-1])

                            if "xpos" in config["probe_params"]["data_sources"]:
                                xpos = np.pad(
                                    [XPOS2IDX[t] for t in o["tags"]],
                                    (
                                        0,
                                        self.config["data_params"]["token_pad"]
                                        - len(o["tags"]),
                                    ),
                                    "constant",
                                    constant_values=XPOS2IDX["."],
                                )
                            else:
                                xpos = torch.tensor([-1])

                            self.token_ids.append(token_ids)
                            self.stacks.append(stacks)
                            self.bufs.append(bufs)
                            self.action_ids.append(action_ids)
                            self.padded_action_ngrams.append(padded_action_ngrams)
                            self.gold_distances.append(gold_distances)
                            self.gold_depths.append(gold_depths)
                            self.lengths.append(len(o["orig_tokens"]))
                            self.gold_tuples.append(gold_tuples)
                            self.cont_mask.append(cont_mask)
                            self.xpos.append(xpos)

                            if config["data_params"][split]["dry_run"]:
                                if (
                                    len(self.embs)
                                    >= config["data_params"][split]["dry_run"]
                                ):
                                    break
            self.gpt = None

    def generate_lines_for_sent(self, lines):
        """Yields batches of lines describing a sentence in conllx.
        Args:
            lines: Each line of a conllx file.
        Yields:
            a list of lines describing a single sentence in conllx.
        """
        buf = []
        for line in lines:
            if line.startswith("#"):
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

    def load_conll_dataset(self, filepath):
        """Reads in a conllx file; generates Observation objects

        For each sentence in a conllx file, generates a single Observation
        object.
        Args:
        filepath: the filesystem path to the conll dataset

        Returns:
        A list of Observations
        """
        observation_class = namedtuple(
            "Observation",
            [
                "index",
                "sentence",
                "lemma_sentence",
                "upos_sentence",
                "xpos_sentence",
                "morph",
                "head_indices",
                "governance_relations",
                "secondary_relations",
                "extra_info",
                "embeddings",
            ],
        )

        observations = []
        lines = (x for x in open(filepath))
        for buf in self.generate_lines_for_sent(lines):
            conllx_lines = []
            for line in buf:
                conllx_lines.append(line.strip().split("\t"))
            embeddings = [None for x in range(len(conllx_lines))]
            observation = observation_class(*zip(*conllx_lines), embeddings)
            observations.append(observation)
        return observations

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, idx):
        return [
            self.token_ids[idx],
            self.stacks[idx],
            self.bufs[idx],
            self.action_ids[idx],
            self.padded_action_ngrams[idx],
            self.embs[idx],
            self.gold_distances[idx],
            self.gold_depths[idx],
            self.lengths[idx],
            self.gold_tuples[idx],
            self.cont_mask[idx],
            self.xpos[idx],
        ]


class PTB_Dataset(LightningDataModule):
    def __init__(self, config=None, probe=None):
        super().__init__()
        self.config = config
        device = "cuda"
        self.probe = probe
        tokenizer = AutoTokenizer.from_pretrained(
            config["pretrained_model"], local_files_only=True
        )
        model = (
            GPT2LMHeadModel.from_pretrained(
                config["pretrained_model"], local_files_only=True
            )
            .to(device)
            .eval()
        )
        self.gpt = GPT2_extended(model=model, tokenizer=tokenizer, tail=None)
        for param in self.gpt.parameters():
            param.requires_grad = False

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset, self.valid_dataset, self.test_dataset = [
            PTB_Split(split=split, probe=self.probe, config=self.config, gpt=self.gpt)
            for split in ["train", "valid", "test"]
        ]

    def produce_dataloader(self, split):
        return DataLoader(
            self.__dict__[f"{split}_dataset"],
            batch_size=self.config["data_params"][split]["batch_size"],
            num_workers=self.config["data_params"]["num_workers"],
            shuffle=self.config["data_params"][split]["shuffle"],
            pin_memory=self.config["data_params"]["pin_memory"],
        )

    def train_dataloader(self) -> DataLoader:
        return self.produce_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.produce_dataloader("valid")

    def test_dataloader(self) -> DataLoader:
        return self.produce_dataloader("test")
