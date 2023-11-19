import torch.nn as nn
import torch
from collections import defaultdict
from torch import optim
from queue import PriorityQueue
from utils import *
from itertools import count

# torch won't bp through time in eval mode unless we set:
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
class ClozeTail_gpt2(nn.Module):
    def __init__(self, cloze_model, layer_idx):
        super(ClozeTail_gpt2, self).__init__()
        self.last_layer = cloze_model.lm_head

    def forward(self, x):
        transformer_output = self.transformer(x)[0]
        return transformer_output


class GPT2_extended(nn.Module):
    def __init__(self, model=None, tokenizer=None, tail=None):
        super(GPT2_extended, self).__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.tail = tail

        for param in self.model.parameters():
            param.requires_grad = False

    def tail_by_layer(self, layer, x):
        if layer < self.model.config.n_layer:
            tl = ClozeTail_gpt2(self.model, layer)
            tl.eval()
            return tl(x)
        else:
            return self.model.lm_head(x)

    def embeddings_w_map(self, sentence, layer):
        untokenized_sent = sentence.split()
        tokenized_sent = self.tokenizer.tokenize(
            self.tokenizer.bos_token + sentence + self.tokenizer.eos_token
        )
        tokens_tensor = self.tokenizer.encode(
            self.tokenizer.bos_token + sentence + self.tokenizer.eos_token,
            return_tensors="pt",
        ).to(self.model.device)
        output = self.model(tokens_tensor, output_hidden_states=True)
        model_embeddings = output["hidden_states"][layer].detach()

        original_embeddings = model_embeddings.detach().clone().to(self.model.device)

        untok_tok_mapping = self.match_tokenized_to_untokenized(
            tokenized_sent, untokenized_sent
        )

        return original_embeddings, untok_tok_mapping

    def align(self, model_embeddings, untok_tok_mapping):
        aligned_model_embeddings = torch.cat(
            [
                torch.mean(
                    model_embeddings[
                        :, untok_tok_mapping[i][0] : untok_tok_mapping[i][-1] + 1, :
                    ],
                    dim=1,
                )
                for i, tok in enumerate(untok_tok_mapping.keys())
            ]
        ).unsqueeze(0)

        aligned_model_embeddings = torch.cat(
            (
                model_embeddings[:, 0:1, :],
                aligned_model_embeddings,
                model_embeddings[:, -1:, :],
            ),
            dim=1,
        ).unsqueeze(0)

        assert aligned_model_embeddings.shape[2] == len(untok_tok_mapping.keys()) + 2

        return aligned_model_embeddings

    def match_tokenized_to_untokenized(self, tokenized_sent, untokenized_sent):
        """Aligns tokenized and untokenized sentence given subwords "##" prefixed
        Assuming that each subword token that does not start a new word is prefixed
        by two hashes, "##", computes an alignment between the un-subword-tokenized
        and subword-tokenized sentences.
        Args:
            tokenized_sent: a list of strings describing a subword-tokenized sentence
            untokenized_sent: a list of strings describing a sentence, no subword tok.
        Returns:
            A dictionary of type {int: list(int)} mapping each untokenized sentence
            index to a list of subword-tokenized sentence indices
        """
        # avoiding |eos|
        tokenized_sent = tokenized_sent[:-1]
        mapping = defaultdict(list)
        untokenized_sent_index = 0
        # avoiding |bos|
        tokenized_sent_index = 1
        while untokenized_sent_index < len(
            untokenized_sent
        ) and tokenized_sent_index < len(tokenized_sent):
            while tokenized_sent_index + 1 < len(tokenized_sent) and not tokenized_sent[
                tokenized_sent_index + 1
            ].startswith("Ġ"):
                mapping[untokenized_sent_index].append(tokenized_sent_index)
                tokenized_sent_index += 1
            mapping[untokenized_sent_index].append(tokenized_sent_index)
            untokenized_sent_index += 1
            tokenized_sent_index += 1
        return mapping

    def gen_counterfactuals(
        self,
        probe=None,
        sent=None,
        label_batch=None,
        num_steps=500000,
        patience=10000,
        verbose=True,
        loss_tolerance=0.05,
        lr=0.0001,
        print_every=5000,
        prefix_freebits=1,
        lastword_freebits=1,
        kl_weight=1,
        scheduler_patience=100,
        compute_kl=True,
    ):
        probe.eval()
        untokenized_sent = sent.split()
        tokenized_sent = self.tokenizer.tokenize(
            self.tokenizer.bos_token + sent + self.tokenizer.eos_token
        )
        tokens_tensor = self.tokenizer.encode(
            self.tokenizer.bos_token + sent + self.tokenizer.eos_token,
            return_tensors="pt",
        ).to(self.model.device)
        model_embeddings = self.model(tokens_tensor, output_hidden_states=True)[
            "hidden_states"
        ][probe.layer].detach()
        original_embeddings = (
            model_embeddings.detach().clone().unsqueeze(0).to(self.model.device)
        )
        model_embeddings = model_embeddings.unsqueeze(0).repeat(
            label_batch["gold_tuples"].shape[0], 1, 1, 1
        )
        untok_tok_mapping = self.match_tokenized_to_untokenized(
            tokenized_sent, untokenized_sent
        )

        model_embeddings.requires_grad = True
        optimizer = torch.optim.Adam([model_embeddings], lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=scheduler_patience
        )

        prediction_loss = 100  # Initialize the prediction loss as high
        increment_idx = 0

        smallest_loss = prediction_loss
        steps_since_best = 0
        # print(prediction_loss > loss_tolerance)
        while prediction_loss > loss_tolerance:
            if increment_idx >= num_steps:
                if verbose:
                    print("Breaking because of increment index")
                break

            if increment_idx % print_every == 0 and verbose:
                print(f"=========== step {increment_idx} ===========")

            if model_embeddings.shape[1] == len(untokenized_sent) + 2:
                aligned_model_embeddings = model_embedding  # s.unsqueeze(0)

            else:
                assert model_embeddings.shape[2] == len(tokenized_sent)

                aligned_model_embeddings = torch.cat(
                    [
                        torch.mean(
                            model_embeddings[
                                :,
                                :,
                                untok_tok_mapping[i][0] : untok_tok_mapping[i][-1] + 1,
                                :,
                            ],
                            dim=2,
                        )
                        for i, tok in enumerate(untokenized_sent)
                    ],
                    dim=1,
                ).unsqueeze(1)

                aligned_model_embeddings = torch.cat(
                    (
                        model_embeddings[:, :, 0:1, :],
                        aligned_model_embeddings,
                        model_embeddings[:, :, -1:, :],
                    ),
                    dim=2,
                )  # .unsqueeze(0)

                assert aligned_model_embeddings.shape[2] == len(untokenized_sent) + 2

            batch = {
                "padded_embeddings": aligned_model_embeddings,
                "gold_tuples": label_batch["gold_tuples"].clone(),
                "action_ids": label_batch["action_ids"].clone(),
                "continuous_action_masks": label_batch[
                    "continuous_action_masks"
                ].clone(),
            }

            loss_dict = probe.batch_step_train(batch)
            loss = loss_dict["loss"]
            prediction_loss = loss.clone().detach()
            if increment_idx == 0:
                initial_loss = loss.clone().detach()

            """kldivloss"""
            if compute_kl and kl_weight > 0:
                print("computing kl")
                postperturb_logits = self.tail(aligned_model_embeddings[0])

                prefix_kl_loss = (
                    F.kl_div(
                        preperturb_logits[:, :-2, :].log_softmax(-1),
                        postperturb_logits[:, :-2, :].log_softmax(-1),
                        size_average=None,
                        reduce=False,
                        log_target=True,
                    )
                    .sum(-1)
                    .squeeze()
                )

                last_word_kl_loss = (
                    F.kl_div(
                        preperturb_logits[:, -2:-1, :].log_softmax(-1),
                        postperturb_logits[:, -2:-1, :].log_softmax(-1),
                        size_average=None,
                        reduce=False,
                        log_target=True,
                    )
                    .sum(-1)
                    .squeeze()
                )

                # output_kl_loss_mean = output_kl_loss.sum()/mask_mask.sum()

                loss += kl_weight * (
                    torch.abs(last_word_kl_loss.mean() - lastword_freebits)
                    + torch.abs(prefix_kl_loss.mean() - prefix_freebits)
                )
                if increment_idx % print_every == 0 and verbose:
                    print(
                        f"abs(last_word_kl - fb): {torch.abs(last_word_kl_loss.mean() - lastword_freebits).detach()}"
                    )
                    print(
                        f"abs(prefix_kl - fb): {torch.abs(prefix_kl_loss.mean() - prefix_freebits).detach()}"
                    )
            """"""

            loss.backward()
            # adwf
            optimizer.step()
            scheduler.step(loss)

            if increment_idx % print_every == 0 and verbose:
                print(f"steps_since_best: {steps_since_best}")
                print(f"total_loss: {loss.detach()}")
                print("==============================")
                print()

            if (smallest_loss - prediction_loss) > 0.001:
                best_embeddings = model_embeddings.detach().clone()
                steps_since_best = 0
                smallest_loss = prediction_loss

            else:
                steps_since_best += 1
                # if steps_since_best == patience/2:
                if steps_since_best == patience and verbose:
                    print("Breaking because of patience with loss", smallest_loss)
                    break
            increment_idx += 1
        if verbose:
            print(f"Exited grad update loop after {increment_idx} steps, ")

        return {
            "padded_embeddings": best_embeddings,
            "original_embeddings": original_embeddings[0],
            "output_logits": None,
            "original_logits": None,
            "cfx_loss": prediction_loss.item(),
            "initial_loss": initial_loss.item(),
        }

    def parse_beamsearch(
        self,
        probe=None,
        sentence=None,
        generative=False,
        topk=30,
        ncont=5,
    ):
        """
        Beam search decoding
        inputs: probe - IncrementalParse Probe
        outputs: [(score, parsestate) x beam_width]
        """
        probe.eval().to(self.model.device)
        init_parserstate = probe.oracle.initial_state()

        original_model_embeddings, untok_tok_mapping = self.embeddings_w_map(
            sentence, probe.layer
        )
        original_model_embeddings = self.align(
            original_model_embeddings, untok_tok_mapping
        )
        init_parserstate.model_embeddings = original_model_embeddings
        sentence_tokens = self.tokenizer.encode(
            self.tokenizer.bos_token + sentence + self.tokenizer.eos_token,
            return_tensors="pt",
        ).to(self.model.device)[0]
        self.model.device
        endstates = []
        states = PriorityQueue()
        state_count = count()
        states.put((0, next(state_count), init_parserstate))

        sentence_len = len(sentence.split())

        while True:
            next_states = []
            while states.qsize():
                if len(next_states) >= topk:
                    break
                score, _, state = states.get()

                ngram_init_state = state
                ngram_beam_width = ncont // 10
                ngram_topk = ncont
                probe.eval()
                # Number of ngrams to generate
                ngram_endstates = []
                ngram_states = PriorityQueue()
                ngram_states.put((0, ngram_init_state))
                # from itertools import count

                while True:
                    ngram_pruned_queue = PriorityQueue()
                    ngram_state_model_embeddings = []
                    ngram_node1s = []
                    ngram_node2s = []
                    ngram_action_ids = []
                    ngram_continuous_action_masks = []
                    # prune to the topl
                    for i in range(ngram_topk):
                        if ngram_states.qsize():
                            ngram_score, ngram_state = ngram_states.get()
                            # check if state has reached a shift or is terminal and check if we have the desired number of states
                            # and state batch data to meta batch
                            # only add to the batch if there is an action to predict
                            if len(ngram_state.stack) > 1:
                                ngram_state_batch = ngram_state.to_batch(probe)
                                ngram_state_model_embeddings.append(
                                    ngram_state_batch["padded_embeddings"]
                                )
                                ngram_node1s.append(ngram_state.stack[0])
                                ngram_node2s.append(ngram_state.stack[1])
                                ngram_action_ids.append(ngram_state_batch["action_ids"])
                                ngram_continuous_action_masks.append(
                                    ngram_state_batch["continuous_action_masks"]
                                )
                            ngram_pruned_queue.put((ngram_score, ngram_state))

                    if ngram_node1s:
                        ngram_batch = {
                            "padded_embeddings": torch.cat(
                                ngram_state_model_embeddings, dim=0
                            ).to(self.model.device),
                            "node1s": torch.tensor(ngram_node1s).to(self.model.device),
                            "node2s": torch.tensor(ngram_node2s).to(self.model.device),
                            "action_ids": torch.cat(ngram_action_ids, dim=0).to(
                                self.model.device
                            ),
                            "continuous_action_masks": torch.cat(
                                ngram_continuous_action_masks, dim=0
                            ).to(self.model.device),
                        }
                        # run once for the whole q
                        ngram_action_dists = probe.action_dists(ngram_batch)
                    else:
                        ngram_action_dists = []
                    ngram_states = ngram_pruned_queue
                    if not ngram_states.qsize():
                        break

                    ngram_c = count()
                    ngram_next_states = []

                    while ngram_states.qsize():
                        ngram_score, ngram_state = ngram_states.get()
                        """get predictions from probe"""
                        ngram_possible_actions = np.array(
                            [i[0] for i in ngram_state.transitionset()]
                        )

                        if len(ngram_state.stack) > 1:
                            # get the action distribution for the current state
                            # if stack <=1 dont need to increment because it's action dist isnt in the batch
                            ngram_inc = next(ngram_c)
                            ngram_node1, ngram_node2 = (
                                ngram_state.stack[0],
                                ngram_state.stack[1],
                            )
                            ngram_actions_dist = ngram_action_dists[ngram_inc][:3]

                        else:
                            ngram_node1, ngram_node2 = -1, -1
                            ngram_actions_dist = (
                                torch.zeros(probe.oracle.num_actions).to(
                                    self.model.device
                                )
                                - 1e10
                            )
                            ngram_actions_dist[probe.oracle.a2i["SHIFT"]] = 0

                        # take the top k scores
                        ngram_log_prob, ngram_indexes = torch.topk(
                            ngram_actions_dist, probe.oracle.num_actions
                        )
                        ngram_possible_action_mask = torch.zeros(
                            probe.oracle.num_actions
                        ).to(self.model.device)

                        for ngram_pa in ngram_possible_actions:
                            ngram_possible_action_mask += ngram_indexes == ngram_pa

                        ngram_log_prob, ngram_indexes = (
                            ngram_log_prob[ngram_possible_action_mask.bool()],
                            ngram_indexes[ngram_possible_action_mask.bool()],
                        )

                        for ngram_new_k, _ in enumerate(ngram_possible_actions):
                            ngram_action = ngram_indexes[ngram_new_k].item()
                            if (
                                0 in ngram_state.heads_idxs()
                                and ngram_node2 == 0
                                and ngram_action == 2
                            ):
                                continue
                            ngram_action_log_prob = ngram_log_prob[ngram_new_k].item()

                            ngram_state_clone = ngram_state.clone()
                            # transition from int doesnt work aparently so we need to give tuple
                            probe.oracle.advance(ngram_state_clone, (ngram_action, -1))
                            probe.oracle._preparetransitionset(ngram_state_clone)

                            ngram_state_clone.action_tuples.append(
                                [ngram_action, ngram_node1, ngram_node2]
                            )
                            ngram_state_clone.log_prob += ngram_action_log_prob

                            ngram_state_clone.action_log_probs.append(
                                ngram_action_log_prob
                            )
                            if (
                                ngram_state_clone.action_tuples[-1][0]
                                == probe.oracle.a2i["SHIFT"]
                                and ngram_state_clone.action_tuples
                                != ngram_init_state.action_tuples
                            ) or len(ngram_state_clone.transitionset()) == 0:
                                ngram_endstates.append((ngram_score, ngram_state_clone))
                                # if we reached maximum # of sentences required
                                if (
                                    len(ngram_endstates) >= ngram_beam_width
                                    or not ngram_states.qsize()
                                ):
                                    break
                                else:
                                    continue

                            ngram_next_states.append(
                                (-ngram_state_clone.log_prob, ngram_state_clone)
                            )

                    for ngram_ss in ngram_next_states:
                        ngram_states.put(ngram_ss)

                scores_conts = sorted(
                    ngram_endstates, key=lambda x: x[0], reverse=False
                )

                for score, cont in scores_conts:
                    if cont.num_shifts == sentence_len:
                        cont.buf = []
                    else:
                        cont.buf = [cont.num_shifts + 1]
                    probe.oracle._preparetransitionset(cont)
                    # next_states.append((-cont.log_prob/len(cont.action_tuples),_, cont))
                    next_states.append((-cont.log_prob, _, cont))

            for ss in next_states:
                states.put((ss[0], next(state_count), ss[2]))

            pruned_queue = PriorityQueue()
            # mask the logits that are not the next token
            next_token_masks = []
            state_action_tuples = []

            # prune to the topk
            for i in range(topk):
                if states.qsize():
                    score, _, state = states.get()
                    # check if state has reached a shift or is terminal and check if we have the desired number of states
                    if len(state.transitionset()) == 0:
                        if (
                            state.num_shifts != sentence_len
                            or (
                                np.array([state.head[i] for i in state.head.keys()])
                                == 0
                            ).sum()
                            > 1
                        ):
                            if states.qsize():
                                continue
                            else:
                                break
                        endstates.append((score, state))

                        if len(endstates) >= topk or not states.qsize():
                            break
                        else:
                            continue

                    if generative:
                        # and state batch data to meta batch
                        state_batch = state.to_batch(probe)
                        mask = (
                            torch.zeros(
                                sentence_tokens.shape[0], self.tokenizer.vocab_size
                            )
                            .to(self.model.device)
                            .unsqueeze(0)
                        )
                        mask[
                            :,
                            untok_tok_mapping[state.num_shifts - 1][
                                0
                            ] : untok_tok_mapping[state.num_shifts - 1][-1]
                            + 1,
                            :,
                        ] = 1
                        next_token_masks.append(mask)

                        state_action_tuples.append(
                            torch.cat(
                                [
                                    state_batch["gold_tuples"],
                                    torch.tensor([-1, -1, -1, -1])
                                    .unsqueeze(0)
                                    .repeat(
                                        400 - state_batch["gold_tuples"].shape[1], 1
                                    )
                                    .unsqueeze(0)
                                    .to(self.model.device),
                                ],
                                dim=1,
                            )
                        )

                    pruned_queue.put((score, _, state))

            if state_action_tuples:
                batch = {"gold_tuples": torch.cat(state_action_tuples, dim=0)}

                counterfactuals = self.gen_counterfactuals(
                    probe=probe,
                    sent=sentence,
                    label_batch=batch,
                    output_probs=False,
                    print_every=100,
                    lr=0.001,
                    patience=100,
                    num_steps=50000,
                    loss_tolerance=0.01,
                    prefix_freebits=0,
                    lastword_freebits=0,
                    kl_weight=0,  # .0001,
                    scheduler_patience=1000,
                    verbose=True,
                    compute_kl=False,
                )

                # run once for the whole q
                counterfactual_logprobs = self.tail_by_layer(
                    probe.layer, counterfactuals["padded_embeddings"][:, 0, :, :]
                ).log_softmax(dim=-1)
                batch_mask = torch.cat(next_token_masks, dim=0)
                next_word_log_probs = (
                    torch.gather(
                        counterfactual_logprobs[:, :-1] * batch_mask[:, :-1],
                        -1,
                        sentence_tokens[1:]
                        .unsqueeze(0)
                        .T.unsqueeze(0)
                        .repeat(batch["gold_tuples"].shape[0], 1, 1),
                    )
                    .sum(-1)
                    .sum(-1)
                )

                new_queue = PriorityQueue()
                inc = count()
                while pruned_queue.qsize():
                    score, _, state = pruned_queue.get()
                    state.log_prob += next_word_log_probs[next(inc)].item()
                    # TODO: if using length norm then use it here
                    new_queue.put((-state.log_prob, _, state))
                states = new_queue
            else:
                states = pruned_queue

            if not states.qsize():
                break

        return endstates
