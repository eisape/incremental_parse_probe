from utils import *
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import transition as transition_system
from sklearn.metrics import accuracy_score, f1_score
import experiment


class IncrementalProbe(nn.Module):
    def __init__(self, args):
        super(IncrementalProbe, self).__init__()
        self.args = args
        if args["oracle_params"]["name"]:
            self.oracle = getattr(transition_system, args["oracle_params"]["name"])(
                args["oracle_params"]["mappings_file"]
            )
        else:
            self.oracle = None

        self.add_root = args["add_root"] if "add_root" in args.keys() else None
        self.embeddings_dropout_rate = (
            args["embeddings_dropout_rate"]
            if "embeddings_dropout_rate" in args.keys()
            else None
        )
        self.layer_dropout_rate = (
            args["layer_dropout_rate"] if "layer_dropout_rate" in args.keys() else None
        )
        self.checkpoint_path = (
            args["checkpoint_path"] if "checkpoint_path" in args.keys() else None
        )
        self.num_layers = args["num_layers"] if "num_layers" in args.keys() else None
        self.layer = args["layer"] if "layer" in args.keys() else None
        self.pretrained_model = (
            args["pretrained_model"] if "pretrained_model" in args.keys() else None
        )

        self.vocab_size = len(self.oracle.a2i)

        self.probe_rank = self.model_dim = MODEL_DATA[self.pretrained_model][
            "feature_count"
        ]

        self.root = nn.Parameter(data=torch.zeros(self.model_dim))
        self.nll = nn.NLLLoss(reduction="none")

    def add_root_distance_labels(self, batch):
        depths_w_root = self.add_root_depth_labels(batch)
        gold_distances = batch["gold_distances"].clone().to(self.device)
        distances_w_root = torch.zeros(
            gold_distances.shape[0],
            gold_distances.shape[1] + 1,
            gold_distances.shape[2] + 1,
            device=self.device,
        )
        distances_w_root[:, 1:, 1:] += gold_distances
        distances_w_root[:, 0, :] += depths_w_root.clone()
        distances_w_root[:, :, 0] += depths_w_root.clone()
        return distances_w_root

    def add_root_depth_labels(self, batch):
        gold_depths = batch["gold_depths"].clone().to(self.device)

        gold_depths += 1
        gold_depths[gold_depths == 0] = -1

        depths_w_root = torch.zeros(
            gold_depths.shape[0], gold_depths.shape[1] + 1, device=self.device
        )
        depths_w_root[:, 1:] += gold_depths.clone().to(self.device)
        return depths_w_root

    def add_root_model_embeddings(self, batch):
        model_embeddings = batch["padded_embeddings"][:, 0, 1:, :].to(self.device)
        embeddings_w_root = torch.zeros(
            model_embeddings.shape[0],
            model_embeddings.shape[1] + 1,
            model_embeddings.shape[2],
            device=self.device,
        )
        embeddings_w_root[:, 1:, :] = model_embeddings.clone()
        embeddings_w_root[:, 0, :] += self.root
        return embeddings_w_root.unsqueeze(1)


class AttentionLayer(nn.Module):
    def __init__(self, y_dim=512, x_dim=512):
        super(AttentionLayer, self).__init__()
        self.key = nn.Linear(y_dim, x_dim, bias=False)
        self.query = nn.Linear(x_dim, x_dim, bias=False)
        self.device = None

    def forward(self, x, y, masks=None, output_attentions=False):
        self.device = next(self.parameters()).device
        q = self.query(x)
        k = self.key(y)
        v = y

        w = torch.matmul(q, k.transpose(-1, -2))

        w = torch.where(
            masks.unsqueeze(2).bool(), w, torch.tensor(-1e10).to(self.device)
        )

        w = nn.Softmax(dim=-1)(w)

        return torch.matmul(w, v)[:, :, 0, :], w[:, :, 0, :]


class AttentiveProbe(IncrementalProbe):
    def __init__(self, args):
        super(AttentiveProbe, self).__init__(args)
        IncrementalProbe.__init__(self, args)
        self.reverse = args["reverse"]
        self.continuous = args["continuous"]
        self.rnn_type = args["rnn_type"]
        self.num_layers = args["num_layers"]
        self.emb_size = args["emb_size"]
        self.state_size = args["state_size"]
        self.vocab_size = len(self.oracle.a2i)

        self.embeddings_dropout = nn.Dropout(self.embeddings_dropout_rate)
        self.layer_dropout = nn.Dropout(self.layer_dropout_rate)
        self.encoder = nn.Embedding(self.vocab_size, self.emb_size)

        self.rnn = getattr(nn, self.rnn_type)(
            self.emb_size, self.state_size, self.num_layers, dropout=0, batch_first=True
        )

        layers = [
            nn.Sequential(
                nn.Linear(self.state_size + self.model_dim, self.state_size), nn.ReLU()
            )
            for layer_idx in range(1)
        ]
        layers = layers + [nn.Linear(self.state_size, len(self.oracle.actions_list()))]
        self.decoder = nn.Sequential(*layers)

        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)

        self.attn = AttentionLayer(y_dim=self.model_dim, x_dim=self.state_size)

    def forward(self, batch):
        model_embeddings = self.add_root_model_embeddings(batch)[:, 0, :, :].to(
            self.device
        )

        inpt = batch["action_ids"].to(self.device)

        models = model_embeddings.unsqueeze(1).repeat(1, inpt.shape[1], 1, 1)
        models = self.embeddings_dropout(models)

        masks = batch["continuous_action_masks"].to(self.device)

        hidden = self.repackage_hidden(self.init_hidden(inpt.shape[0]))
        emb = self.encoder(inpt)
        output, hidden = self.rnn(emb, hidden)
        context, attentions = self.attn(output.unsqueeze(2), models, masks)
        context = self.layer_dropout(context)

        output = torch.cat((output, context), dim=-1)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, len(self.oracle.actions_list()))
        return F.log_softmax(decoded, dim=-1), hidden, attentions

    def batch_step_eval(self, batch):
        self.device = next(self.parameters()).device
        dists, hidden, attentions = self(batch)

        targets = batch["action_ids"].roll(-1, dims=-1).to(self.device)
        targets[:, -1] = self.oracle.a2i["PAD"]

        mask = targets.flatten() - self.oracle.a2i["PAD"]
        mask = mask.nonzero(as_tuple=True)

        loss = self.nll(dists[mask], targets.flatten()[mask])

        predicted_actions = dists[mask].argmax(dim=-1).detach().cpu().numpy()
        losses = {
            "loss": loss.mean(),
            "f1": torch.tensor(
                f1_score(
                    predicted_actions,
                    targets.flatten()[mask].detach().cpu().numpy(),
                    average="macro",
                )
            ),
            "accuracy": torch.tensor(
                accuracy_score(
                    predicted_actions, targets.flatten()[mask].detach().cpu().numpy()
                )
            ),
            "perplexity": torch.exp(loss.mean()),
        }

        return losses

    def action_dists(self, batch):
        masks = batch["continuous_action_masks"]
        self.device = next(self.parameters()).device
        model_embeddings = self.add_root_model_embeddings(batch)[:, 0, :, :]

        inpt = batch["action_ids"]

        models = model_embeddings.unsqueeze(1).repeat(1, inpt.shape[1], 1, 1)
        models = self.embeddings_dropout(models)

        hidden = self.repackage_hidden(self.init_hidden(inpt.shape[0]))
        emb = self.encoder(inpt)
        output, hidden = self.rnn(emb, hidden)
        context, attentions = self.attn(output.unsqueeze(2), models, masks)
        output = torch.cat((output, context), dim=-1)
        decoded = self.decoder(output)

        return F.log_softmax(decoded[:, -1], dim=-1)

    def batch_step_train(self, batch, deterministic_action_loss=False):
        self.device = next(self.parameters()).device
        dists, hidden, attentions = self(batch)
        targets = batch["action_ids"].roll(-1, dims=-1)
        targets[:, -1] = self.oracle.a2i["PAD"]
        mask = targets.flatten() - self.oracle.a2i["PAD"]
        mask = mask.nonzero(as_tuple=True)
        loss = self.nll(dists[mask], targets.flatten()[mask])
        return {"loss": loss.mean()}

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (
                weight.new_zeros(self.num_layers, bsz, self.state_size),
                weight.new_zeros(self.num_layers, bsz, self.state_size),
            )
        else:
            return weight.new_zeros(self.num_layers, bsz, self.state_size)


class StackActionProbe(IncrementalProbe):
    def __init__(self, args):
        super(IncrementalProbe, self).__init__()
        IncrementalProbe.__init__(self, args)
        self.input_size = MODEL_DATA[self.pretrained_model]["feature_count"] * 2
        self.num_layers = args["num_layers"]
        layers = [
            nn.Sequential(
                nn.Linear(self.input_size, self.input_size),
                nn.ReLU(),
                nn.Dropout(self.layer_dropout_rate),
            )
            for layer_idx in range(self.num_layers - 1)
        ]
        layers = (
            [nn.Dropout(self.embeddings_dropout_rate)]
            + layers
            + [nn.Linear(self.input_size, len(self.oracle.actions_list()))]
        )
        self.transform = nn.Sequential(*layers)
        self.device = next(self.parameters()).device

    def forward(self, embeddings):
        return self.transform(embeddings).log_softmax(-1)

    def batch_step_eval(self, batch):
        self.device = next(self.parameters()).device

        if self.add_root:
            model_embeddings = self.add_root_model_embeddings(batch)[:, 0, :, :].to(
                self.device
            )

        oracle_action_idxs, targets = self.oracle.targets_idxs(batch)

        first_emb_indx, second_emb_indx = (
            oracle_action_idxs[[0, 1], :],
            oracle_action_idxs[[0, 2], :],
        )

        emb_pairs = torch.cat(
            (model_embeddings[first_emb_indx], model_embeddings[second_emb_indx]), dim=1
        )

        output_distributions = self.forward(emb_pairs)
        predicted_actions = output_distributions.argmax(dim=-1).detach().cpu().numpy()

        loss = self.nll(
            output_distributions, torch.tensor(targets, device=self.device)
        ).mean()
        losses = {
            "loss": loss,
            "accuracy": torch.tensor(accuracy_score(predicted_actions, targets)),
            "f1": torch.tensor(f1_score(predicted_actions, targets, average="macro")),
            "perplexity": torch.exp(loss),
        }
        return losses

    def action_dists(self, batch):
        self.device = next(self.parameters()).device
        if self.add_root:
            model_embeddings = self.add_root_model_embeddings(batch)[:, 0, :, :].to(
                self.device
            )

        emb_pairs = torch.cat(
            (
                model_embeddings[
                    np.array(
                        [
                            torch.arange(model_embeddings.shape[0]).cpu(),
                            batch["node1s"].cpu(),
                        ]
                    )
                ],
                model_embeddings[
                    np.array(
                        [
                            torch.arange(model_embeddings.shape[0]).cpu(),
                            batch["node2s"].cpu(),
                        ]
                    )
                ],
            ),
            dim=1,
        )

        return self.forward(emb_pairs)

    def batch_step_train(self, batch):
        self.device = next(self.parameters()).device

        if self.add_root:
            model_embeddings = self.add_root_model_embeddings(batch)[:, 0, :, :].to(
                self.device
            )

        oracle_action_idxs, targets = self.oracle.targets_idxs(batch)

        first_emb_indx, second_emb_indx = (
            oracle_action_idxs[[0, 1], :],
            oracle_action_idxs[[0, 2], :],
        )
        emb_pairs = torch.cat(
            (model_embeddings[first_emb_indx], model_embeddings[second_emb_indx]), dim=1
        )

        output_distributions = self.forward(emb_pairs)
        return {
            "loss": self.nll(
                output_distributions, torch.tensor(targets, device=self.device)
            ).mean()
        }


class GeometricProbe(IncrementalProbe):
    def __init__(self, args):
        super(IncrementalProbe, self).__init__()
        IncrementalProbe.__init__(self, args)

        self.loss_types = args["loss_types"]
        self.verbose = args["verbose"]
        self.threshold = args["threshold"]
        self.temp = args["temp"]
        self.num_layers = args["num_layers"]

        layers = [
            nn.Sequential(
                nn.Linear(self.probe_rank, self.probe_rank, bias=False),
                nn.ReLU(),
                nn.Dropout(self.layer_dropout_rate),
            )
            for layer_idx in range(self.num_layers - 1)
        ]
        layers = (
            [nn.Dropout(self.embeddings_dropout_rate)]
            + layers
            + [nn.Linear(self.probe_rank, self.probe_rank, bias=False)]
        )
        self.transform = nn.Sequential(*layers)
        if self.checkpoint_path:
            print(f"Loading checkpoint from {self.checkpoint_path}")
            check_probe = (
                experiment.IncrementalParseProbeExperiment.load_from_checkpoint(
                    self.checkpoint_path
                ).probe
            )
            self.transform = copy.deepcopy(check_probe.transform)
            if self.add_root:
                self.root = copy.deepcopy(check_probe.root)
        self.device = next(self.parameters()).device

    def t_sigmoid(self, x, threshold=1.5, temp=0.1):
        return torch.sigmoid((x - threshold) / (temp)).clamp(min=1e-7, max=1 - 1e-7)

    def p_shift(self, model_embeddings, temp, threshold):
        return self.t_sigmoid(
            self.distance_matrix(model_embeddings), threshold=self.threshold, temp=temp
        )

    def marginal_p_reduce(self, model_embeddings, temp):
        return self.t_sigmoid(
            self.depth_matrix(model_embeddings), threshold=0, temp=temp
        )

    def forward_distance(self, batch, add_root=True):
        transformed = self.transform(batch)
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1, 2)
        diffs = transformed - transposed
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)
        return squared_distances

    def forward_depth(self, batch):
        transformed = self.transform(batch)
        batchlen, seqlen, rank = transformed.size()
        norms = torch.bmm(
            transformed.view(batchlen * seqlen, 1, rank),
            transformed.view(batchlen * seqlen, rank, 1),
        )
        norms = norms.view(batchlen, seqlen)
        return norms

    def distance_matrix(self, batch):
        return self.forward_distance(batch)

    def depth_matrix(self, batch):
        predictions = self.forward_depth(batch)
        return predictions[..., None] - predictions[..., None, :]

    def L1DistanceLoss(self, predictions, label_batch, length_batch):
        """Computes L1 loss on distance matrices.

        Ignores all entries where label_batch=-1
        Normalizes first within sentences (by dividing by the square of the sentence length)
        and then across the batch.

        Args:
        predictions: A pytorch batch of predicted distances
        label_batch: A pytorch batch of true distances
        length_batch: A pytorch batch of sentence lengths
        Returns:
        A tuple of:
            batch_loss: average loss in the batch
            total_sents: number of sentences in the batch
        """
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        total_sents = torch.sum((length_batch != 0)).float()
        squared_lengths = length_batch.pow(2).float()
        if total_sents > 0:
            loss_per_sent = torch.sum(
                torch.abs(predictions_masked - labels_masked), dim=(1, 2)
            )
            normalized_loss_per_sent = loss_per_sent / squared_lengths
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        else:
            batch_loss = torch.tensor(0.0, device=self.device)
        return batch_loss

    def L1DepthLoss(self, predictions, label_batch, length_batch):
        """Computes L1 loss on depth sequences.

        Ignores all entries where label_batch=-1
        Normalizes first within sentences (by dividing by the sentence length)
        and then across the batch.

        Args:
        predictions: A pytorch batch of predicted depths
        label_batch: A pytorch batch of true depths
        length_batch: A pytorch batch of sentence lengths
        Returns:
        A tuple of:
            batch_loss: average loss in the batch
            total_sents: number of sentences in the batch
        """
        total_sents = torch.sum(length_batch != 0).float()
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        if total_sents > 0:
            loss_per_sent = torch.sum(
                torch.abs(predictions_masked - labels_masked), dim=1
            )
            normalized_loss_per_sent = loss_per_sent / length_batch.float()
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        else:
            batch_loss = torch.tensor(0.0, device=self.device)
        return batch_loss

    def dist_spearmanr(self, predictions, label_batch, length_batch):
        dist_lengths_to_spearmanrs = defaultdict(list)
        for prediction, label, length in zip(
            predictions.detach().cpu().numpy(), label_batch, length_batch
        ):
            length = int(length)
            prediction = prediction[:length, :length]
            label = label[:length, :length].cpu()
            dist_spearmanrs = [
                spearmanr(pred, gold) for pred, gold in zip(prediction, label)
            ]
            dist_lengths_to_spearmanrs[length].extend(
                [x.correlation for x in dist_spearmanrs]
            )
        dist_mean_spearman_for_each_length = {
            length: np.mean(dist_lengths_to_spearmanrs[length])
            for length in dist_lengths_to_spearmanrs
        }
        return np.mean(
            [
                dist_mean_spearman_for_each_length[x]
                for x in range(5, 51)
                if x in dist_mean_spearman_for_each_length
            ]
        )

    def dep_spearmanr(self, depth_predictions, depth_label_batch, depth_length_batch):
        depth_lengths_to_spearmanrs = defaultdict(list)
        for prediction, label, length in zip(
            depth_predictions.detach().cpu().numpy(),
            depth_label_batch,
            depth_length_batch,
        ):
            length = int(length)
            prediction = prediction[:length]
            label = label[:length].cpu()
            depth_sent_spearmanr = spearmanr(prediction, label)
            depth_lengths_to_spearmanrs[length].append(depth_sent_spearmanr.correlation)

        depth_mean_spearman_for_each_length = {
            length: np.mean(depth_lengths_to_spearmanrs[length])
            for length in depth_lengths_to_spearmanrs
        }
        return np.mean(
            [
                depth_mean_spearman_for_each_length[x]
                for x in range(5, 51)
                if x in depth_mean_spearman_for_each_length
            ]
        )

    def root_accuracy_spanning_tree(
        self, depth_predictions, depth_label_batch, depth_length_batch, tags
    ):
        """Computes the root prediction accuracy and writes to disk.
        For each sentence in the corpus, the root token in the sentence
        should be the least deep
        Args:
        batch: A sequence of observations
        """
        correct_root_predictions = 0
        total_sents = 0
        print(depth_label_batch.shape, "depth label batch shape")
        print(depth_length_batch.shape, "depth length batch shape")
        for tag, prediction, label, length in zip(
            tags,
            depth_predictions.detach().cpu().numpy(),
            depth_label_batch,
            depth_length_batch,
        ):
            length = int(length)
            prediction = prediction[1 : length + 1]

            label = label[1 : length + 1].cpu().numpy().tolist()
            poses = tag

            correct_root_predictions += label.index(1) == get_nopunct_argmin(
                prediction, poses
            )
            total_sents += 1
        return correct_root_predictions / float(total_sents)

    def uuas_spanning_tree(self, predictions, label_batch, length_batch, tags):
        """Computes the UUAS score for a batch.
        From the true and predicted distances, computes a minimum spanning tree
        of each, and computes the percentage overlap between edges in all
        predicted and gold trees."""
        uspan_correct = 0
        uspan_total = 0
        total_sents = 0
        for tag, prediction, label, length in zip(
            tags, predictions.detach().cpu().numpy(), label_batch, length_batch
        ):
            length = int(length)
            prediction = prediction[1 : length + 1, 1 : length + 1]
            label = label[1 : length + 1, 1 : length + 1].cpu()
            poses = tag
            gold_edges = prims_matrix_to_edges(label, poses)
            pred_edges = prims_matrix_to_edges(prediction, poses)
            uspan_correct += len(
                set([tuple(sorted(x)) for x in gold_edges]).intersection(
                    set([tuple(sorted(x)) for x in pred_edges])
                )
            )
            uspan_total += len(gold_edges)
            total_sents += 1
        uuas = uspan_correct / float(uspan_total)
        return uuas

    def batch_step_eval(self, batch):
        if "lengths" in batch:
            max_tok_length = batch["lengths"].max()
            batch["padded_embeddings"] = batch["padded_embeddings"][
                :, :, : max_tok_length + 4, :
            ]
            batch["gold_distances"] = batch["gold_distances"][
                :, : max_tok_length + 4, : max_tok_length + 4
            ]
            batch["gold_depths"] = batch["gold_depths"][:, : max_tok_length + 4]

        self.device = next(self.parameters()).device

        if self.add_root:
            model_embeddings = self.add_root_model_embeddings(batch)[:, 0, :, :].to(
                self.device
            )

            gold_distances = self.add_root_distance_labels(batch)[:, :-1, :-1].to(
                self.device
            )
            gold_depths = self.add_root_depth_labels(batch)[:, :-1].to(self.device)

        else:
            model_embeddings = batch["padded_embeddings"][:, 0, 1:, :].to(self.device)
            gold_distances = batch["gold_distances"][:, :-1, :-1].to(self.device)
            gold_depths = batch["gold_depths"][:, :-1].to(self.device)

        lengths = batch["lengths"].to(self.device)

        losses = {
            "L2": torch.linalg.norm(self.transform(model_embeddings)),
            "temperature": torch.tensor(self.temp, device=self.device),
        }

        distance_predictions = self.forward_distance(model_embeddings)
        depth_predictions = self.forward_depth(model_embeddings)
        losses["distance_mse"] = self.L1DistanceLoss(
            distance_predictions, gold_distances, lengths
        )
        losses["depth_mse"] = self.L1DepthLoss(depth_predictions, gold_depths, lengths)

        action_dists = self.oracle.action_dists(
            self.p_shift(model_embeddings, temp=self.temp, threshold=self.threshold),
            self.marginal_p_reduce(model_embeddings, temp=self.temp),
        )
        oracle_action_idxs, targets = self.oracle.targets_idxs(batch)

        losses["oracle_action_nll"] = self.nll(
            action_dists[oracle_action_idxs], torch.tensor(targets, device=self.device)
        ).mean()

        predicted_actions = (
            action_dists[oracle_action_idxs].argmax(dim=-1).detach().cpu().numpy()
        )

        losses["f1"] = torch.tensor(
            f1_score(predicted_actions, targets, average="macro")
        )
        losses["accuracy"] = torch.tensor(accuracy_score(predicted_actions, targets))
        losses["perplexity"] = torch.exp(losses["oracle_action_nll"].detach())
        losses["uuas_spanning_tree"] = torch.tensor(
            self.uuas_spanning_tree(
                distance_predictions, gold_distances, lengths, batch["xpos"]
            )
        )
        losses["root_accuracy_spanning_tree"] = torch.tensor(
            self.root_accuracy_spanning_tree(
                depth_predictions, gold_depths, lengths, batch["xpos"]
            )
        )
        losses["dep_spearman"] = torch.tensor(
            self.dep_spearmanr(depth_predictions, gold_depths, lengths)
        )
        losses["dist_spearman"] = torch.tensor(
            self.dist_spearmanr(distance_predictions, gold_distances, lengths)
        )

        losses["loss"] = sum(losses[loss_type] for loss_type in self.loss_types)

        for key in losses:
            losses[key] = losses[key].detach()

        return losses

    def action_dists(self, batch):
        self.device = next(self.parameters()).device
        if self.add_root:
            model_embeddings = self.add_root_model_embeddings(batch)[:, 0, :, :]

        action_dists = self.oracle.action_dists(
            self.p_shift(model_embeddings, temp=self.temp, threshold=self.threshold),
            self.marginal_p_reduce(model_embeddings, temp=self.temp),
        )

        return action_dists[
            np.array(
                [
                    torch.arange(model_embeddings.shape[0]).cpu(),
                    batch["node1s"].cpu(),
                    batch["node2s"].cpu(),
                ]
            )
        ]

    def batch_step_train(self, batch):
        if "lengths" in batch:
            max_tok_length = batch["lengths"].max()
            batch["padded_embeddings"] = batch["padded_embeddings"][
                :, :, : max_tok_length + 4, :
            ]
            batch["gold_distances"] = batch["gold_distances"][
                :, : max_tok_length + 4, : max_tok_length + 4
            ]
            batch["gold_depths"] = batch["gold_depths"][:, : max_tok_length + 4]

        self.device = next(self.parameters()).device

        if self.add_root:
            model_embeddings = self.add_root_model_embeddings(batch)[:, 0, :, :].to(
                self.device
            )

            if self.args["probe_name"] == "Geometric_Regression":
                gold_distances = self.add_root_distance_labels(batch)[:, :-1, :-1].to(
                    self.device
                )
                gold_depths = self.add_root_depth_labels(batch)[:, :-1].to(self.device)
                lengths = batch["lengths"].to(self.device)

        else:
            model_embeddings = batch["padded_embeddings"][:, 0, 1:, :].to(self.device)
            if self.args["probe_name"] == "Geometric_Regression":
                gold_distances = batch["gold_distances"][:, :-1, :-1].to(self.device)
                gold_depths = batch["gold_depths"][:, :-1].to(self.device)
                lengths = batch["lengths"].to(self.device)

        losses = {
            "L2": torch.linalg.norm(self.transform(model_embeddings)),
            "temperature": torch.tensor(self.temp, device=self.device),
        }

        if self.args["probe_name"] == "Geometric_Regression":
            distance_predictions = self.forward_distance(model_embeddings)
            depth_predictions = self.forward_depth(model_embeddings)
            losses["distance_mse"] = self.L1DistanceLoss(
                distance_predictions, gold_distances, lengths
            )
            losses["depth_mse"] = self.L1DepthLoss(
                depth_predictions, gold_depths, lengths
            )

        else:
            action_dists = self.oracle.action_dists(
                self.p_shift(
                    model_embeddings, temp=self.temp, threshold=self.threshold
                ),
                self.marginal_p_reduce(model_embeddings, temp=self.temp),
            )
            oracle_action_idxs, targets = self.oracle.targets_idxs(batch)

            losses["oracle_action_nll"] = self.nll(
                action_dists[oracle_action_idxs],
                torch.tensor(targets, device=self.device),
            ).mean()
        losses["loss"] = sum(losses[loss_type] for loss_type in self.loss_types)

        for key in losses:
            if not key == "loss":
                losses[key] = losses[key].detach()

        return losses
