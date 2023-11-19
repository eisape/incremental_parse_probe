import torch.optim as optim
import pytorch_lightning as pl

class IncrementalParseProbeExperiment(pl.LightningModule):
    def __init__(self, probe=None, params: dict = None) -> None:
        super(IncrementalParseProbeExperiment, self).__init__()
        self.save_hyperparameters()
        self.probe = probe
        self.params = params
        self.curr_device = 'cuda'
        self.hold_graph = False
        try: self.hold_graph = self.params['retain_first_backpass']
        except: pass

    def format_batch(self, batch):
        token_ids, gold_stacks, gold_buffers, action_ids, padded_action_ngrams, padded_embeddings, gold_distances, gold_depths ,lengths, gold_tuples, cont_masks, xpos = batch
        return {'token_ids': token_ids.to('cuda'), #batch_size x token_pad
                'gold_stacks': gold_stacks.to('cuda'), #batch_size x token_pad
                'gold_buffers': gold_buffers.to('cuda'), #batch_size x token_pad
                'action_ids': action_ids.to('cuda'), #batch_size x action_pad
                'padded_action_ngrams': padded_action_ngrams.to('cuda'),#batch_size x token_pad x action_ngram_pad
                'padded_embeddings': padded_embeddings.to('cuda'), #batch_size x model_layers x token_pad x feature_count
                'gold_distances': gold_distances.to('cuda'), #matrix of distances (batch_size x token_pad x token_pad)
                'gold_depths': gold_depths.to('cuda'), #matrix of depths (batch_size x token_pad x token_pad)
                'lengths': lengths.to('cuda'),
                'gold_tuples': gold_tuples.to('cuda'),
                'continuous_action_masks': cont_masks.to('cuda'), 
                'xpos': xpos.to('cuda') # xpos for evaluation
                }  

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        train_loss = self.probe.batch_step_train(self.format_batch(batch))
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        val_loss = self.probe.batch_step_eval(self.format_batch(batch))
        val_loss['loss'] = val_loss['loss'].detach()
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
     
    def on_validation_end(self) -> None: return None

    def configure_optimizers(self):
        optimizer = getattr(optim, self.params['optimizer_type'])(filter(lambda p: p.requires_grad, self.probe.parameters()), **self.params['optimizer_params'])
        scheduler = getattr(optim.lr_scheduler, self.params['scheduler_type'])(optimizer, **self.params['scheduler_params'])
        if self.params['scheduler_type'] == 'ReduceLROnPlateau': return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
        else: return {'optimizer': optimizer, 'lr_scheduler': scheduler}


