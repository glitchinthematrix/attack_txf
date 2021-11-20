import numpy as np
import torch
import torch.nn as nn
import transformers
from gpt2 import GPT2Model


class AttackTransformer(nn.Module):

    """
    This model uses GPT to model (Block1, Attack_1, Block_2, Attack_2, ...)
    """

    def __init__(
            self,
            block_dims,
            attack_classes
            hidden_size,
            max_length,
            max_seq_len,
            random_seed = 0,
            **kwargs
    ):
        super().__init__()
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        self.block_dims = block_dims
        self.attack_classes = attack_classes
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

        config = transformers.GPT2Config(
            vocab_size=1, 
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT2Model(config)
        self.embed_time = nn.Embedding(max_seq_len, hidden_size)
        self.embed_block = nn.Linear(self.block_dims, hidden_size)
        self.embed_attack = nn.Embedding(attack_classes, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_attack = nn.Linear(hidden_size, attack_classes)



    def forward(self, blocks, attacks, timestamps, attention_mask=None):

        batch_size, seq_length = blocks.shape[0], blocks.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        block_embeddings = self.embed_block(blocks)
        time_embeddings = self.embed_time(timestamps)
        attack_embeddings = self.embed_attack(attacks)

        block_embeddings = block_embeddings + time_embeddings
        attack_embeddings = attack_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (block_embeddings, attack_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)


        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        attack_pred = self.predict_attack(x[:,0])    # predict next state given state and action

        return attack_pred

    def get_design(self, blocks, attacks, timestamps, attention_mask, **kwargs):

        blocks = blocks.reshape(1, -1, self.block_dims)
        attacks = attacks.reshape(1, -1)
        generations = generations.reshape(1, -1)

        if self.max_length is not None:
            blocks = blocks[:,-self.max_length:]
            attacks = attacks[:,-self.max_length:]
            timestamps = timestamps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-blocks.shape[1]), torch.ones(blocks.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
            blocks = torch.cat([torch.zeros((blocks.shape[0], self.max_length-blocks.shape[1], self.design_dims)), blocks], dim=1).to(dtype=torch.float32)
            attacks = torch.cat([torch.zeros((attacks.shape[0], self.max_length-attacks.shape[1], 1)), perf_to_go],dim=1).to(dtype=torch.long)
            timestamps = torch.cat([torch.zeros((timestamps.shape[0], self.max_length-timestamps.shape[1])), timestamps], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        attack_pred= self.forward(blocks, perf_to_go, timestamps, attention_mask=attention_mask, **kwargs)
        preds = torch.argmax(attack_pred[0,-1],-1)
        
        return preds


