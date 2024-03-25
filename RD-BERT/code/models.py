import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import data

from transformers import AutoModel, AutoConfig, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import Seq2SeqLMOutput

class AraT5RevDict(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        if args.resume_train:
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained(args.resume_file)
            raise NotImplementedError()
        else:
            if args.from_pretrained:
                self.base_model = AutoModelForSeq2SeqLM.from_pretrained("UBC-NLP/AraT5v2-base-1024")
            else:
                model_config = AutoConfig.from_pretrained("UBC-NLP/AraT5v2-base-1024")
                self.base_model = AutoModelForSeq2SeqLM.from_config(model_config)
        
        self.linear = nn.Linear(self.base_model.config.hidden_size, args.max_len)

    def forward(self, input_ids, attention_mask, labels):
        outputs:Seq2SeqLMOutput = self.base_model(input_ids=input_ids, 
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )        

        pooled_emb = (outputs.encoder_last_hidden_state * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(1)

        embedding = self.linear(pooled_emb)
        return outputs.loss, embedding     

    def save(self, file):
        torch.save(self, file)
        print("\n--\nsave1\n--\n")

    @staticmethod
    def load(file):
        return torch.load(file)

class ARBERTRevDict(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        if args.resume_train:
            self.base_model = AutoModel.from_pretrained(args.resume_file)
            raise NotImplementedError()
        else:
            if args.from_pretrained:
                self.base_model = AutoModel.from_pretrained(args.model_name)
            else:
                model_config = AutoConfig.from_pretrained(args.model_name)
                self.base_model = AutoModel.from_config(model_config)
        
        self.linear = nn.Linear(self.base_model.config.hidden_size, args.max_len)

    def forward(self, input_ids, token_type_ids , attention_mask):
        feats = self.base_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        embedding = self.linear(feats)
        return embedding     

    def save(self, file):
        self.base_model.save_pretrained(file,from_pt=True)
        print("\n--\nsave_pretrained\n--\n")
        # torch.save(self, file)

    @staticmethod
    def load(file):
        return AutoModel.from_pretrained(file)
    
class PositionalEncoding(nn.Module):
    """From PyTorch"""

    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)



class RevdictModel(nn.Module):
    """A transformer architecture for Reverse Dictionary"""

    def __init__(
        self, vocab, d_model=256, n_head=4, n_layers=4, dropout=0.3, maxlen=512
    ):
        super(RevdictModel, self).__init__()
        self.d_model = d_model
        self.padding_idx = vocab[data.PAD]
        self.eos_idx = vocab[data.EOS]
        self.maxlen = maxlen

        self.embedding = nn.Embedding(len(vocab), d_model, padding_idx=self.padding_idx)
        self.positional_encoding = PositionalEncoding(
            d_model, dropout=dropout, max_len=maxlen
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dropout=dropout, dim_feedforward=d_model * 2
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.dropout = nn.Dropout(p=dropout)
        self.e_proj = nn.Linear(d_model, d_model)
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            else:  # gain parameters of the layer norm
                nn.init.ones_(param)

    def forward(self, gloss_tensor):
        src_key_padding_mask = gloss_tensor == self.padding_idx
        embs = self.embedding(gloss_tensor)
        src = self.positional_encoding(embs)
        transformer_output = self.dropout(
            self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask.t())
        )
        summed_embs = transformer_output.masked_fill(
            src_key_padding_mask.unsqueeze(-1), 0
        ).sum(dim=0)
        return self.e_proj(F.relu(summed_embs))

    @staticmethod
    def load(file):
        return torch.load(file)

    def save(self, file):
        torch.save(self, file)
        print("\n--\nsave2\n--\n")
