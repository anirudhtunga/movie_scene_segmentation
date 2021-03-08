import torch
import torch.nn as nn
from transformers import LongformerConfig, LongformerModel
from torch.nn import functional as F
from transformers.activations import ACT2FN, gelu
from sklearn.metrics import average_precision_score

configuration = LongformerConfig(attention_window = 4)

class MultimodalLongFormer(nn.Module):
    def __init__(self):
        super(MultimodalLongFormer, self).__init__()
        self.coarse_embedding = nn.Embedding(3, 768)
        self.input_feat_encoding = torch.nn.Linear(3584,768)
        self.model = LongformerModel(configuration)
        self.encoder = self.model.encoder
        self.ln = self.model.embeddings.LayerNorm
        self.dp = self.model.embeddings.dropout
        
        self.pooler = self.model.pooler
        
        self.dense = nn.Linear(768, 768)
        self.layer_norm = nn.LayerNorm(768, eps=self.model.config.layer_norm_eps)
        
        self.decoder = nn.Linear(768, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))
        self.decoder.bias = self.bias
        
        self.position_embedding = nn.Embedding(3100, 768)
        
    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):

        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:

            attention_mask = global_attention_mask + 1
        return attention_mask
    
    def _pad_to_window_size(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    position_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    pad_token_id: int,coarse_ids
    ):
        
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.model.config.attention_window
            if isinstance(self.model.config.attention_window, int)
            else max(self.model.config.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]
        


        padding_len = (attention_window - seq_len % attention_window) % attention_window

        if padding_len > 0:

            if input_ids is not None:
                input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
                position_ids = F.pad(position_ids, (0, padding_len), value=pad_token_id)
                coarse_ids = F.pad(coarse_ids, (0, padding_len), value=2)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len,3584),
                    self.model.config.pad_token_id,
                    dtype=torch.float,
                )
                inputs_embeds_padding = self.input_feat_encoding(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            attention_mask = F.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
            token_type_ids = F.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds,coarse_ids
    
    


    def forward(self, input_feat, coarse, mask, global_mask ):
        input_emb = self.input_feat_encoding(input_feat)
        input_shape = input_emb.size()[:-1]
        
        device = input_emb.device 
        

        
        pos_ids  = torch.arange(input_emb.shape[1], dtype=torch.long,device=device) #add .cuda() and seq length

        
        attention_mask = self.model._merge_to_attention_mask(mask, global_mask)
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        
        padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds,coarse_ids = self._pad_to_window_size(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            inputs_embeds=input_emb,
            pad_token_id=self.model.config.pad_token_id,coarse_ids=coarse
        )
        
        extended_attention_mask: torch.Tensor = self.model.get_extended_attention_mask(attention_mask, input_shape, device)[
            :, 0, 0, :
        ]
        
        coarse_emb = self.coarse_embedding(coarse_ids)       
        pos_emb = self.position_embedding(position_ids)
        
#         print(inputs_embeds.shape,coarse_emb.shape,pos_emb.shape)
        
        embeddings = inputs_embeds + coarse_emb + pos_emb
        embeddings = self.ln(embeddings)
        embeddings = self.dp(embeddings)
        
        
        

        encoder_outputs = self.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        
        sequence_output = encoder_outputs[0]
        
      
        

        
        if padding_len > 0:
            # unpad `sequence_output` because the calling function is expecting a length == input_ids.size(1)
            sequence_output = sequence_output[:, :-padding_len]
            
        x = self.dense(sequence_output)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)
        
        x = x[:,:-1]
        


        

        
        return x