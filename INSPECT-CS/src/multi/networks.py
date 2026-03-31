# This file contains the implementation of the networks used in the model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math, copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# Soft Attention Mechanism
class SoftAttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(SoftAttentionLayer, self).__init__()
        self.input_size = input_size
        self.classifier = nn.Linear(input_size, 1)
        # initialize self.classifier with mean 0 and bias 1
        with torch.no_grad():
            self.classifier.weight.fill_(0)
            self.classifier.bias.fill_(1)

    def forward(self, inputs): 
        # get batch size and sequence length from inputs
        batch_size, seq_len, _ = inputs.size()
        # calculate attention scores
        scores = self.classifier(inputs).view(batch_size, seq_len)
        # apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)

        # compute the weighted mean
        attention_weights = attention_weights.unsqueeze(2)
        weighted_sum = torch.bmm(inputs.transpose(1, 2), attention_weights).squeeze(2)

        return weighted_sum 
    
    def get_attention_weights(self, inputs):
        # get batch size and sequence length from inputs
        batch_size, seq_len, _ = inputs.size()
        scores = self.classifier(inputs).view(batch_size, seq_len)
        attention_weights = F.softmax(scores, dim=1)
        return attention_weights

# MHA; based on HF implementation
class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, ctx_dim=None):
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = hidden_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(context))
        value_layer = self.transpose_for_scores(self.value(context))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # mask: 1, 0; same setup as HF
            assert attention_mask.unique().sum() == 1
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)


        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob) -> None:
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dropout(self.dense(hidden_states))
        return hidden_states + input_tensor

class BertSelfAttnLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super().__init__()

        self.attn = BertAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = BertAttOutput(hidden_size, attention_probs_dropout_prob)

    def forward(self, input_tensor, attention_mask=None):
        output = self.attn(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output

class BertCrossAttnLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, ctx_dim=None):
        super().__init__()

        self.attn = BertAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob, ctx_dim)
        self.output = BertAttOutput(hidden_size, attention_probs_dropout_prob)

    def forward(self, input_tensor, ctx_tensor, attention_mask=None):
        output = self.attn(input_tensor, ctx_tensor, attention_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output

class BertMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
            nn.Dropout(hidden_dropout_prob)
        )
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.mlp(self.LayerNorm(input_tensor))
        return hidden_states + input_tensor


class BertEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout):
        super().__init__()

        self.mha = BertSelfAttnLayer(hidden_size, num_attention_heads, dropout)
        self.mlp = BertMLP(hidden_size, intermediate_size, dropout)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.mha(hidden_states, attention_mask)
        layer_output = self.mlp(attention_output)
        return layer_output


class CrossEncoderLayer(nn.Module):
    def __init__(self, img_size, txt_size, num_attention_heads, dropout):
        super().__init__()

        self.ts2txt_cross_attention = BertAttention(img_size, num_attention_heads, dropout, txt_size)
        self.txt2img_cross_attention = BertAttention(txt_size, num_attention_heads, dropout, img_size)

        self.img_self_encoder = BertEncoderLayer(img_size, num_attention_heads, img_size, dropout)
        self.txt_self_encoder = BertEncoderLayer(txt_size, num_attention_heads, txt_size, dropout)

    def forward(self, hidden_states_img, hidden_states_txt, attention_mask_img=None, attention_mask_txt=None):
        # not attending to cls token
        if attention_mask_img is not None:
            attention_mask_img[:, 0] = 0
        if attention_mask_txt is not None:
            attention_mask_txt[:, 0] = 0

        new_img = self.ts2txt_cross_attention(hidden_states_img, hidden_states_txt, attention_mask_txt)
        new_img = self.img_self_encoder(new_img, attention_mask_img)
        
        new_txt = self.txt2img_cross_attention(hidden_states_txt, hidden_states_img, attention_mask_img)
        new_txt = self.txt_self_encoder(new_txt, attention_mask_txt)

        return new_img, new_txt 

class SingleCrossEncoderLayer(nn.Module):
    def __init__(self, img_size, txt_size, num_attention_heads, dropout):
        super().__init__()

        self.ts2txt_cross_attention = BertAttention(img_size, num_attention_heads, dropout, txt_size)

        self.img_self_encoder = BertEncoderLayer(img_size, num_attention_heads, img_size, dropout)

    def forward(self, hidden_states_img, hidden_states_txt, attention_mask_img=None, attention_mask_txt=None):
        # not attending to cls token
        if attention_mask_img is not None:
            attention_mask_img[:, 0] = 0
        if attention_mask_txt is not None:
            attention_mask_txt[:, 0] = 0

        new_img = self.ts2txt_cross_attention(hidden_states_img, hidden_states_txt, attention_mask_txt)
        new_img = self.img_self_encoder(new_img, attention_mask_img)

        return new_img 
    
class BertEncoder(nn.Module):
    def __init__(self, config, hidden_size, num_layer):
        super().__init__()

        layer = BertEncoderLayer(
            hidden_size=hidden_size,
            num_attention_heads=config.model.fusion.num_attention_heads,
            intermediate_size=int(hidden_size*config.model.fusion.intermediate_multiplier),
            dropout=config.model.fusion.dropout,
        )
        self.hidden_size = hidden_size

        self.layers = _get_clones(layer, num_layer)

        self.pos_type = 'sinusoid'
        self._init_pos_embed(self.pos_type)


    def _init_pos_embed(self, pos_type='sinusoid'):

        max_position_embeddings = 512

        if pos_type == 'absolute':
            self.position_embeddings = nn.Embedding(max_position_embeddings, self.hidden_size)
            self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        elif pos_type == 'sinusoid':
            d_model = self.hidden_size
            pe = torch.zeros(max_position_embeddings, d_model)
            position = torch.arange(0, max_position_embeddings, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe, persistent=False)

    def forward(self, hidden_states, attention_mask=None):

        length = hidden_states.size(1)
        hidden_states = hidden_states + self.pe[:, :length]

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states
    
class CrossEncoder(nn.Module):
    def __init__(self, config, double_cross=True):
        super().__init__()
        self.double_cross = double_cross
        if double_cross:
            layer = CrossEncoderLayer(
                img_size=config.model.fusion.embed_size,
                txt_size=config.model.fusion.embed_size,
                num_attention_heads=config.model.fusion.num_attention_heads,
                dropout=config.model.fusion.dropout
            )
        else:
            layer = SingleCrossEncoderLayer(
                img_size=config.model.fusion.embed_size,
                txt_size=config.model.fusion.embed_size,
                num_attention_heads=config.model.fusion.num_attention_heads,
                dropout=config.model.fusion.dropout
            )

        self.img_size = config.model.fusion.embed_size
        self.txt_size = config.model.fusion.embed_size

        self.layers = _get_clones(layer, config.model.fusion.num_layer_cross)

    def forward(self, hidden_img, hidden_txt, mask_img=None, mask_txt=None):
        
        for layer in self.layers:
            if self.double_cross:
                hidden_img, hidden_txt = layer(hidden_img, hidden_txt, mask_img, mask_txt)
            else:
                hidden_img = layer(hidden_img, hidden_txt, mask_img, mask_txt)

                return hidden_img

        return hidden_img, hidden_txt 

# Cross-Attention fusion
class CrossFusionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self._init_layers(config)
    
    def _init_layers(self, config):
        embed_size = config.model.fusion.embed_size
        modalities = config.modalities  # e.g., ['img', 'txt'] or ['img', 'txt', 'ehr']

        # Report soft attention pooling
        if "report" in config.modalities:
            self.report_soft_attention = SoftAttentionLayer(config.model.report_size)

        # Embedding layers
        self.embeddings = nn.ModuleDict()
        for mod in modalities:
            orig_size = getattr(config.model, f"{mod}_size")
            if orig_size != config.model.fusion.embed_size:
                self.embeddings[mod] = nn.Linear(orig_size, embed_size)
            else:
                self.embeddings[mod] = nn.Identity()

        # Store encoders in a dict dynamically
        self.encoders = nn.ModuleDict()
        for mod in modalities:
            num_layer = getattr(config.model.fusion, f'num_layer_{mod}', 1)  # default 1 if not defined
            self.encoders[mod] = BertEncoder(config, embed_size, num_layer)

        # Cross encoders: we create n-1 cross encoders if n modalities
        self.cross_encoders = nn.ModuleList()
        for _ in range(len(modalities) - 1):
            self.cross_encoders.append(CrossEncoder(config, double_cross=False))

        # Output layer
        self.output_layer = nn.Linear(embed_size, 1)

    def forward(self, *args, return_features=False):
        """
        args: tensori delle modalità seguiti opzionalmente dalle attention mask
            es. (x1, x2, mask1, mask2) o (x1, x2, x3, mask1, mask2, mask3)
        """
        n_modalities = len(self.encoders)
        modalities = list(self.encoders.keys())

        # Report sof attention pooling
        new_args = []
        for i, mod in enumerate(modalities):
            if mod == "report":
                tmp = self.report_soft_attention(args[i].view(-1, *args[i].shape[2:]))
                new_args.append(tmp.view(*args[i].shape[:2], -1))
            else:
                new_args.append(args[i])

        # Split inputs and masks
        inputs = new_args[:n_modalities]
        attn_masks = args[n_modalities:] if len(args) > n_modalities else []

        inputs_dict = {mod: x for mod, x in zip(modalities, inputs)}
        attn_masks_dict = {mod: m for mod, m in zip(modalities, attn_masks)} if attn_masks else None

        # Embed dimension for each modality
        embedding_dict = {mod: self.embeddings[mod](x) for mod, x in inputs_dict.items()}

        # Encode each modality
        hidden_dict = {mod: self.encoders[mod](x) for mod, x in embedding_dict.items()}

        features = {}
        if return_features:
            features['first'] = hidden_dict[modalities[0]]
            features['second'] = hidden_dict[modalities[1]]
            if len(modalities) == 3:
                features['third'] = hidden_dict[modalities[2]]

        # Cross attention fusion 
        current_hidden = hidden_dict[modalities[0]]
        current_mask = attn_masks_dict[modalities[0]] if attn_masks_dict else None

        for i in range(1, len(modalities)):
            next_hidden = hidden_dict[modalities[i]]
            next_mask = attn_masks_dict[modalities[i]] if attn_masks_dict else None

            cross_idx = min(i - 1, len(self.cross_encoders) - 1)
            current_hidden = self.cross_encoders[cross_idx](
                current_hidden, next_hidden, current_mask, next_mask
            )
            if return_features:
                features[f'cross_{i}'] = current_hidden

        # Mean pooling on axis 1
        final_hidden = current_hidden.mean(dim=1)

        # Output layer
        logits = self.output_layer(final_hidden)

        if return_features:
            return logits, 0.0, features
        return logits, 0.0


# ARMOUR fusion (without contrastive)
class FusionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self._init_cls(config)
        self._init_layers(config)

    def _init_layers(self, config):
        # settings
        embed_size = config.model.fusion.embed_size
        img_num_layer = config.model.fusion.num_layer_img
        txt_num_layer = config.model.fusion.num_layer_txt

        # encoders 
        self.img_encoder = BertEncoder(config, embed_size, img_num_layer)
        self.txt_encoder= BertEncoder(config, embed_size, txt_num_layer)

        self.cross_encoder = CrossEncoder(config)

    def _init_cls(self, config):
        # cls tokens 
        cls_img = Parameter(torch.randn(1, 1, config.model.fusion.embed_size))
        cls_txt= Parameter(torch.randn(1, 1, config.model.fusion.embed_size))
        torch.nn.init.xavier_uniform_(cls_img.data)
        torch.nn.init.xavier_uniform_(cls_txt.data)
        self.register_parameter('cls_img', cls_img)
        self.register_parameter('cls_txt', cls_txt)

    def _append_cls(self, img_input, txt_input, img_attn_mask, txt_attn_mask):

        bsize = img_input.size(0)
        cls_img = self.cls_img.expand(bsize, -1, -1)
        img_input = torch.cat([cls_img, img_input], 1)
        cls_txt= self.cls_txt.expand(bsize, -1, -1)
        txt_input= torch.cat([cls_txt, txt_input], 1)

        mask_one = torch.ones(bsize, 1)
        if img_attn_mask is not None:
            img_attn_mask = torch.cat([mask_one.type_as(img_attn_mask), img_attn_mask], 1)
        if txt_attn_mask is not None:
            txt_attn_mask = torch.cat([mask_one.type_as(txt_attn_mask), txt_attn_mask], 1)

        return img_input, txt_input, img_attn_mask, txt_attn_mask


    def forward(self, img_input, txt_input, img_attn_mask=None, txt_attn_mask=None, return_features=False):
        # cls 
        img_input, txt_input, img_attn_mask, txt_attn_mask = self._append_cls(img_input, txt_input, img_attn_mask, txt_attn_mask)

        # encode image and text
        hidden_img = self.img_encoder(img_input)
        hidden_txt = self.txt_encoder(txt_input)
        
        # cross attention
        hidden_img, hidden_txt = self.cross_encoder(hidden_img, hidden_txt, img_attn_mask, txt_attn_mask)

        # output
        cls_img = hidden_img[:, 0]
        cls_txt= hidden_txt[:, 0]

        final_hidden = torch.cat([cls_img, cls_txt], -1)

        if return_features:
            return final_hidden, 0.0, {'cls_img': cls_img, 'cls_txt': cls_txt}
        return final_hidden, 0.0

# ARMOUR fusion
class ContrastFusionModel(FusionModel):
    def __init__(self, config) -> None:
        super().__init__(config)

        self._init_cls(config)
        self._init_layers(config)

        if config.model.fusion.add_contrast:
            self._init_contrast(config)
            self._init_momentum_layers(config)
    
    def _init_cls(self, config):
        # cls tokens 
        cls_img = Parameter(torch.randn(1, 1, config.model.fusion.embed_size))
        cls_txt= Parameter(torch.randn(1, 1, config.model.fusion.embed_size))
        torch.nn.init.xavier_uniform_(cls_img.data)
        torch.nn.init.xavier_uniform_(cls_txt.data)
        self.register_parameter('cls_img', cls_img)
        self.register_parameter('cls_txt', cls_txt)

    def _init_layers(self, config):
        # settings
        embed_size = config.model.fusion.embed_size
        img_num_layer = config.model.fusion.num_layer_img
        txt_num_layer = config.model.fusion.num_layer_txt

        # encoders 
        self.img_encoder = BertEncoder(config, embed_size, img_num_layer)
        self.txt_encoder= BertEncoder(config, embed_size, txt_num_layer)
        self.cross_encoder = CrossEncoder(config)

    def _init_contrast(self, config):

        embed_dim = config.model.fusion.contrast_embed_dim

        self.img_proj = nn.Linear(config.model.fusion.embed_size, embed_dim)
        self.txt_proj = nn.Linear(config.model.fusion.embed_size, embed_dim)

        self.temp = Parameter(torch.ones([]) * config['model']['temp'])

        self.queue_size = config.model.queue_size
        self.momentum = config.model.momentum
        # self.alpha = config['model']['alpha']

        self.train_stage = True

        # create the queue
        self.register_buffer("img_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("txt_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.img_queue = nn.functional.normalize(self.img_queue, dim=0)
        self.txt_queue = nn.functional.normalize(self.txt_queue, dim=0)

    def _init_momentum_layers(self, config):

        embed_size = config.model.fusion.embed_size
        img_num_layer = config.model.fusion.num_layer_img
        txt_num_layer = config.model.fusion.num_layer_txt

        self.img_encoder_m = BertEncoder(config, embed_size, img_num_layer)
        self.txt_encoder_m = BertEncoder(config, embed_size, txt_num_layer)
        
        self.img_proj_m = nn.Linear(embed_size, config.model.fusion.contrast_embed_dim)
        self.txt_proj_m= nn.Linear(embed_size, config.model.fusion.contrast_embed_dim)


        self.model_pairs = [
            [self.img_encoder, self.img_encoder_m],
            [self.txt_encoder, self.txt_encoder_m],
            [self.img_proj, self.img_proj_m],
            [self.txt_proj, self.txt_proj_m],
        ]

        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    def forward(self, img_input, txt_input, img_attn_mask=None, txt_attn_mask=None, return_features=False):
        # constrain temperature
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)  

        # cls 
        img_input, txt_input, img_attn_mask, txt_attn_mask = self._append_cls(img_input, txt_input, img_attn_mask, txt_attn_mask)

        # encode
        hidden_img = self.img_encoder(img_input, img_attn_mask)
        hidden_txt = self.txt_encoder(txt_input, txt_attn_mask)
        
        img_feat = F.normalize(self.img_proj(hidden_img[:, 0]), dim=-1)
        txt_feat = F.normalize(self.txt_proj(hidden_txt[:, 0]), dim=-1)

        # get momentum fts 
        with torch.no_grad():
            self._momentum_update()
            
            hidden_img_m = self.img_encoder_m(img_input, img_attn_mask)
            hidden_txt_m = self.txt_encoder_m(txt_input, txt_attn_mask)

            img_feat_m = F.normalize(self.img_proj_m(hidden_img_m[:, 0]), dim=-1)
            txt_feat_m = F.normalize(self.txt_proj_m(hidden_txt_m[:, 0]), dim=-1)

            img_feat_all = torch.cat([img_feat_m.T, self.img_queue.clone().detach()], dim=1)
            txt_feat_all = torch.cat([txt_feat_m.T, self.txt_queue.clone().detach()], dim=1)

            sim_targets = torch.zeros(img_feat_m.size(0), img_feat_all.size(1)).to(img_feat_m.device)
            sim_targets.fill_diagonal_(1) 

        sim_img2txt = img_feat @ txt_feat_all / self.temp
        sim_txt2ts = txt_feat @ img_feat_all / self.temp

        loss_img2txt = -torch.sum(F.log_softmax(sim_img2txt, dim=1)*sim_targets,dim=1).mean()
        loss_txt2ts = -torch.sum(F.log_softmax(sim_txt2ts, dim=1)*sim_targets,dim=1).mean()

        loss_contrastive = (loss_img2txt+loss_txt2ts)/2
        
        # print(self.train_stage)
        if self.train_stage:
            # no update when forward on val and test
            self._dequeue_and_enqueue(img_feat_m, txt_feat_m)
        # else:
        #     print('skip')

        # cross attn fusion and final output
        hidden_img, hidden_txt = self.cross_encoder(hidden_img, hidden_txt, img_attn_mask, txt_attn_mask)

        # output
        cls_img = hidden_img[:, 0]
        cls_txt= hidden_txt[:, 0]

        final_hidden = torch.cat([cls_img, cls_txt], -1)

        if return_features:
            return final_hidden, loss_contrastive, {'cls_img': cls_img, 'cls_txt': cls_txt}
        return final_hidden, loss_contrastive


    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, img_feat, txt_feat):
        # # gather keys before updating queue
        # img_feats = concat_all_gather(img_feat)
        # txt_feats = concat_all_gather(txt_feat)

        img_feats = img_feat
        txt_feats = txt_feat

        batch_size = img_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # print(ptr, ptr+batch_size, batch_size)
        self.img_queue[:, ptr:ptr + batch_size] = img_feats.T
        self.txt_queue[:, ptr:ptr + batch_size] = txt_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 

# ARMOUR model
class ExpModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.txt_size = config.model.report_size
        self.img_size = config.model.image_size
        self.ehr_size = config.model.ehr_size
        self.embed_size  = config.model.fusion.embed_size
        self.modalities = config.modalities  # eg. ["image", "report", "ehr"]

        # Embedding layers
        self.embeddings = nn.ModuleDict()
        for mod in self.modalities:
            orig_size = getattr(config.model, f"{mod}_size")
            if orig_size != config.model.fusion.embed_size:
                self.embeddings[mod] = nn.Linear(orig_size, self.embed_size)
            else:
                self.embeddings[mod] = nn.Identity()
        
        # Soft attention layer
        self.soft_attention_txt = SoftAttentionLayer(self.txt_size)  

        # Store encoders in a dict dynamically
        self.encoders = nn.ModuleDict()
        for mod in ["image", "report"]:
            num_layer = getattr(config.model.fusion, f'num_layer_{mod}', 1)  # default 1 if not defined
            self.encoders[mod] = BertEncoder(config, self.embed_size, num_layer)

        # Fusion Model
        if config.model.fusion.add_contrast:
            self.fusion = ContrastFusionModel(config)
        else:
            self.fusion = FusionModel(config)

        # Output layer 
        self.output_layer = nn.Linear(self.embed_size*3, 1)

    def forward(self, *args, return_features=False):
        # Report sof attention pooling
        n_modalities = len(self.embeddings) 
        new_args = []
        for i, mod in enumerate(self.modalities):
            if mod == "report":
                tmp = self.soft_attention_txt(args[i].view(-1, *args[i].shape[2:]))
                new_args.append(tmp.view(*args[i].shape[:2], -1))
            else:
                new_args.append(args[i])

        # split inputs and masks
        inputs = new_args[:n_modalities]
        attn_masks = args[n_modalities:] if len(args) > n_modalities else []

        inputs_dict = {mod: x for mod, x in zip(self.modalities, inputs)}
        attn_masks_dict = {mod: m for mod, m in zip(self.modalities, attn_masks)} if attn_masks else None

        # Embed dimension for each modality
        embedding_dict = {mod: self.embeddings[mod](x) for mod, x in inputs_dict.items()}

        # Encode only image and report modalities
        hidden_dict = {mod: self.encoders[mod](x) for mod, x in embedding_dict.items() if mod in ["image", "report"]}
        ehr_embedding = embedding_dict.get("ehr", None)

        # Cross attention fusion 
        hidden_image = hidden_dict.get("image", None)
        hidden_report = hidden_dict.get("report", None) if attn_masks_dict else None
        image_attn_mask = attn_masks_dict.get("image", None) if attn_masks_dict else None
        report_attn_mask = attn_masks_dict.get("report", None) if attn_masks_dict else None

        # Multimodal fusion
        if return_features:
            final_hidden, loss_contrastive, features = self.fusion(hidden_image, hidden_report, image_attn_mask, report_attn_mask, return_features=True)
            if ehr_embedding is not None:
                features['ehr'] = ehr_embedding.squeeze()
                features['fusion'] = final_hidden
        else:
            final_hidden, loss_contrastive = self.fusion(hidden_image, hidden_report, image_attn_mask, report_attn_mask)

        # Concatenate EHR embedding if present
        if ehr_embedding is not None:
            final_hidden = torch.cat([final_hidden, ehr_embedding.squeeze()], dim=1)
        
        # Final output prediction
        logits = self.output_layer(final_hidden)

        if return_features:
            return logits, loss_contrastive, features
        return logits, loss_contrastive     
    
class EarlyFusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.txt_size = config.model.report_size
        self.img_size = config.model.image_size
        self.ehr_size = config.model.ehr_size
        self.fusion_method = config.model.fusion.fusion_method
        self.modalities = config.modalities  # es. ["image", "ehr"] o ["report", "ehr"] o ["image", "report"]

        # Definizione soft attention per le modalità che lo richiedono
        if "image" in self.modalities:
            self.soft_attention_img = SoftAttentionLayer(self.img_size)
        if "report" in self.modalities:   
            self.soft_attention_txt = SoftAttentionLayer(self.txt_size)

        # Calcolo dimensione di input dopo fusione
        if self.fusion_method == "concat":
            input_size = sum(self._get_feature_size(m) for m in self.modalities)
        
        elif self.fusion_method == "average":
            # Proiettiamo tutto allo stesso size (uso l’ehr_size come dimensione comune)
            self.fc_layers = nn.ModuleDict({
                "image": nn.Linear(self.img_size, self.ehr_size),
                "report": nn.Linear(self.txt_size, self.ehr_size),
                "ehr": nn.Identity()  # EHR già nello spazio giusto
            })
            input_size = self.ehr_size

        elif self.fusion_method == "soft":
            self.fc_layers = nn.ModuleDict({
                "image": nn.Linear(self.img_size, self.ehr_size),
                "report": nn.Linear(self.txt_size, self.ehr_size),
                "ehr": nn.Identity()
            })
            self.soft_pooling = SoftAttentionLayer(self.ehr_size)
            input_size = self.ehr_size

        else:
            raise ValueError("Invalid fusion method. Choose 'concat', 'average' or 'soft'.")

        # Output MLP
        self.output_layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def _get_feature_size(self, modality: str):
        if modality == "image":
            return self.img_size
        elif modality == "report":
            return self.txt_size
        elif modality == "ehr":
            return self.ehr_size
        else:
            raise ValueError(f"Unknown modality {modality}")

    def _process_modality(self, x, modality):
        """Applica soft-attention o identity a seconda della modalità"""
        if modality == "image":
            return self.soft_attention_img(x)
        elif modality == "report":
            x_ = x.view(-1, *x.shape[2:])       # [B*S, T, D]
            x_ = self.soft_attention_txt(x_)    # [B*S, D]
            x_ = x_.view(*x.shape[:2], -1)      # [B, S, D]
            return self.soft_attention_txt(x_)  # [B, D]
        elif modality == "ehr":
            return x  # già feature vector
        else:
            raise ValueError(f"Unknown modality {modality}")

    def forward(self, *xs, return_features=False):
        # Check input consistency
        if len(xs) != len(self.modalities):
            raise ValueError(
                f"Expected {len(self.modalities)} inputs, got {len(xs)}."
            )

        # Encode all modalities dynamically
        hs = [self._process_modality(x, m) for x, m in zip(xs, self.modalities)]

        # Fusion strategies
        if self.fusion_method == "concat":
            # Concatenate all modality embeddings
            final_hidden = torch.cat(hs, dim=1)

        elif self.fusion_method == "average":
            # Project embeddings into a common space and average them
            projected = [self.fc_layers[m](h) for h, m in zip(hs, self.modalities)]
            final_hidden = torch.stack(projected, dim=0).mean(dim=0)

        elif self.fusion_method == "soft":
            # Project embeddings, stack them along a new dimension, and apply soft pooling
            projected = [self.fc_layers[m](h).unsqueeze(1) for h, m in zip(hs, self.modalities)]
            final_hidden = self.soft_pooling(torch.cat(projected, dim=1))

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Final output prediction
        logits = self.output_layer(final_hidden)
        
        if return_features:
            features = {m: h for m, h in zip(self.modalities, hs)}
            features["fusion"] = final_hidden
            return logits, 0.0, features
            
        return logits, 0.0
    
def init_model(cfg):
    # instantiate the model
    if cfg.model.name == "armour":
        model = ExpModel(cfg)   
    elif cfg.model.name == "cross":
        model = CrossFusionModel(cfg) 
    elif cfg.model.name == "early":
        model = EarlyFusionModel(cfg)
    else:
        raise ValueError("Invalid model name")
    return model