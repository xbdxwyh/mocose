import torch
import torch.nn as nn

import random
import math
import copy
import logging


from transformers import (
    BertPreTrainedModel,
)

from transformers.models.bert.modeling_bert import (
    BertEncoder
)
from transformers.file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    add_code_sample_docstrings,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutput
)

logger = logging.getLogger(__name__)


class EMA(torch.nn.Module):
    """
    [https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage]
    """

    def __init__(self, model, decay,total_step=15000):
        super().__init__()
        self.decay = decay
        self.total_step = total_step
        self.step = 0
        self.model = copy.deepcopy(model).eval()

    def update(self, model):
        self.step = self.step+1
        decay_new = 1-(1-self.decay)*(math.cos(math.pi*self.step/self.total_step)+1)/2
        # 慢慢把self.model往model移动
        with torch.no_grad():
            e_std = self.model.state_dict().values()
            m_std = model.state_dict().values()
            for e, m in zip(e_std, m_std):
                e.copy_(decay_new * e + (1. - decay_new) * m)


class ProjectionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj_layers = config.proj_layers
        self.proj = nn.Sequential()
        
        for i in range(config.proj_layers-1):
            self.mlp.add_module("mlp_"+str(i),nn.Linear(config.hidden_size, config.hidden_size))
            self.mlp.add_module("layer_norm_"+str(i),nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))
            self.mlp.add_module("dropout_"+str(i),nn.Dropout(config.hidden_dropout_prob))
        
        self.dense = nn.Linear(config.hidden_size, config.out_size)
        self.LayerNorm = nn.LayerNorm(config.out_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, x, **kwargs):
        if self.proj_layers > 1:
            x = self.proj(x)
            
        if self.proj_layers > 0:
            x = self.dense(x)
            x = self.LayerNorm(x)
            x = self.dropout(x)
        
        return x


class MLP(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """
    def __init__(self, config):
        super().__init__()
        self.mlp_layers = config.mlp_layers
        self.mlp = nn.Sequential()
        for i in range(config.mlp_layers-1):
            self.mlp.add_module("mlp_"+str(i),nn.Linear(config.out_size, config.out_size))
            self.mlp.add_module("layer_norm_"+str(i),nn.LayerNorm(config.out_size, eps=config.layer_norm_eps))
            self.mlp.add_module("dropout_"+str(i),nn.Dropout(config.hidden_dropout_prob))
        
        self.dense = nn.Linear(config.out_size, config.out_size)
        self.activation = nn.Tanh()

    def forward(self, x, **kwargs):
        if self.mlp_layers > 1:
            x = self.mlp(x)
        
        x = self.dense(x)
        x = self.activation(x)
        
        return x

class PoolerWithoutActive(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, **kwargs):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output


# doc text needed for transformers model

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"


BERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""



class InfoNCEWithQueue(nn.Module):
    def __init__(self,temp=0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)
        self.loss_fct = nn.CrossEntropyLoss()
        
    def forward(self,query,keys,queue):
        target = torch.LongTensor([i for i in range(query.shape[0])]).cuda()

        sim_matrix_pos = self.cos(query.unsqueeze(1),keys.unsqueeze(0))
        sim_matrix_neg = self.cos(query.unsqueeze(1),queue.T.unsqueeze(0))
        
        sim_matrix = torch.cat((sim_matrix_pos,sim_matrix_neg),dim=1).cuda() / self.temp
        
        loss = self.loss_fct(sim_matrix,target)
        return loss



features_grad=0.0
# 为了读取模型中间参数变量的梯度而定义的辅助函数
def extract(g):
    global features_grad
    features_grad = g

def fgsm_attack(embeddings,epsilon,gradient,mean=0.5,std=0.1):
    try:
        # 第一个训练的时候没有梯度
        neg_grad = gradient.sign()
    except:
        return embeddings
    else:
        perturbed_embeddings = embeddings + epsilon*neg_grad#*random_choice
        return perturbed_embeddings


def position_ids_shuffle(position_ids):
    position_ids_shuffle = torch.zeros_like(position_ids)
    for i in range(position_ids.shape[0]):
        index_tensor = torch.randperm(position_ids[i].nelement())
        for index_i,element in enumerate(index_tensor):
            position_ids_shuffle[i][element] = position_ids[i][index_i]
    return position_ids_shuffle


def token_cut_off(inputs_embeds,seq_length,prob=0.1):
    for i in range(inputs_embeds.shape[0]):
        cut_size = max(1,int(seq_length * prob))
        cut_index_list = random.sample([i for i in range(seq_length)],cut_size)
        for cut_index in cut_index_list:
            inputs_embeds[i][cut_index].fill_(0)
    return inputs_embeds


def feature_cut_off(inputs_embeds,prob=0.01):
    for i in range(inputs_embeds.shape[0]):
        cut_size = max(1,int(inputs_embeds.shape[2] * prob))
        cut_index_list = random.sample([i for i in range(inputs_embeds.shape[2])],cut_size)
        for cut_index in cut_index_list:
            inputs_embeds[i][:,cut_index].fill_(0)
    return inputs_embeds



class MoCoSEEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.embedding_drop_prob)
        
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self,
        input_ids=None, 
        token_type_ids=None, 
        position_ids=None, 
        inputs_embeds=None, 
        past_key_values_length=0,
        sent_emb=False
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            
        
        if not sent_emb and self.config.fgsm > 0:
            try:
                inputs_embeds.register_hook(extract)
            except:
                pass
            else:
                inputs_embeds = fgsm_attack(inputs_embeds,self.config.fgsm,features_grad)
        
        
        if not sent_emb:
            # token cut off
            if self.config.token_drop_prob > 0:
                inputs_embeds = token_cut_off(inputs_embeds,seq_length,prob = self.config.token_drop_prob)
            # feature cut off
            if self.config.feature_drop_prob > 0:
                inputs_embeds = feature_cut_off(inputs_embeds,prob = self.config.feature_drop_prob)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            # position shuffle
            if self.config.token_shuffle:
                if not sent_emb:
                    position_embeddings = self.position_embeddings(position_ids_shuffle(position_ids))
                else:
                    position_embeddings = self.position_embeddings(position_ids)
            else:
                position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        
        # drop out
        if not sent_emb:
            embeddings = self.dropout(embeddings)
        return embeddings





# implement of SimCSEModel

@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)

class MoCoSEModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.decay = config.ema_decay
        self.K = config.K
        self.K_start = config.K_start
        
        self.online_embeddings = MoCoSEEmbeddings(config)
        self.online_encoder = BertEncoder(config)
        self.online_pooler = PoolerWithoutActive(config)
        self.online_projection = ProjectionLayer(config)
        
        self.prodiction = MLP(config)
        
        self.loss_fct = InfoNCEWithQueue()
        
        self.init_weights()
        
        # create the queue 
        self.register_buffer("queue", torch.randn(config.out_size, config.K_start))
        self.queue_size = config.K_start
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.prepare()
        
    # 需要改 init之类的问题
    def prepare(self):
        #self.target_embeddings = EMA(self.online_embeddings, decay = self.decay)
        self.target_encoder = EMA(self.online_encoder,decay = self.decay)
        self.target_pooler = EMA(self.online_pooler,decay = self.decay)
        self.target_projection = EMA(self.online_projection,decay = self.decay)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    

    def dequeue_and_enqueue(self,keys):
        with torch.no_grad():
            if self.queue_size == 0:
                self.queue = keys.detach().T
                self.queue_size += keys.shape[0]
            else:
                self.queue = torch.cat((keys.detach().T,self.queue),dim=1)
                self.queue_size += keys.shape[0]
                if self.queue_size > self.K:
                    self.queue = self.queue[:,0:self.K]
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False# if true, return sentence embedding for evaluation
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        if sent_emb:
            pass
        else:
            input_ids = input_ids[:,0]
            attention_mask = attention_mask[:,0]
            token_type_ids = token_type_ids[:,0]
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        # Embedding
        v_online = self.online_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            #sent_emb=sent_emb
        )
        if sent_emb:
            attention_online = self.online_encoder(
                v_online,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            attention_online_last = attention_online[0]
            cls_vec = self.online_pooler(attention_online_last)
            attention_online.pooler_output = cls_vec
            return attention_online
       
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        #self.online_embeddings.update(self.online_embeddings)
        self.target_encoder.update(self.online_encoder)
        self.target_pooler.update(self.online_pooler)
        self.target_projection.update(self.online_projection)
        
        v_target = self.online_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            #sent_emb=sent_emb
        )
        
        # Encoder
        attention_online = self.online_encoder(
            v_online,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        attention_target = self.target_encoder.model(
            v_target,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # pooler
        attention_online_out = attention_online[0]
        attention_target_out = attention_target[0]
        
        # 进行pooler
        proj_online = self.online_pooler(attention_online_out)
        proj_target = self.target_pooler.model(attention_target_out)
        
        # 进行 project
        proj_online = self.online_projection(proj_online)
        proj_target = self.target_projection.model(proj_target)
        
        # prediction
        online_out = self.prodiction(proj_online)
        target_out = proj_target
        
        # 设置pooler 输出
        attention_online.pooler_output = online_out
        attention_target.pooler_output = target_out
        
        loss = self.loss_fct(online_out,target_out,self.queue.clone().detach())
        self.dequeue_and_enqueue(target_out)
        
        return SequenceClassifierOutput(
            loss=loss,
            hidden_states=[online_out, target_out],
            attentions=None,
        )
        

