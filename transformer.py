import torch
from torch import nn
from torch import Tensor
import numpy as np


class Embedding(nn.Module):

    def __init__(self, vocab_size, hidden_size, max_len, drop_prob=0.1):
        """
        vocab_size: размер словаря
        hidden_size: размер скрытого слоя
        max_len: максимальная возможная длина текста
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_len, hidden_size)

        self.register_buffer(
            "position_ids", torch.arange(max_len).expand((1, -1)), persistent=False
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, input_ids) -> Tensor:
        """
        input_ids: Tensor[bs, seq_len] – индексы токенов текста
        """
        assert input_ids.ndim == 2
        tok_emb = self.word_embeddings(input_ids)

        position_ids = self.position_ids[:, :input_ids.shape[1]]
        pos_emb = self.position_embeddings(position_ids)

        emb = tok_emb + pos_emb
        return self.dropout(self.layer_norm(emb))


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, n_head, drop_prob=0.1):
        """
        hidden_size: размер скрытого слоя
        n_head: число голов внимания
        drop_prob: вероятность удаления нейрона в dropout
        """

        super().__init__()
        self.n_head = n_head

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, q, k, v, attention_mask=None) -> Tensor:
        """
        q, k, v: Tensor[bs, seq_len, hidden_size] – входные аргументы для соответствующих линейных слоев
        attention_mask: Tensor[bs, 1, 1 or seq_len, seq_len] – маска внимания, содержащая значения 0 и -inf;
                                                               добавляется к скорам механизма внимания (до softmax)
        """
        assert q.ndim == k.ndim == v.ndim == 3
        assert attention_mask.ndim == 4

        q, k, v = self.query(q), self.key(k), self.value(v)

        batch_size, seq_len, hidden_size = q.shape

        head_dim = hidden_size // self.n_head
        q = q.view(batch_size, -1, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_head, head_dim).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(head_dim)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, v)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, seq_len, -1)

        out = self.out(context_layer)

        return out


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, drop_prob=0.1):
        """
        hidden_size: размер скрытого слоя
        intermediate_size: размер промежуточного слоя
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()

        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states) -> Tensor:
        """
        hidden_states: Tensor[bs, seq_len, hidden_size] – входное представление текста
        """
        assert hidden_states.ndim == 3

        input_hidden_states = hidden_states
        hidden_states = self.relu(self.linear1(hidden_states))
        hidden_states = self.dropout(self.linear2(hidden_states))
        hidden_states = self.layer_norm(hidden_states + input_hidden_states)
        return hidden_states


class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, n_head, drop_prob=0.1):
        """
        hidden_size: размер скрытого слоя
        intermediate_size: размер промежуточного слоя
        n_head: число голов внимания
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()

        self.attention = MultiHeadAttention(hidden_size, n_head, drop_prob=drop_prob)
        self.dropout = nn.Dropout(p=drop_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.feed_forward = FeedForward(hidden_size, intermediate_size, drop_prob=drop_prob)

    def forward(self, hidden_states, attention_mask) -> Tensor:
        """
        hidden_states: Tensor[bs, seq_len, hidden_size] – входное представление текста
        attention_mask: Tensor[bs, 1, 1, seq_len] – маска внимания, содержащая значения 0 и -inf
        """
        assert hidden_states.ndim == 3
        assert attention_mask.ndim == 4

        input_hidden_states = hidden_states
        hidden_states = self.attention(
            q=hidden_states,
            k=hidden_states,
            v=hidden_states,
            attention_mask=attention_mask
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_hidden_states)

        hidden_states = self.feed_forward(hidden_states)

        return hidden_states


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size,
                 intermediate_size, n_head, n_layers, drop_prob=0.1):
        """
        vocab_size: размер словаря
        max_len: максимальная возможная длина текста
        hidden_size: размер скрытого слоя
        intermediate_size: размер промежуточного слоя
        n_head: число голов внимания
        n_layers: число блоков Encoder
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()

        self.embedding = Embedding(vocab_size, hidden_size, max_len, drop_prob=0.1)

        self.layers = nn.ModuleList([
            EncoderBlock(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                n_head=n_head,
                drop_prob=drop_prob
            ) for _ in range(n_layers)
        ])

    def forward(self, input_ids, attention_mask) -> Tensor:
        """
        input_ids: Tensor[bs, seq_len] – индексы токенов текста
        attention_mask: Tensor[bs, 1, 1, seq_len] – маска внимания, содержащая значения 0 и -inf
        """
        assert input_ids.ndim == 2
        assert attention_mask.ndim == 4

        hidden_states = self.embedding(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return hidden_states


class DecoderBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, n_head, drop_prob=0.1):
        """
        hidden_size: размер скрытого слоя
        intermediate_size: размер промежуточного слоя
        n_head: число голов внимания
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, n_head, drop_prob=drop_prob)
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        self.cross_attention = MultiHeadAttention(hidden_size, n_head, drop_prob=drop_prob)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(p=drop_prob)

        self.feed_forward = FeedForward(hidden_size, intermediate_size, drop_prob=drop_prob)

    def forward(self, hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask) -> Tensor:
        """
        hidden_states: Tensor[bs, trg_seq_len, hidden_size] – входное представление целевого текста
        attention_mask: Tensor[bs, 1, trg_seq_len, trg_seq_len] – маска внимания Decoder'a
        encoder_hidden_states: Tensor[bs, src_seq_len, hidden_size] – выход последнего слоя Encoder
        encoder_attention_mask: Tensor[bs, 1, 1, src_seq_len] – маска внимания Encoder'a
        """
        assert hidden_states.ndim == encoder_hidden_states.ndim == 3
        assert attention_mask.ndim == encoder_attention_mask.ndim == 4

        input_hidden_states = hidden_states
        hidden_states = self.attention(
            q=hidden_states,
            k=hidden_states,
            v=hidden_states,
            attention_mask=attention_mask
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm1(hidden_states + input_hidden_states)

        if encoder_hidden_states is not None:
            input_hidden_states = hidden_states
            hidden_states = self.cross_attention(
                q=hidden_states,
                k=encoder_hidden_states,
                v=encoder_hidden_states,
                attention_mask=encoder_attention_mask
            )
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.layer_norm2(hidden_states + input_hidden_states)

        hidden_states = self.feed_forward(hidden_states)

        return hidden_states


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size,
                 intermediate_size, n_head, n_layers, drop_prob=0.1):
        """
        vocab_size: размер словаря
        max_len: максимальная возможная длина текста
        hidden_size: размер скрытого слоя
        intermediate_size: размер промежуточного слоя
        n_head: число голов внимания
        n_layers: число блоков Decoder
        drop_prob: вероятность удаления нейрона в dropout
        """
        super().__init__()

        self.embedding = Embedding(vocab_size, hidden_size, max_len, drop_prob=0.1)

        self.layers = nn.ModuleList([
            DecoderBlock(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                n_head=n_head,
                drop_prob=drop_prob
            ) for _ in range(n_layers)
        ])

        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask) -> Tensor:
        """
        input_ids: Tensor[bs, seq_len] – индексы токенов текста
        attention_mask: Tensor[bs, 1, trg_seq_len, trg_seq_len] – маска внимания Decoder'a
        encoder_hidden_states: Tensor[bs, src_seq_len, hidden_size] – выход последнего слоя Encoder
        encoder_attention_mask: Tensor[bs, 1, 1, src_seq_len] - маска внимания Encoder'a
        """
        assert input_ids.ndim == 2
        assert encoder_hidden_states.ndim == 3
        assert attention_mask.ndim == encoder_attention_mask.ndim == 4

        hidden_states = self.embedding(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask)

        return self.out(hidden_states)

def get_extended_attention_mask(attention_mask, dtype=torch.float):
    extended_attention_mask = attention_mask[:, None, None, :].to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


def get_causal_extended_attention_mask(attention_mask, dtype=torch.float):
    device = attention_mask.device
    batch_size, seq_len = attention_mask.shape

    seq_ids = torch.arange(seq_len, device=device)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_len, 1) <= seq_ids[None, :, None]
    causal_mask = causal_mask.to(attention_mask.dtype)

    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :].to(dtype=dtype)

    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask

class Transformer(nn.Module):
    def __init__(self, encoder_vocab_size, decoder_vocab_size, hidden_size, n_head,
                 intermediate_size, encoder_max_len, decoder_max_len, n_layers, drop_prob=0.1):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=encoder_vocab_size,
            max_len=encoder_max_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            n_head=n_head,
            n_layers=n_layers,
            drop_prob=drop_prob
        )

        self.decoder = Decoder(
            vocab_size=decoder_vocab_size,
            max_len=decoder_max_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            n_head=n_head,
            n_layers=n_layers,
            drop_prob=drop_prob
        )

    def forward(self, src_input_ids, trg_input_ids, src_attention_mask=None, trg_attention_mask=None) -> Tensor:
        """
        src_input_ids: Tensor[bs, src_seq_len] –индексы токенов входного текста
        trg_input_ids: Tensor[bs, trg_seq_len] –индексы токенов выходного текста
        src_attention_mask: Tensor[bs, scr_seq_len] – маска внимания входного текста
        trg_attention_mask: Tensor[bs, trg_seq_len] – маска внимания выходного текста
        """
        if src_attention_mask is None:
            src_attention_mask = torch.ones(src_input_ids.shape, device=src_input_ids.device)
        src_attention_mask = get_extended_attention_mask(src_attention_mask)

        if trg_attention_mask is None:
            trg_attention_mask = torch.ones(trg_input_ids.shape, device=trg_input_ids.device)
        trg_attention_mask = get_causal_extended_attention_mask(trg_attention_mask)

        encoder_hidden_states = self.encoder(src_input_ids, src_attention_mask)
        output = self.decoder(trg_input_ids, trg_attention_mask, encoder_hidden_states, src_attention_mask)

        return output