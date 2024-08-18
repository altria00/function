import torch
from torch import nn
import torch.nn.functional as F
import math


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, maxlen, device):
        super().__init__()
        self.P = torch.zeros(1, maxlen, d_model, device)
        self.P.requires_grad(False)

        position = torch.arange(0, maxlen, dtype=torch.float, device=device).unsqueeze(1)
        _2i = torch.arange(0, d_model, 2)

        self.P[:, :, 0:2] = torch.sin(position / (torch.pow(10000, _2i / d_model)))
        self.P[:, :, 1:2] = torch.cos(position / (torch.pow(10000, _2i / d_model)))

    def forward(self, x):
        return self.P[:, :x.shape(1), :].to(x.device)


class TransformEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, maxlen, device, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionEmbedding(d_model, maxlen, device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        token_embedded = self.token_embedding(x)
        position_embedded = self.position_embedding(x)
        return self.dropout(token_embedded + position_embedded)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, d_model, 1))
        self.beta = nn.Parameter(torch.zeros(1, d_model, 1))
        self.eps = eps

    def foward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.alpha * normalized_x + self.beta


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        _x = x
        x = x + self.drop1(self.self_attn(x, x, x, mask))
        x = self.norm1(x + _x)
        _x = x
        x = x + self.drop2(self.feed_forward(x))
        x = self.norm2(x + _x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3 = LayerNorm(d_model)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, dec, enc, t_mask, s_mask):
        _dec = dec
        q = dec
        k = dec
        v = dec

        q = self.self_attn(q, k, v, t_mask)  # 下三角掩码 使其看不到未来的信息
        q = self.drop1(q)
        q = self.norm1(q + _dec)
        _dec = q

        if enc is not None:
            q = q + self.cross_attn(q, enc, enc, s_mask)  # 位置的掩码，不用关注padding的信息
            q = self.drop2(q)
            q = self.norm2(q + _dec)
            _dec = q

        q = q + self.feed_forward(q)
        q = self.drop3(q)
        q = self.norm3(q + _dec)

        return q


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_K = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, dim = q.shape
        d_k = self.d_model // self.n_heads
        Q = self.linear_Q(q).view(batch_size, seq_len, self.n_heads, d_k).permute(0, 2, 1, 3)
        K = self.linear_K(k).view(batch_size, seq_len, self.n_heads, d_k).permute(0, 2, 1, 3)
        V = self.linear_V(v).view(batch_size, seq_len, self.n_heads, d_k).permute(0, 2, 1, 3)

        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, 1e-9)
        attention_weights = self.softmax(attention_weights) @ V
        attention_weights = attention_weights.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, dim)
        return self.linear_out(attention_weights)


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, maxlen, device, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.embedding = TransformEmbedding(vocab_size, d_model, maxlen, device)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, maxlen, device, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.embedding = TransformEmbedding(vocab_size, d_model, maxlen, device)
        self.linear_out = nn.Linear(d_model, vocab_size)

    def forward(self, dec, enc, t_mask, s_mask):
        x = self.embedding(dec)
        for layer in self.layers:
            x = layer(x, enc, t_mask, s_mask)
        x = self.linear_out(x)
        return x


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_vocab_size, dec_vocab_size, maxlen, d_model, n_layers, n_heads,
                 d_ff, device, dropout=0.1):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.encoder = Encoder(enc_vocab_size, d_model, n_layers, n_heads, d_ff, maxlen, device, dropout)
        self.decoder = Decoder(dec_vocab_size, d_model, n_layers, n_heads, d_ff, maxlen, device, dropout)

    def make_t_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones((len_q, len_k), device=self.device)).bool()
        return mask

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)
        mask_q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        mask_q.repeat(1, 1, 1, len_k)
        mask_k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        mask_k.repeat(1, 1, len_q, 1)

        mask = mask_q & mask_k
        return mask

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx) * self.make_t_mask(trg, trg)
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, trg_mask, src_trg_mask)
        return dec_output
# X = torch.randn(128, 64, 512)
# d_model = 512
# n_heads = 8
# print(X.shape)
#
# attention = MultiHeadAttention(d_model, n_heads)
# output = attention(X, X, X)
# print(output)
