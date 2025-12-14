"""
캡션 디코더 모듈

변경점(Attention + alpha 저장):
- Bahdanau(Additive) Attention 추가
- 매 토큰 생성 시 encoder_out(공간 feature)에 attention을 걸어 context를 만들고,
  word embedding과 concat하여 LSTM 입력으로 사용
- step() 함수 추가: 추론 시 동일 로직으로 1-step 디코딩 + alpha 반환
- forward()에서 return_alpha=True이면 [B, T, P] 형태로 alpha도 반환 가능
"""

import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    """
    Additive(Bahdanau) Attention

    encoder_out: [B, P, E]  (P = H*W)
    decoder_h  : [B, H]
    """
    def __init__(self, encoder_dim: int, decoder_dim: int, attn_dim: int):
        super().__init__()
        self.W_enc = nn.Linear(encoder_dim, attn_dim)
        self.W_dec = nn.Linear(decoder_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1)

    def forward(self, encoder_out, decoder_h):
        enc_proj = self.W_enc(encoder_out)               # [B, P, A]
        dec_proj = self.W_dec(decoder_h).unsqueeze(1)    # [B, 1, A]
        scores = self.v(torch.tanh(enc_proj + dec_proj)).squeeze(-1)  # [B, P]
        alpha = torch.softmax(scores, dim=1)             # [B, P]
        context = (encoder_out * alpha.unsqueeze(-1)).sum(dim=1)       # [B, E]
        return context, alpha


class CaptionDecoder(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.5, attn_dim=None):
        super(CaptionDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.attn_dim = attn_dim if attn_dim is not None else hidden_size
        self.attention = BahdanauAttention(encoder_dim=embed_size, decoder_dim=hidden_size, attn_dim=self.attn_dim)

        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 첫 레이어 입력: [word_embed(E) ; context(E)] => 2E
        self.lstm_cells = nn.ModuleList()
        self.lstm_cells.append(LSTMCell(embed_size * 2, hidden_size))
        for _ in range(1, num_layers):
            self.lstm_cells.append(LSTMCell(hidden_size, hidden_size))

        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers - 1)])

        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

    def init_hidden_state(self, features):
        h0 = self.init_h(features)
        c0 = self.init_c(features)
        h_states = [h0.clone() for _ in range(self.num_layers)]
        c_states = [c0.clone() for _ in range(self.num_layers)]
        return h_states, c_states

    def _build_context(self, encoder_out, h_last, word_embed):
        if encoder_out is None:
            context = word_embed.new_zeros(word_embed.size(0), self.embed_size)
            alpha = None
            return context, alpha
        context, alpha = self.attention(encoder_out, h_last)
        return context, alpha

    def step(self, word_embed, h_states, c_states, encoder_out=None, return_alpha=False):
        context, alpha = self._build_context(encoder_out, h_states[-1], word_embed)
        x_t = torch.cat([word_embed, context], dim=-1)  # [B, 2E]

        for layer_idx in range(self.num_layers):
            h_states[layer_idx], c_states[layer_idx] = self.lstm_cells[layer_idx](x_t, h_states[layer_idx], c_states[layer_idx])
            if layer_idx < self.num_layers - 1:
                h_states[layer_idx] = self.dropout_layers[layer_idx](h_states[layer_idx])
            x_t = h_states[layer_idx]

        logits = self.linear(h_states[-1])
        if return_alpha:
            return logits, h_states, c_states, alpha
        return logits, h_states, c_states, None

    def forward(self, features, captions, lengths=None, encoder_out=None, return_alpha=False):
        h_states, c_states = self.init_hidden_state(features)

        embeddings = self.embedding(captions[:, :-1])  # [B, T, E]
        embeddings = self.dropout(embeddings)

        T = embeddings.size(1)
        outputs = []
        alphas = []

        for t in range(T):
            word_embed = embeddings[:, t, :]
            logits, h_states, c_states, alpha = self.step(word_embed, h_states, c_states, encoder_out=encoder_out, return_alpha=return_alpha)
            outputs.append(logits)
            if return_alpha:
                alphas.append(alpha)

        outputs = torch.stack(outputs, dim=1)  # [B, T, V]
        if return_alpha:
            alphas = torch.stack(alphas, dim=1)  # [B, T, P]
            return outputs, alphas
        return outputs


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_C = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat((x, h_prev), dim=1)
        f_t = torch.sigmoid(self.W_f(combined))
        i_t = torch.sigmoid(self.W_i(combined))
        o_t = torch.sigmoid(self.W_o(combined))
        c_tilde = torch.tanh(self.W_C(combined))
        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t
