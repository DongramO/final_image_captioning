"""
캡션 디코더 모듈
"""

import torch
import torch.nn as nn


class CaptionDecoder(nn.Module):
    """
    RNN/LSTM 기반 캡션 디코더
    """
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.5):
        """
        Args:
            embed_size: 임베딩 차원 (단어 임베딩 + 이미지 특징 차원)
            hidden_size: LSTM 히든 상태 차원
            vocab_size: 단어장 크기
            num_layers: LSTM 레이어 수
            dropout: 드롭아웃 비율
        """
        super(CaptionDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # TODO: 단어 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # TODO: LSTM 레이어
        self.lstm_cells = nn.ModuleList()
        self.lstm_cells.append(LSTMCell(embed_size, hidden_size))
        
        for _ in range(1, num_layers):
            self.lstm_cells.append(LSTMCell(hidden_size, hidden_size))  # 나머지 레이어들
        
        # Dropout 레이어 (레이어 간에 적용)
        self.dropout_layers = nn.ModuleList()
        for _ in range(num_layers - 1):  # 마지막 레이어 전까지만 dropout
            self.dropout_layers.append(nn.Dropout(dropout))
        
        # 출력 레이어
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # 이미지 특징을 LSTM 초기 상태로 변환하는 레이어
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

    def init_hidden_state(self, features):
        """
        이미지 특징을 LSTM의 초기 hidden state와 cell state로 변환
        
        Args:
            features: 인코딩된 이미지 특징 [batch_size, embed_size]
            
        Returns:
            h0: 초기 hidden state 리스트 [num_layers, batch_size, hidden_size]
            c0: 초기 cell state 리스트 [num_layers, batch_size, hidden_size]
        """
        h0 = self.init_h(features)  # [batch_size, hidden_size]
        c0 = self.init_c(features)  # [batch_size, hidden_size]
        
        # 각 레이어마다 초기 상태 생성 (모든 레이어가 같은 초기 상태 사용)
        h_states = [h0.clone() for _ in range(self.num_layers)]
        c_states = [c0.clone() for _ in range(self.num_layers)]
        
        return h_states, c_states


    def forward(self, features, captions, lengths=None):
        """
        Args:
            features: 인코딩된 이미지 특징 [batch_size, embed_size]
            captions: 캡션 텐서 [batch_size, seq_length] (<start> 토큰 포함, <end> 토큰 제외)
            lengths: 각 캡션의 실제 길이 (packed sequence용, 선택적)
            
        Returns:
            outputs: 캡션 로짓 [batch_size, seq_length, vocab_size]
        """
      
        
        # 1. 이미지 특징을 LSTM의 초기 상태로 변환
        h_states, c_states = self.init_hidden_state(features)
        
        # 2. 캡션 임베딩 (마지막 토큰 제외 - teacher forcing)
        embeddings = self.embedding(captions[:, :-1])  # [batch_size, seq_length-1, embed_size]
        embeddings = self.dropout(embeddings)

        batch_size = features.size(0)
        seq_length = embeddings.size(1)
        
        # 3. 시퀀스 처리
        outputs = []
        for t in range(seq_length):
            x_t = embeddings[:, t, :]  # [batch_size, embed_size]
            
            # 각 레이어를 순차적으로 통과
            for layer_idx in range(self.num_layers):
                h_states[layer_idx], c_states[layer_idx] = self.lstm_cells[layer_idx].forward(
                    x_t, h_states[layer_idx], c_states[layer_idx]
                )
                
                # 마지막 레이어가 아니면 dropout 적용
                if layer_idx < self.num_layers - 1:
                    h_states[layer_idx] = self.dropout_layers[layer_idx](h_states[layer_idx])
                
                # 다음 레이어의 입력으로 사용
                x_t = h_states[layer_idx]
            # 출력 레이어를 통한 로짓 생성
            output = self.linear(h_states[-1])  # [batch_size, vocab_size]
            outputs.append(output)
        
        # 결과 합치기
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_length, vocab_size]
        
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

        # Cell state 업데이트
        # f_t = sigmoid(Wf * [h_t-1, x_t] + bf)
        # i_t = sigmoid(Wi * [h_t-1, x_t] + bi)
        # C_tilde = tanh(WC * [h_t-1, x_t] + bC)
        # o_t = sigmoid(Wo * [h_t-1, x_t] + bo)
        # c_t = f_t * c_prev + i_t * C_tilde
        # h_t = o_t * tanh(c_t)
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Hidden state 업데이트
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t