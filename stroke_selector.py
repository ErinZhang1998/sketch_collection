import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.nn.utils.rnn as rnn

class StrokeSelector(nn.Module):
    def __init__(self, num_embeddings, num_classes, embed_dim=512, hidden_embed_dim=512, lstm_layers=1, lstm_drop_prob=0.4, k=30):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=hidden_embed_dim, 
            num_layers=lstm_layers, 
            dropout=lstm_drop_prob,
            batch_first
        )
        # self.tanh = nn.Tanh()

        # self.W_b = nn.Parameter(torch.randn(embed_dim, embed_dim))
        # self.W_v = nn.Parameter(torch.randn(k, embed_dim))
        # self.W_q = nn.Parameter(torch.randn(k, embed_dim))
        # self.w_hv = nn.Parameter(torch.randn(k, 1))
        # self.w_hq = nn.Parameter(torch.randn(k, 1))

        # self.W_w = nn.Linear(embed_dim, embed_dim)
        # self.W_p = nn.Linear(embed_dim*2, embed_dim)
        # self.W_s = nn.Linear(embed_dim*2, embed_dim)

        self.fc = nn.Linear(embed_dim, num_classes)

    def parallel_co_attention(self, V, Q):  # V : B x 512 x 196, Q : B x L x 512
        C = torch.matmul(Q, torch.matmul(self.W_b, V)) # B x L x 196

        H_v = self.tanh(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))                            # B x k x 196
        H_q = self.tanh(torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))           # B x k x L

        a_v = fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2) # B x 1 x 196
        a_q = fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2) # B x 1 x L

        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1))) # B x 512
        q = torch.squeeze(torch.matmul(a_q, Q))                  # B x 512

        return v, q

    def forward(self, image, seq, seq_lengths): # Image: B x 512 x 196
        '''
        # vectorized_seqs => [[6, 9, 8, 4, 1, 11, 12, 10],
        #                     [12, 5, 8, 14],
        #                     [7, 3, 2, 5, 13, 7]]
        seq_lengths = LongTensor(list(map(len, vectorized_seqs)))
        '''
        embedded_seq_tensor = self.embed(seq) # Words: B x L x 512, (batch_size X max_seq_len X embedding_dim)             
        packed_input = rnn.pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)
        packed_input = rnn.pack_padded_sequence(embedded_seq_tensor, seq_lengths.cpu().numpy(), batch_first=True)
                  
        hidden = None
        phrase_packed = nn.utils.rnn.pack_padded_sequence(torch.transpose(phrase, 0, 1), lens)
        sentence_packed, hidden = self.lstm(phrase_packed, hidden)
        sentence, _ = rnn.pad_packed_sequence(sentence_packed)
        sentence = torch.transpose(sentence, 0, 1)                          # B x L x 512

        v_word, q_word = self.parallel_co_attention(image, words)
        v_phrase, q_phrase = self.parallel_co_attention(image, phrase)
        v_sent, q_sent = self.parallel_co_attention(image, sentence)
        h_w = self.tanh(self.W_w(q_word + v_word))
        h_p = self.tanh(self.W_p(torch.cat(((q_phrase + v_phrase), h_w), dim=1)))
        h_s = self.tanh(self.W_s(torch.cat(((q_sent + v_sent), h_p), dim=1)))

        logits = self.fc(h_s)

        return logits