import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):

    def __init__(self, d_model, num_head=8, dropout=0.0, use_linear=True):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.hidden_dim = d_model // num_head
        self.T = (d_model / num_head) ** 0.5
        self.use_linear = use_linear
        if use_linear:
            self.linear_Q = nn.Linear(d_model, d_model)
            self.linear_K = nn.Linear(d_model, d_model)
            self.linear_V = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.drop_prob = dropout
        self.projection = nn.Linear(d_model, d_model)
        self._init_weight()

    def forward(self, Q, K, V, is_return_attn_weight=False):
        """
        :param Q: A 3d tensor with shape of [T_q, bs, C_q]
        :param K: A 3d tensor with shape of [T_k, bs, C_k]
        :param V: A 3d tensor with shape of [T_v, bs, C_v]
        """
        num_head = self.num_head
        hidden_dim = self.hidden_dim
        bs = Q.size()[1]
        if self.use_linear:
            Q = self.linear_Q(Q)
            K = self.linear_K(K)
            V = self.linear_V(V)
        if is_return_attn_weight:
            Q = Q / self.T
            Q = Q.contiguous().view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
            K = K.view(-1, bs, num_head, hidden_dim).permute(1, 2, 3, 0)
            V = V.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
            QK = Q @ K
            attn = torch.softmax(QK, dim=-1)
            attn = self.dropout(attn)
            outputs = (attn @ V).permute(2, 0, 1, 3)
        else:
            Q = Q.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
            K = K.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
            V = V.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
            dropout_p = self.drop_prob if self.training else 0.0
            enable_mem_efficient = False if self.training else True
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=enable_mem_efficient):
                outputs = F.scaled_dot_product_attention(Q, K, V, None, dropout_p, is_causal=False)
            outputs = outputs.permute(2, 0, 1, 3)
            attn = None
        outputs = outputs.reshape(-1, bs, self.d_model)
        outputs = self.projection(outputs)
        return (outputs, attn)

    def forward_with_entry_scores(self, Q, K, V, memory_entry_num, tokens_per_entry, query_chunk_size=None):
        """Run fused attention and return exact per-memory-entry scores.

        A second SDPA call uses entry one-hot vectors as values, making its
        first ``memory_entry_num`` output channels equal to the attention mass
        of each memory entry. Keeping that value tensor at ``hidden_dim`` is
        important: CUDA can then use the same numerically stable fused kernel
        as the feature-attention call. No enormous attention matrix is ever
        materialized. ``query_chunk_size`` is retained for compatibility.
        """
        num_head = self.num_head
        hidden_dim = self.hidden_dim
        bs = Q.size(1)
        if self.use_linear:
            Q = self.linear_Q(Q)
            K = self.linear_K(K)
            V = self.linear_V(V)
        query_num = Q.size(0)
        expected_key_num = memory_entry_num * tokens_per_entry
        if K.size(0) != expected_key_num:
            raise ValueError(f'Attention key count {K.size(0)} does not match {memory_entry_num} entries * {tokens_per_entry} tokens')
        if memory_entry_num > hidden_dim:
            raise ValueError(f'Cannot encode {memory_entry_num} memory entries in a {hidden_dim}-channel attention value')
        Q = Q.contiguous().view(query_num, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
        K = K.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
        V = V.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)
        entry_ids = torch.arange(memory_entry_num, device=V.device).repeat_interleave(tokens_per_entry)
        entry_values = torch.zeros(expected_key_num, hidden_dim, device=V.device, dtype=V.dtype)
        entry_values[:, :memory_entry_num] = F.one_hot(entry_ids, num_classes=memory_entry_num).to(dtype=V.dtype)
        entry_values = entry_values.view(1, 1, expected_key_num, hidden_dim).expand(bs, num_head, -1, -1)
        dropout_p = self.drop_prob if self.training else 0.0
        enable_mem_efficient = False if self.training else True
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=enable_mem_efficient):
            outputs = F.scaled_dot_product_attention(Q, K, V, None, dropout_p, is_causal=False)
            score_outputs = F.scaled_dot_product_attention(Q, K, entry_values, None, dropout_p, is_causal=False)
        entry_scores = score_outputs[..., :memory_entry_num].float().mean(dim=(1, 2))
        outputs = outputs.permute(2, 0, 1, 3)
        outputs = outputs.reshape(-1, bs, self.d_model)
        outputs = self.projection(outputs)
        return (outputs, entry_scores)

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
