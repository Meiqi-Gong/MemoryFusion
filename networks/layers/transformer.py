from typing import Iterable, List
import torch.nn.functional as F
from torch import nn
import torch
from networks.layers.basic import DropPath, GroupNorm1D, GNActDWConv2d
from networks.layers.attention import MultiheadAttention
from utils.tensor import bchw_2_lbc
import numpy as np
from networks.debug import debug
VALID_LTMB_UPDATE_STRATEGIES = ('ours', 'random', 'fifo', 'attention', 'rmem')

def compute_ltmb_attention_entry_score(attn_prob, memory_entry_num, tokens_per_entry):
    if attn_prob is None:
        return None
    b, h, nq, key_tokens = attn_prob.size()
    expected_key_tokens = memory_entry_num * tokens_per_entry
    if key_tokens != expected_key_tokens:
        raise ValueError(f'LTMB attention shape mismatch: got {key_tokens} key tokens, expected {memory_entry_num} * {tokens_per_entry} = {expected_key_tokens}')
    attn = attn_prob.reshape(b, h, nq, memory_entry_num, tokens_per_entry)
    return attn.sum(dim=-1).mean(dim=(1, 2))

def aggregate_ltmb_attention_scores(attention_scores):
    valid_scores = [score for score in attention_scores if score is not None]
    if len(valid_scores) == 0:
        return None
    return torch.stack(valid_scores, dim=0).mean(dim=0)

def build_keep_indices_after_drop(drop_idx_batch, entry_num, device):
    all_idx = torch.arange(entry_num, device=device).unsqueeze(0).expand(drop_idx_batch.size(0), -1)
    mask = all_idx != drop_idx_batch.unsqueeze(1)
    return all_idx[mask].view(drop_idx_batch.size(0), entry_num - 1)

def update_ltmb_frame_indexes_after_drop(long_memories_indexes, drop_idx_batch):
    updated_lmi = []
    for b in range(drop_idx_batch.size(0)):
        drop_pos = drop_idx_batch[b].item()
        updated_lmi.append([frame_idx for i, frame_idx in enumerate(long_memories_indexes[b]) if i != drop_pos])
    long_memories_indexes[:] = updated_lmi
    return long_memories_indexes

def normalize_rmem_relevance(attention_score):
    return attention_score / (attention_score.sum(dim=1, keepdim=True) + 1e-06)

def update_rmem_stats_with_attention(relevance_ema, access_count, attention_score, beta):
    attention_score = normalize_rmem_relevance(attention_score.detach().float())
    if relevance_ema is None or access_count is None:
        relevance_ema = attention_score
        access_count = torch.ones_like(attention_score)
    else:
        if relevance_ema.shape != attention_score.shape:
            raise ValueError(f'RMem stats shape {relevance_ema.shape} does not match attention score shape {attention_score.shape}.')
        relevance_ema = beta * relevance_ema.to(attention_score.device) + (1 - beta) * attention_score
        access_count = access_count.to(attention_score.device) + 1
    return (relevance_ema.detach(), access_count.detach())

def compute_rmem_retention_score(relevance_ema, access_count, freshness_weight, freshness_add):
    total_access = access_count.sum(dim=1, keepdim=True).clamp_min(1.0)
    freshness_score = freshness_weight * torch.sqrt(torch.log(total_access) / (access_count + freshness_add))
    return relevance_ema + freshness_score

def append_rmem_new_entry(relevance_ema, access_count):
    new_relevance = torch.zeros(relevance_ema.size(0), 1, dtype=relevance_ema.dtype, device=relevance_ema.device)
    new_access = torch.ones(access_count.size(0), 1, dtype=access_count.dtype, device=access_count.device)
    return (torch.cat([relevance_ema, new_relevance], dim=1).detach(), torch.cat([access_count, new_access], dim=1).detach())

def prune_rmem_stats_and_append(relevance_ema, access_count, drop_idx_batch):
    entry_num = relevance_ema.size(1)
    keep_idx = build_keep_indices_after_drop(drop_idx_batch, entry_num, relevance_ema.device)
    relevance_ema = torch.gather(relevance_ema, 1, keep_idx)
    access_count = torch.gather(access_count, 1, keep_idx)
    return append_rmem_new_entry(relevance_ema, access_count)

def _get_norm(indim, type='ln', groups=8):
    if type == 'gn' and groups != 1:
        return GroupNorm1D(indim, groups)
    elif type == 'gn' and groups == 1:
        return nn.GroupNorm(groups, indim)
    else:
        return nn.LayerNorm(indim)

class ConvGRUCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize the ConvLSTM cell
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.padding = 'same'
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim, out_channels=2 * self.hidden_dim, kernel_size=kernel_size, padding=self.padding, bias=self.bias)
        self.conv_can = nn.Conv2d(in_channels=input_dim + hidden_dim, out_channels=self.hidden_dim, kernel_size=kernel_size, padding=self.padding, bias=self.bias)

    def init_hidden(batch_size, hidden_dim, input_size, dtype):
        height, width = input_size
        return torch.autograd.Variable(torch.zeros(batch_size, hidden_dim, height, width)).type(dtype)

    def forward(self, input_tensor, h_cur):
        """

        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)
        combined = torch.cat([input_tensor, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)
        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next

class ConvGRUCellOutput(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, output_dim) -> None:
        super(ConvGRUCellOutput, self).__init__()
        self.conv_gru_cell = ConvGRUCell(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, bias=bias)
        self.output_conv = nn.Conv2d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=1)

    def forward(self, input_tensor, h_cur):
        h_next = self.conv_gru_cell(input_tensor, h_cur)
        return (h_next, self.output_conv(h_next))
atten_condenser = MultiheadAttention(d_model=256, num_head=8, use_linear=False)
atten_condenser.projection = nn.Identity()

class LongShortTermTransformer(nn.Module):

    def __init__(self, num_layers=2, d_model=256, self_nhead=8, att_nhead=8, dim_feedforward=1024, emb_dropout=0.0, droppath=0.1, lt_dropout=0.0, st_dropout=0.0, droppath_lst=False, droppath_scaling=False, activation='gelu', return_intermediate=False, intermediate_norm=True, final_norm=True, linear_q=False, norm_inp=False, time_encode=False, gru_memory=False):
        super().__init__()
        self.intermediate_norm = intermediate_norm
        self.final_norm = final_norm
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.emb_dropout = nn.Dropout(emb_dropout, True)
        self.gru_memory = gru_memory
        self._long_memories_indexes_initialized = True
        layers: List[SimplifiedTransformerBlock] = []
        for idx in range(num_layers):
            if droppath_scaling:
                if num_layers == 1:
                    droppath_rate = 0
                else:
                    droppath_rate = droppath * idx / (num_layers - 1)
            else:
                droppath_rate = droppath
            layers.append(SimplifiedTransformerBlock(d_model, self_nhead, att_nhead, dim_feedforward, droppath_rate, activation, linear_q=linear_q, time_encode=time_encode, gru_memory=gru_memory))
        self.layers: Iterable[SimplifiedTransformerBlock] = nn.ModuleList(layers)
        num_norms = num_layers - 1 if intermediate_norm else 0
        if final_norm:
            num_norms += 1
        self.decoder_norms = [_get_norm(d_model, type='ln') for _ in range(num_norms)] if num_norms > 0 else None
        if self.decoder_norms is not None:
            self.decoder_norms = nn.ModuleList(self.decoder_norms)
        self.clear_memory()

    def forward(self, tgt, curr_id_emb=None, self_pos=None, size_2d=None, temporal_encoding=None, is_outer_memory=False, outer_long_memories=None, outer_short_memories=None, save_atten_weights=False, return_ltmb_attn_scores=False):
        output = self.emb_dropout(tgt)
        intermediate = []
        intermediate_memories = []
        ltmb_attention_scores = []
        for idx, layer in enumerate(self.layers):
            if is_outer_memory:
                output, memories = layer(output, outer_long_memories[idx], outer_short_memories[idx], curr_id_emb=curr_id_emb, self_pos=self_pos, size_2d=size_2d, temporal_encoding=temporal_encoding, save_atten_weights=save_atten_weights, return_ltmb_attn_scores=return_ltmb_attn_scores)
            else:
                output, memories = layer(output, self.long_term_memories[idx] if self.long_term_memories is not None else None, self.short_term_memories[idx] if self.short_term_memories is not None else None, curr_id_emb=curr_id_emb, self_pos=self_pos, size_2d=size_2d, temporal_encoding=temporal_encoding, save_atten_weights=save_atten_weights, return_ltmb_attn_scores=return_ltmb_attn_scores)
            if return_ltmb_attn_scores and (not is_outer_memory):
                ltmb_attention_scores.append(layer.long_term_attention_entry_score)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_memories.append(memories)
        if self.decoder_norms is not None:
            if self.final_norm:
                output = self.decoder_norms[-1](output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                if self.intermediate_norm:
                    for idx in range(len(intermediate) - 1):
                        intermediate[idx] = self.decoder_norms[idx](intermediate[idx])
        if self.return_intermediate:
            if not is_outer_memory:
                self.lstt_curr_memories, self.lstt_long_memories, self.lstt_short_memories = zip(*intermediate_memories)
                if return_ltmb_attn_scores:
                    self.latest_long_term_attention_score = aggregate_ltmb_attention_scores(ltmb_attention_scores)
            return intermediate
        if return_ltmb_attn_scores and (not is_outer_memory):
            self.latest_long_term_attention_score = aggregate_ltmb_attention_scores(ltmb_attention_scores)
        return output

    def update_short_memories(self, curr_id_emb, short_term_mem_skip, size_2d, is_update_long_memory):
        curr_id_emb = bchw_2_lbc(curr_id_emb)
        lstt_curr_memories_2d = []
        for layer_idx in range(len(self.lstt_curr_memories)):
            curr_v = self.lstt_curr_memories[layer_idx][1]
            curr_v = self.layers[layer_idx].linear_V(curr_v + curr_id_emb)
            self.lstt_curr_memories[layer_idx][1] = curr_v
            curr_v = self.lstt_short_memories[layer_idx][1]
            curr_v = self.layers[layer_idx].linear_VMem(curr_v + curr_id_emb)
            self.lstt_short_memories[layer_idx][1] = curr_v
            lstt_curr_memories_2d.append([self.lstt_short_memories[layer_idx][0], self.lstt_short_memories[layer_idx][1]])
        self.short_term_memories_list.append(lstt_curr_memories_2d)
        for temp in self.short_term_memories_list[0]:
            for x in temp:
                x.cpu()
        self.short_term_memories_list = self.short_term_memories_list[-short_term_mem_skip:]
        self.short_term_memories = self.short_term_memories_list[0]
        if is_update_long_memory:
            self.update_long_term_memory(self.lstt_curr_memories)

    def update_long_term_memory(self, new_long_term_memories):
        updated_long_term_memories = []
        max_size = 48840
        for new_long_term_memory, last_long_term_memory in zip(new_long_term_memories, self.long_term_memories):
            updated_e = []
            for new_e, last_e in zip(new_long_term_memory, last_long_term_memory):
                new_mem = torch.cat([last_e, new_e[None, ...]], dim=0)
                updated_e.append(new_mem)
            updated_long_term_memories.append(updated_e)
        self.long_term_memories = updated_long_term_memories

    def restrict_long_memories(self, former_memory_len, latter_memory_len, loss, frame_step, use_atten_weight=False, long_memories_indexes=[], foreground_proba: torch.Tensor=None, ltmb_update_strategy='ours', debug_ltmb_update=False, recon_score_weight=1.0, rmem_relevance_ema_beta=0.2, rmem_freshness_weight=1.5, rmem_freshness_add=8.0):
        key_tensor = self.long_term_memories[0][0]
        N, D, B, W = key_tensor.shape
        if ltmb_update_strategy not in VALID_LTMB_UPDATE_STRATEGIES:
            raise ValueError(f'Unknown LTMB update strategy: {ltmb_update_strategy}. Expected one of {VALID_LTMB_UPDATE_STRATEGIES}.')
        if ltmb_update_strategy == 'rmem':
            old_entry_num = N - 1
            if old_entry_num > 0:
                attention_score = self.latest_long_term_attention_score
                if attention_score is None:
                    raise RuntimeError('RMem LTMB pruning requires attention scores from the current frame retrieval, but none were recorded.')
                attention_score = attention_score.detach()
                if attention_score.size(1) != old_entry_num:
                    raise ValueError(f'RMem attention score entry count {attention_score.size(1)} does not match LTMB size before current write {old_entry_num}.')
                self.rmem_relevance_ema, self.rmem_access_count = update_rmem_stats_with_attention(self.rmem_relevance_ema, self.rmem_access_count, attention_score, rmem_relevance_ema_beta)
            if N <= former_memory_len + latter_memory_len:
                if old_entry_num > 0:
                    self.rmem_relevance_ema, self.rmem_access_count = append_rmem_new_entry(self.rmem_relevance_ema, self.rmem_access_count)
                    if long_memories_indexes and isinstance(long_memories_indexes[0], list):
                        self.rmem_frame_indices = [lst.copy() for lst in long_memories_indexes]
                    else:
                        self.rmem_frame_indices = [long_memories_indexes.copy() for _ in range(B)]
                return (long_memories_indexes, True)
        if self.long_term_memories[0][0].size(0) <= former_memory_len + latter_memory_len:
            return (long_memories_indexes, True)
        if N <= former_memory_len + latter_memory_len:
            return
        if self._long_memories_indexes_initialized:
            long_memories_indexes = [long_memories_indexes.copy() for _ in range(B)]
            self._long_memories_indexes_initialized = False
        if ltmb_update_strategy != 'ours':
            if ltmb_update_strategy == 'attention':
                attention_score = self.latest_long_term_attention_score
                if attention_score is None:
                    raise RuntimeError('LTMB attention pruning requires attention scores from the current frame retrieval, but none were recorded.')
                attention_score = attention_score.detach()
                old_entry_num = attention_score.size(1)
                if old_entry_num != N - 1:
                    raise ValueError(f'Attention score entry count {old_entry_num} does not match LTMB size before current write {N - 1}.')
                drop_idx_batch = torch.argmin(attention_score, dim=1)
            elif ltmb_update_strategy == 'fifo':
                old_entry_num = N - 1
                attention_score = None
                drop_idx_batch = torch.zeros(B, dtype=torch.long, device=key_tensor.device)
            elif ltmb_update_strategy == 'rmem':
                old_entry_num = N - 1
                attention_score = self.latest_long_term_attention_score.detach()
                retention_score = compute_rmem_retention_score(self.rmem_relevance_ema, self.rmem_access_count, rmem_freshness_weight, rmem_freshness_add)
                drop_idx_batch = torch.argmin(retention_score, dim=1)
                self.rmem_relevance_ema, self.rmem_access_count = prune_rmem_stats_and_append(self.rmem_relevance_ema, self.rmem_access_count, drop_idx_batch)
            else:
                old_entry_num = N - 1
                attention_score = None
                drop_idx_batch = torch.randint(0, old_entry_num, (B,), device=key_tensor.device)
            keep_old_idx = build_keep_indices_after_drop(drop_idx_batch, old_entry_num, key_tensor.device)
            new_entry_idx = torch.full((B, 1), old_entry_num, dtype=torch.long, device=key_tensor.device)
            keep_idx = torch.cat([keep_old_idx, new_entry_idx], dim=1)
            for i in range(len(self.long_term_memories)):
                k, v = self.long_term_memories[i]
                _, k_channels, _, k_width = k.shape
                _, v_channels, _, v_width = v.shape
                k_perm = k.permute(2, 0, 1, 3)
                v_perm = v.permute(2, 0, 1, 3)
                k_idx = keep_idx.unsqueeze(-1).unsqueeze(-1).expand(B, old_entry_num, k_channels, k_width)
                v_idx = keep_idx.unsqueeze(-1).unsqueeze(-1).expand(B, old_entry_num, v_channels, v_width)
                k_new = torch.gather(k_perm, 1, k_idx)
                v_new = torch.gather(v_perm, 1, v_idx)
                self.long_term_memories[i] = (k_new.permute(1, 2, 0, 3), v_new.permute(1, 2, 0, 3))
            if debug_ltmb_update:
                for b in range(B):
                    scores = None if attention_score is None else attention_score[b].detach().cpu().tolist()
                    retention_scores = None
                    if ltmb_update_strategy == 'rmem':
                        retention_scores = retention_score[b].detach().cpu().tolist()
                    before_indexes = long_memories_indexes[b]
                    drop_pos = drop_idx_batch[b].item()
                    debug(f'LTMB {ltmb_update_strategy}: frame={frame_step}, batch={b}, before={before_indexes}, scores={scores}, retention_scores={retention_scores}, drop_idx={drop_pos}, drop_frame={before_indexes[drop_pos]}')
            update_ltmb_frame_indexes_after_drop(long_memories_indexes, drop_idx_batch)
            if ltmb_update_strategy == 'rmem':
                self.rmem_frame_indices = [lst.copy() for lst in long_memories_indexes]
            if debug_ltmb_update:
                debug(f'LTMB {ltmb_update_strategy}: after={long_memories_indexes}')
            return (long_memories_indexes, False)
        M = key_tensor.permute(2, 0, 1, 3).reshape(B, N, -1)
        M = F.normalize(M, dim=2)
        idx_mask = torch.eye(N, device=M.device).bool()
        idx_mask = idx_mask.unsqueeze(0).expand(B, -1, -1)
        mask = ~idx_mask.unsqueeze(-1).expand(B, N, N, M.size(2))
        M_expand = M.unsqueeze(1).expand(B, N, N, M.size(2))
        M_minus_i_all = M_expand[mask].view(B, N, N - 1, M.size(2))
        m_i_all = M
        A = M_minus_i_all.permute(0, 1, 3, 2)
        b = m_i_all.unsqueeze(-1)
        A = A.reshape(B * N, M.size(2), N - 1)
        b = b.reshape(B * N, M.size(2), 1)
        try:
            res = torch.linalg.lstsq(A, b)
            w = res.solution.squeeze(-1)
        except RuntimeError:
            w = torch.ones(B * N, N - 1, device=M.device) / (N - 1)
        M_minus_i_flat = M_minus_i_all.reshape(B * N, N - 1, M.size(2))
        m_hat = torch.bmm(w.unsqueeze(1), M_minus_i_flat).squeeze(1)
        m_i_flat = m_i_all.reshape(B * N, M.size(2))
        recon_errors = F.mse_loss(m_hat, m_i_flat, reduction='none').sum(dim=1)
        recon_errors = recon_errors.view(B, N)
        recon_scores = 1 - recon_errors / (recon_errors.sum(dim=1, keepdim=True) + 1e-06)
        recon_scores = (recon_scores - recon_scores.min(dim=1, keepdim=True)[0]) / (recon_scores.max(dim=1, keepdim=True)[0] - recon_scores.min(dim=1, keepdim=True)[0] + 1e-06)
        loss_tensor = torch.stack(loss, dim=0)
        idx_tensor = torch.stack([torch.tensor(lst, dtype=torch.long) for lst in long_memories_indexes], dim=0)
        selected_loss = loss_tensor[idx_tensor, torch.arange(B).unsqueeze(1)]
        loss_scores = selected_loss / (selected_loss.sum(dim=1, keepdim=True) + 1e-06)
        loss_scores = (loss_scores - loss_scores.min(dim=1, keepdim=True)[0]) / (loss_scores.max(dim=1, keepdim=True)[0] - loss_scores.min(dim=1, keepdim=True)[0] + 1e-06)
        scores = recon_score_weight * recon_scores + loss_scores
        drop_idx_batch = torch.argmax(scores, dim=1)
        all_idx = torch.arange(N, device=key_tensor.device).unsqueeze(0).expand(B, -1)
        mask = all_idx != drop_idx_batch.unsqueeze(1)
        keep_idx = all_idx[mask].view(B, N - 1)
        for i in range(len(self.long_term_memories)):
            k, v = self.long_term_memories[i]
            keep_idx_expanded = keep_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, D, W)
            k_perm = k.permute(2, 0, 1, 3)
            v_perm = v.permute(2, 0, 1, 3)
            k_new = torch.gather(k_perm, 1, keep_idx_expanded)
            v_new = torch.gather(v_perm, 1, keep_idx_expanded)
            self.long_term_memories[i] = (k_new.permute(1, 2, 0, 3), v_new.permute(1, 2, 0, 3))
        updated_lmi = []
        for b in range(B):
            drop_step = long_memories_indexes[b][drop_idx_batch[b].item()]
            updated_lmi.append([idx for idx in long_memories_indexes[b] if idx != drop_step])
        long_memories_indexes[:] = updated_lmi
        return (long_memories_indexes, False)

    def init_memory(self, size_2d=(30, 30)):
        self.long_term_memories = self.lstt_long_memories
        self.short_term_memories_list = [self.lstt_short_memories]
        self.short_term_memories = self.lstt_short_memories
        self.stored_attn_weight_dict = {}
        self.stored_frame_times = {}
        self.latest_long_term_attention_score = None
        self.rmem_relevance_ema = None
        self.rmem_access_count = None
        self.rmem_frame_indices = None
        if self.gru_memory:
            l, b, c = self.short_term_memories[0][0].size()
            dtype = self.short_term_memories[0][0].dtype
            device = self.short_term_memories[0][0].device
            self.long_term_memory_hidden_states = [[ConvGRUCell.init_hidden(b, c, size_2d, dtype).to(device), ConvGRUCell.init_hidden(b, c, size_2d, dtype).to(device)] for _ in range(len(self.layers))]

    def clear_memory(self):
        self.lstt_curr_memories = None
        self.lstt_long_memories = None
        self.lstt_short_memories = None
        self.short_term_memories_list = []
        self.short_term_memories = None
        self.long_term_memories = None
        self.latest_long_term_attention_score = None
        self.rmem_relevance_ema = None
        self.rmem_access_count = None
        self.rmem_frame_indices = None
        self.long_term_memory_hidden_states = None

class SimplifiedTransformerBlock(nn.Module):

    def __init__(self, d_model, self_nhead, att_nhead, dim_feedforward=1024, droppath=0.1, activation='gelu', linear_q=False, time_encode=False, gru_memory=False):
        super().__init__()
        self.norm1 = _get_norm(d_model)
        self.self_attn = MultiheadAttention(d_model, self_nhead)
        self.norm2 = _get_norm(d_model)
        self.linear_Q = nn.Linear(d_model, d_model)
        self.linear_V = nn.Linear(d_model, d_model)
        self.linear_QMem = nn.Linear(d_model, d_model)
        self.linear_VMem = nn.Linear(d_model, d_model)
        if not linear_q:
            self.norm4 = _get_norm(d_model)
        self.linear_KMem = nn.Linear(d_model, d_model)
        self.long_term_attn = MultiheadAttention(d_model, att_nhead, use_linear=False)
        self.short_term_attn = MultiheadAttention(d_model, att_nhead, use_linear=False)
        self.norm3 = _get_norm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = GNActDWConv2d(dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.droppath = DropPath(droppath, batch_dim=1)
        self.linear_q = linear_q
        self._init_weight()
        if time_encode:
            self.Q_time_encode = nn.Sequential(nn.Linear(in_features=d_model, out_features=d_model), nn.ReLU(), nn.Linear(in_features=d_model, out_features=d_model))
            self.K_time_encode = nn.Sequential(nn.Linear(in_features=d_model, out_features=d_model), nn.ReLU(), nn.Linear(in_features=d_model, out_features=d_model))
        if gru_memory:
            self.memory_grus = nn.ModuleList([ConvGRUCellOutput(input_dim=d_model, hidden_dim=d_model, kernel_size=(2, 2), bias=True, output_dim=d_model), ConvGRUCellOutput(input_dim=d_model, hidden_dim=d_model, kernel_size=(1, 1), bias=True, output_dim=d_model)])

    def with_pos_embed(self, tensor, pos=None):
        size = tensor.size()
        if len(size) == 4 and pos is not None:
            n, c, h, w = size
            pos = pos.view(h, w, n, c).permute(2, 3, 0, 1)
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, long_term_memory=None, short_term_memory=None, curr_id_emb=None, self_pos=None, size_2d=(30, 30), temporal_encoding=None, save_atten_weights=False, return_ltmb_attn_scores=False):
        _tgt = self.norm1(tgt)
        q = k = self.with_pos_embed(_tgt, self_pos)
        v = _tgt
        tgt2 = self.self_attn(q, k, v)[0]
        tgt = tgt + self.droppath(tgt2)
        _tgt = self.norm2(tgt)
        curr_Q = self.linear_Q(_tgt)
        curr_K = curr_Q
        curr_V = _tgt
        local_Q = curr_Q
        if long_term_memory is None:
            global_K = curr_K
            global_V = self.linear_V(curr_V)
            local_K = global_K
            local_V = global_V
            global_K = global_K[None, ...]
            global_V = global_V[None, ...]
            has_existing_long_term_memory = False
        else:
            global_K, global_V = long_term_memory
            local_K, local_V = short_term_memory
            has_existing_long_term_memory = True
        if temporal_encoding is not None:
            T, token_num, bs, embed_dim = global_K.shape
            cur_pos_emb, mem_pos_emb = (temporal_encoding[0:1], temporal_encoding[1:])
            max_T = 4
            if T <= mem_pos_emb.size(0):
                mem_pos_emb = mem_pos_emb[:T]
            if T == 1:
                flatten_global_K = global_K + mem_pos_emb[0].view(1, 1, 1, embed_dim)
            else:
                interpolated_mem_pe = mem_pos_emb.clone().permute(1, 0).view(1, embed_dim, -1)
                if T <= max_T:
                    interpolated_mem_pe = F.interpolate(interpolated_mem_pe, size=T, mode='linear', align_corners=True)
                else:
                    interpolated_mem_pe = F.interpolate(interpolated_mem_pe, size=max_T, mode='linear', align_corners=True)
                    interpolated_mem_pe = torch.flip(interpolated_mem_pe, dims=(-1,))
                    interpolated_mem_pe = F.interpolate(interpolated_mem_pe, size=T, mode='nearest')
                    interpolated_mem_pe = torch.flip(interpolated_mem_pe, dims=(-1,))
                interpolated_mem_pe = interpolated_mem_pe.view(embed_dim, T).permute(1, 0).contiguous()
                flatten_global_K = global_K.view(T, token_num, bs, embed_dim) + interpolated_mem_pe.view(T, 1, 1, embed_dim)
            flatten_global_K = flatten_global_K.flatten(0, 1)
            curr_Q_add_time = curr_Q + cur_pos_emb.view(1, 1, embed_dim)
        else:
            flatten_global_K = global_K.flatten(0, 1)
            curr_Q_add_time = curr_Q
        flatten_global_V = global_V.flatten(0, 1)
        self.long_term_attention_entry_score = None
        if return_ltmb_attn_scores and has_existing_long_term_memory and (not save_atten_weights):
            tgt2, self.long_term_attention_entry_score = self.long_term_attn.forward_with_entry_scores(curr_Q_add_time, flatten_global_K, flatten_global_V, global_K.size(0), global_K.size(1))
            attn = None
        else:
            tgt2, attn = self.long_term_attn(curr_Q_add_time, flatten_global_K, flatten_global_V, is_return_attn_weight=save_atten_weights or return_ltmb_attn_scores)
            if return_ltmb_attn_scores and has_existing_long_term_memory:
                self.long_term_attention_entry_score = compute_ltmb_attention_entry_score(attn, global_K.size(0), global_K.size(1))
        if save_atten_weights:
            bs, head, hw, thw = attn.size()
            self.record_T = thw // hw
            attn = attn.view((bs, head, hw, self.record_T, hw))
            attn = attn.mean(dim=1)
            assert attn.size(0) == 1
            attn = attn.squeeze(0)
            self.record_attn_weight = attn.sum(dim=2)
            self.attn_values, self.attn_indices = attn.detach().view((hw, thw)).topk(32, dim=-1)
            self.attn_values, self.attn_indices = (self.attn_values.cpu().numpy(), self.attn_indices.cpu().numpy())
            self.attn_indices = np.unravel_index(self.attn_indices, (self.record_T, hw))
        if self.linear_q:
            tgt3 = self.short_term_attn(local_Q, torch.cat((local_K, curr_K), 0), torch.cat((local_V, curr_V), 0))[0]
        else:
            tgt3, short_attn = self.short_term_attn(local_Q, self.norm4(local_K + curr_K), self.norm4(local_V + curr_V), is_return_attn_weight=save_atten_weights)
        if save_atten_weights:
            short_attn = short_attn.mean(dim=1)
            assert short_attn.size(0) == 1
            short_attn = short_attn.squeeze(0)
            self.short_attn_values, self.short_attn_indices = short_attn.detach().topk(32, dim=-1)
            self.short_attn_values, self.short_attn_indices = (self.short_attn_values.cpu().numpy(), self.short_attn_indices.cpu().numpy())
        _tgt3 = tgt3
        local_K = self.linear_QMem(_tgt3)
        local_V = _tgt3
        if curr_id_emb is not None:
            local_V = self.linear_VMem(local_V + curr_id_emb)
        w1 = 0.8
        w2 = 1.2
        tgt = tgt + w1 * tgt2 + w2 * tgt3
        _tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.activation(self.linear1(_tgt), size_2d))
        tgt = tgt + self.droppath(tgt2)
        return (tgt, [[curr_K, curr_V], [global_K, global_V], [local_K, local_V]])

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
