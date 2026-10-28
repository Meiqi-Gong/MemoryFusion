import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.math import generate_permute_matrix
from networks.models.aot import AOT
from networks.layers.position import get_temporal_positional_encoding
from loss.loss import Fusion_singleloss, Fusion_allloss, laplacian_var_noise_score
from networks.debug import debug
USE_ATTEN_WEIGHT_DROP = True

class AOTEngine(nn.Module):

    def __init__(self, aot_model: AOT, gpu_id=0, long_term_mem_gap=9999, short_term_mem_skip=1):
        super().__init__()
        self.cfg = aot_model.cfg
        self.align_corners = aot_model.cfg.MODEL_ALIGN_CORNERS
        self.AOT = aot_model
        self.max_obj_num = aot_model.max_obj_num
        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        self.losses = None
        self.calculate_loss = Fusion_singleloss().cuda()
        self.calculate_allloss = Fusion_allloss().cuda()
        self.restart_engine()
        self.first_update = True

    def forward(self, vis_imgs, ir_imgs, all_visframes, all_irframes, batch_size, step=0, tf_board=False, use_prev_pred=False):
        self.freeze_id = True if use_prev_pred else False
        bs, self.total_offline_frame_num = (all_visframes.shape[0], all_visframes.shape[2])
        self.offline_enc_embs = self.AOT.encode_image(all_visframes[:, :, self.frame_step], all_irframes[:, :, self.frame_step])
        fused_frames, curr_losses, lstt_feats, cur_feats = ([], [], [], [])
        pred_s1, pred_s2, pred_fused, lstt_feat, curr_embbeds = self.add_reference_frame(vis=all_visframes[:, :, self.frame_step], ir=all_irframes[:, :, self.frame_step], frame_step=0)
        fused_frames.append([pred_s1, pred_s2, pred_fused])
        quality_score = laplacian_var_noise_score(torch.clamp(pred_fused, 0.0, 1.0))
        curr_losses.append(quality_score)
        lstt_feats.append(lstt_feat)
        cur_feats.append(curr_embbeds)
        self.frame_step += 1
        self.offline_enc_embs = self.AOT.encode_image(all_visframes[:, :, self.frame_step], all_irframes[:, :, self.frame_step])
        pred_s1, pred_s2, pred_fused, lstt_feat, curr_embbeds = self.match_propogate_one_frame(all_visframes[:, :, self.frame_step], all_irframes[:, :, self.frame_step])
        fused_frames.append([pred_s1, pred_s2, pred_fused])
        quality_score = laplacian_var_noise_score(torch.clamp(pred_fused, 0.0, 1.0))
        curr_losses.append(quality_score)
        lstt_feats.append(lstt_feat)
        cur_feats.append(curr_embbeds)
        for _ in range(self.total_offline_frame_num - 2):
            self.update_short_term_memory(self.offline_enc_embs[-1], step=step, loss=curr_losses)
            self.frame_step += 1
            self.offline_enc_embs = self.AOT.encode_image(all_visframes[:, :, self.frame_step], all_irframes[:, :, self.frame_step])
            pred_s1, pred_s2, pred_fused, lstt_feat, curr_embbeds = self.match_propogate_one_frame(all_visframes[:, :, self.frame_step], all_irframes[:, :, self.frame_step])
            fused_frames.append([pred_s1, pred_s2, pred_fused])
            quality_score = laplacian_var_noise_score(torch.clamp(pred_fused, 0.0, 1.0))
            curr_losses.append(quality_score)
            lstt_feats.append(lstt_feat)
            cur_feats.append(curr_embbeds)
        s1_video = torch.cat([frame[0] for frame in fused_frames], dim=0)
        s2_video = torch.cat([frame[1] for frame in fused_frames], dim=0)
        fused_video = torch.cat([frame[2] for frame in fused_frames], dim=0)
        ir_gt = torch.cat(ir_imgs, dim=0)
        vis_gt = torch.cat(vis_imgs, dim=0)
        bs = ir_imgs[0].shape[0]
        curr_loss = self.calculate_allloss(fused_video, ir_gt, vis_gt, len(fused_frames), bs)
        curr_loss1 = self.calculate_allloss(s1_video, F.interpolate(ir_gt, scale_factor=0.25, mode='bilinear', align_corners=False), F.interpolate(vis_gt, scale_factor=0.25, mode='bilinear', align_corners=False), len(fused_frames), bs)
        curr_loss2 = self.calculate_allloss(s2_video, F.interpolate(ir_gt, scale_factor=0.5, mode='bilinear', align_corners=False), F.interpolate(vis_gt, scale_factor=0.5, mode='bilinear', align_corners=False), len(fused_frames), bs)
        if step % self.cfg.TRAIN_LOG_STEP == 0:
            print(curr_loss)
        self.first_update = True
        self.AOT.LSTT._long_memories_indexes_initialized = True
        self.AOT.decoder.smoother1.prev_feature = None
        self.AOT.decoder.smoother2.prev_feature = None
        self.AOT.decoder.smoother3.prev_feature = None
        self.AOT.decoder.smoother1.prev_vis = None
        self.AOT.decoder.smoother2.prev_vis = None
        self.AOT.decoder.smoother3.prev_vis = None
        return (curr_loss, curr_loss['loss_fusion'] + 0.1 * curr_loss1['loss_fusion'] + 0.1 * curr_loss2['loss_fusion'], torch.stack([frame[2] for frame in fused_frames], dim=1).permute(0, 2, 1, 3, 4), lstt_feats, cur_feats)

    def add_reference_frame(self, img=None, frame_step=-1, img_embs=None, vis=None, ir=None):
        if frame_step == -1:
            frame_step = self.frame_step
        curr_enc_embs = self.offline_enc_embs
        self.ref_enc_embs = curr_enc_embs
        if self.input_size_2d is None:
            self.update_size(curr_enc_embs[-1].size()[2:])
        if self.pos_emb is None:
            self.pos_emb = self.AOT.get_pos_emb(curr_enc_embs[-1]).expand(self.batch_size, -1, -1, -1).view(self.batch_size, -1, self.enc_hw).permute(2, 0, 1)
        if self.cfg.TIME_ENCODE and (not self.cfg.TIME_ENCODE_NORM):
            self.temporal_encoding = get_temporal_positional_encoding(max_sequence_len=32, channels=curr_enc_embs[-1].size()[1], device=curr_enc_embs[-1].device, is_normalize=True, scale=1.57)
        else:
            self.temporal_encoding = None
        if self.AOT.use_temporal_pe:
            temporal_pos_emb = torch.cat((self.AOT.cur_pos_emb, self.AOT.mem_pos_emb), dim=0)
        else:
            temporal_pos_emb = None
        curr_lstt_output = self.AOT.LSTT_forward(curr_enc_embs, pos_emb=self.pos_emb, size_2d=self.enc_size_2d, temporal_encoding=temporal_pos_emb, return_ltmb_attn_scores=False)
        self.last_mem_step = frame_step
        self.AOT.init_LSTT_memory(size_2d=self.enc_size_2d)
        self.long_memories_indexes.append(self.frame_step)
        pred_s1, pred_s2, pred_fused, lstt_feat, curr_embbeds = self.decode_fusion(curr_enc_embs, curr_lstt_output, vis, ir)
        return (pred_s1, pred_s2, pred_fused, lstt_feat, curr_embbeds)

    def update_short_term_memory(self, curr_emb, step, loss):
        is_update_long_memory = False
        if not (hasattr(self.cfg, 'NO_LONG_MEMORY') and self.cfg.NO_LONG_MEMORY) and self.frame_step - self.last_mem_step >= self.long_term_mem_gap:
            is_update_long_memory = True
            self.last_mem_step = self.frame_step
        self.AOT.update_short_term_memory(curr_id_emb=curr_emb, short_term_mem_skip=self.short_term_mem_skip, size_2d=self.enc_size_2d, is_update_long_memory=is_update_long_memory)
        if is_update_long_memory:
            if self.first_update:
                self.long_memories_indexes.append(self.frame_step)
            else:
                for lst in self.long_memories_indexes:
                    lst.append(self.frame_step)
            debug(f'self.long_memories_indexes = {self.long_memories_indexes!r}')
            self.long_memories_indexes, self.first_update = self.AOT.LSTT.restrict_long_memories(former_memory_len=self.cfg.FORMER_MEM_LEN, latter_memory_len=self.cfg.LATTER_MEM_LEN, use_atten_weight=USE_ATTEN_WEIGHT_DROP and (not self.training), long_memories_indexes=self.long_memories_indexes, loss=loss, frame_step=self.frame_step, ltmb_update_strategy=getattr(self.cfg, 'LTMB_UPDATE_STRATEGY', 'ours'), debug_ltmb_update=getattr(self.cfg, 'DEBUG_LTMB_UPDATE', False), recon_score_weight=getattr(self.cfg, 'RECON_SCORE_WEIGHT', 1.0), rmem_relevance_ema_beta=getattr(self.cfg, 'RMEM_RELEVANCE_EMA_BETA', 0.2), rmem_freshness_weight=getattr(self.cfg, 'RMEM_FRESHNESS_WEIGHT', 1.5), rmem_freshness_add=getattr(self.cfg, 'RMEM_FRESHNESS_ADD', 8.0))
            debug(f'self.AOT.LSTT.long_term_memories[0][0].size(0) = {self.AOT.LSTT.long_term_memories[0][0].size(0)!r}')

    def match_propogate_one_frame(self, vis, ir):
        curr_enc_embs = self.offline_enc_embs
        if self.cfg.TIME_ENCODE_NORM:
            self.temporal_encoding = get_temporal_positional_encoding(max_sequence_len=self.AOT.LSTT.long_term_memories[0][0].size(0) + 1, channels=curr_enc_embs[-1].size()[1], device=curr_enc_embs[-1].device, is_normalize=True, scale=1.0)
        if self.AOT.use_temporal_pe:
            temporal_pos_emb = torch.cat((self.AOT.cur_pos_emb, self.AOT.mem_pos_emb), dim=0)
        else:
            temporal_pos_emb = None
        curr_lstt_output = self.AOT.LSTT_forward(curr_enc_embs, None, pos_emb=self.pos_emb, size_2d=self.enc_size_2d, temporal_encoding=temporal_pos_emb, save_atten_weights=False, return_ltmb_attn_scores=getattr(self.cfg, 'LTMB_UPDATE_STRATEGY', 'ours') in ('attention', 'rmem'))
        pred_s1, pred_s2, result, lstt_feat, curr_embbeds = self.decode_fusion(curr_enc_embs, curr_lstt_output, vis, ir)
        return (pred_s1, pred_s2, result, lstt_feat, curr_embbeds)

    def decode_fusion(self, curr_enc_embs, curr_lstt_embs, vis, ir):
        pred_s1, pred_s2, pred_fusion, lstt_feat, curr_embbeds = self.AOT.decode(curr_enc_embs, curr_lstt_embs, vis, ir)
        return (pred_s1, pred_s2, pred_fusion, lstt_feat, curr_embbeds)

    def restart_engine(self, batch_size=1, enable_id_shuffle=False):
        self.batch_size = batch_size
        self.frame_step = 0
        self.last_mem_step = -1
        self.enable_id_shuffle = enable_id_shuffle
        self.freeze_id = False
        self.obj_nums = None
        self.pos_emb = None
        self.enc_size_2d = None
        self.enc_hw = None
        self.input_size_2d = None
        self.AOT.clear_LSTT_memory()
        self.long_memories_indexes = []
        self.enable_offline_enc = False
        self.offline_enc_embs = None
        self.offline_one_hot_masks = None
        self.offline_ignore_masks = None
        self.offline_frames = -1
        self.total_offline_frame_num = 0
        self.curr_memories = None
        self.curr_id_embs = None
        if enable_id_shuffle:
            self.id_shuffle_matrix = generate_permute_matrix(self.max_obj_num + 1, batch_size, gpu_id=self.gpu_id)
        else:
            self.id_shuffle_matrix = None

    def update_size(self, enc_size):
        self.enc_size_2d = enc_size
        self.enc_hw = self.enc_size_2d[0] * self.enc_size_2d[1]

class AOTInferEngine(nn.Module):

    def __init__(self, aot_model, gpu_id=0, long_term_mem_gap=9999, short_term_mem_skip=1, max_aot_obj_num=None):
        super().__init__()
        self.cfg = aot_model.cfg
        self.AOT = aot_model
        if max_aot_obj_num is None or max_aot_obj_num > aot_model.max_obj_num:
            self.max_aot_obj_num = aot_model.max_obj_num
        else:
            self.max_aot_obj_num = max_aot_obj_num
        self.gpu_id = gpu_id
        self.long_term_mem_gap = long_term_mem_gap
        self.short_term_mem_skip = short_term_mem_skip
        self.aot_engines = AOTEngine(self.AOT, self.gpu_id, self.long_term_mem_gap, self.short_term_mem_skip).eval()
        self.restart_engine()

    def restart_engine(self):
        self.aot_engines.restart_engine()
        self.frame_step = 0

    def forward(self, vis_imgs, ir_imgs, all_visframes, all_irframes, frame_num=None, step=0, use_prev_pred=False, return_features=True, compute_loss=True):
        fused_frames, curr_losses, lstt_feats, enc_feats = ([], [], [], [])
        needs_frame_loss = compute_loss or getattr(self.cfg, 'LTMB_UPDATE_STRATEGY', 'ours') == 'ours'
        self.freeze_id = True if use_prev_pred else False
        self.offline_enc_embs = self.AOT.encode_image(all_visframes[:, :, self.frame_step], all_irframes[:, :, self.frame_step])
        self.total_offline_frame_num = all_visframes.shape[2] if frame_num == None else frame_num
        self.total_offline_frame_num = min(all_visframes.shape[2], self.total_offline_frame_num)
        bs = all_visframes.shape[0]
        _, _, pred_fused, lstt_feat, curr_embbeds = self.add_reference_frame(frame_step=0, offline_enc_embs=self.offline_enc_embs, vis=all_visframes[:, :, self.frame_step], ir=all_irframes[:, :, self.frame_step])
        if not torch.isfinite(pred_fused).all():
            raise FloatingPointError('Non-finite prediction at frame 0')
        fused_frames.append(pred_fused)
        if needs_frame_loss:
            curr_loss = self.aot_engines.calculate_loss(pred_fused, ir_imgs[self.frame_step], vis_imgs[self.frame_step], len(fused_frames), bs)
            curr_losses.append(curr_loss['loss_fusion'])
        if return_features:
            lstt_feats.append(lstt_feat)
            enc_feats.append(curr_embbeds)
        self.frame_step += 1
        self.offline_enc_embs = self.AOT.encode_image(all_visframes[:, :, self.frame_step], all_irframes[:, :, self.frame_step])
        _, _, pred_fused, lstt_feat, curr_embbeds = self.match_propogate_one_frame(offline_enc_embs=self.offline_enc_embs, vis=all_visframes[:, :, self.frame_step], ir=all_irframes[:, :, self.frame_step])
        if not torch.isfinite(pred_fused).all():
            raise FloatingPointError(f'Non-finite prediction at frame {self.frame_step}')
        fused_frames.append(pred_fused)
        if needs_frame_loss:
            curr_loss = self.aot_engines.calculate_loss(pred_fused, ir_imgs[self.frame_step], vis_imgs[self.frame_step], len(fused_frames), bs)
            curr_losses.append(curr_loss['loss_fusion'])
        if return_features:
            lstt_feats.append(lstt_feat)
            enc_feats.append(curr_embbeds)
        for _ in range(self.total_offline_frame_num - 2):
            self.aot_engines.frame_step = self.frame_step
            self.aot_engines.update_short_term_memory(self.offline_enc_embs[-1], step=step, loss=curr_losses)
            self.frame_step += 1
            self.offline_enc_embs = self.AOT.encode_image(all_visframes[:, :, self.frame_step], all_irframes[:, :, self.frame_step])
            _, _, pred_fused, lstt_feat, curr_embbeds = self.match_propogate_one_frame(offline_enc_embs=self.offline_enc_embs, vis=all_visframes[:, :, self.frame_step], ir=all_irframes[:, :, self.frame_step])
            if not torch.isfinite(pred_fused).all():
                raise FloatingPointError(f'Non-finite prediction at frame {self.frame_step}')
            fused_frames.append(pred_fused)
            if needs_frame_loss:
                curr_loss = self.aot_engines.calculate_loss(pred_fused, ir_imgs[self.frame_step], vis_imgs[self.frame_step], len(fused_frames), bs)
                curr_losses.append(curr_loss['loss_fusion'])
            if return_features:
                lstt_feats.append(lstt_feat)
                enc_feats.append(curr_embbeds)
        val_loss = None
        if compute_loss:
            fused_video = torch.cat(fused_frames, dim=0)
            bs = ir_imgs[0].shape[0]
            if isinstance(ir_imgs, list):
                ir_cat = torch.cat(ir_imgs[:self.total_offline_frame_num], dim=0)
                vis_cat = torch.cat(vis_imgs[:self.total_offline_frame_num], dim=0)
            else:
                ir_cat = ir_imgs.squeeze(1)
                vis_cat = vis_imgs.squeeze(1)
            val_loss = self.aot_engines.calculate_allloss(fused_video, ir_cat, vis_cat, len(fused_frames), bs)
        self.aot_engines.first_update = True
        self.AOT.LSTT._long_memories_indexes_initialized = True
        self.AOT.decoder.smoother1.prev_feature = None
        self.AOT.decoder.smoother2.prev_feature = None
        self.AOT.decoder.smoother3.prev_feature = None
        self.AOT.decoder.smoother1.prev_vis = None
        self.AOT.decoder.smoother2.prev_vis = None
        self.AOT.decoder.smoother3.prev_vis = None
        return (val_loss, torch.stack(fused_frames, dim=1).permute(0, 2, 1, 3, 4), lstt_feats, enc_feats)

    def add_reference_frame(self, frame_step=-1, offline_enc_embs=None, vis=None, ir=None):
        self.aot_engines.offline_enc_embs = offline_enc_embs
        pred_s1, pred_s2, pred_fused, lstt_feat, curr_embbeds = self.aot_engines.add_reference_frame(frame_step=frame_step, vis=vis, ir=ir)
        self.update_size()
        return (pred_s1, pred_s2, pred_fused, lstt_feat, curr_embbeds)

    def match_propogate_one_frame(self, offline_enc_embs=None, vis=None, ir=None):
        self.aot_engines.offline_enc_embs = offline_enc_embs
        pred_s1, pred_s2, pred_fused, lstt_feat, curr_embbeds = self.aot_engines.match_propogate_one_frame(vis, ir)
        return (pred_s1, pred_s2, pred_fused, lstt_feat, curr_embbeds)

    def update_size(self):
        self.input_size_2d = self.aot_engines.input_size_2d
        self.enc_size_2d = self.aot_engines.enc_size_2d
        self.enc_hw = self.aot_engines.enc_hw
