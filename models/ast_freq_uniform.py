"""
AST Model with Frequency Band Embedding (FBE)
创新点：在patch进入Transformer之前，注入频率先验知识
每个patch额外增加一个频段嵌入向量，引导注意力机制优先关注病理相关的频率区域
"""
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
import timm
from copy import deepcopy
from timm.models.layers import to_2tuple, trunc_normal_


# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class FrequencyBandEmbedding(nn.Module):
    """
    频段嵌入模块 (Frequency Band Embedding, FBE)

    核心思想：
    在呼吸音频谱图中，不同频段携带不同的病理信息：
    - 低频段 (0-500 Hz): 心音、背景噪声
    - 中频段 (500-1500 Hz): 啰音 (crackles) 的主要能量区域
    - 中高频段 (1500-2500 Hz): 哮鸣音 (wheezes) 的主要能量区域
    - 高频段 (>2500 Hz): 呼吸音细节、噪声

    通过为每个patch注入频段嵌入，模型可以：
    1. 明确知道每个patch属于哪个频段
    2. 学习频段特异性的注意力模式
    3. 自动关注病理相关的频率区域

    参数说明：
    - num_freq_bands: 频率维度的patch数量 (由输入频谱图大小和patch参数决定)
    - embed_dim: 嵌入维度 (与Transformer隐藏维度一致)
    - dropout: dropout率
    """
    def __init__(self, num_freq_bands, embed_dim=768, dropout=0.1):
        super().__init__()
        self.num_freq_bands = num_freq_bands
        self.embed_dim = embed_dim

        # 可学习的频段嵌入表
        # 每个频率位置对应一个独立的嵌入向量
        self.freq_band_embed = nn.Embedding(num_freq_bands, embed_dim)

        # 可选的频段感知投影层（增强表达能力）
        self.freq_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化频段嵌入权重"""
        nn.init.trunc_normal_(self.freq_band_embed.weight, std=0.02)
        for m in self.freq_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, patch_embeddings, f_dim):
        """
        参数：
            patch_embeddings: patch嵌入 (B, num_patches, embed_dim)
            f_dim: 频率维度的patch数量

        返回：
            带有频段信息的patch嵌入 (B, num_patches, embed_dim)
        """
        B, num_patches, dim = patch_embeddings.shape
        device = patch_embeddings.device

        # 生成频率带索引
        # patches按行优先排列：前f_dim个patch对应第0行(最低频)，接下来f_dim个对应第1行，...
        freq_indices = torch.arange(f_dim, device=device)  # [0, 1, ..., f_dim-1]
        # 重复每个频率索引t_dim次（对应时间维度）
        t_dim = num_patches // f_dim
        freq_indices = freq_indices.repeat_interleave(t_dim)  # [0,0,...,0, 1,1,...,1, ...]
        # 扩展到batch维度
        freq_indices = freq_indices.unsqueeze(0).expand(B, -1)  # (B, num_patches)

        # 获取频段嵌入
        freq_embed = self.freq_band_embed(freq_indices)  # (B, num_patches, embed_dim)

        # 通过投影层增强表达能力
        freq_embed = self.freq_proj(freq_embed)

        # 与patch嵌入相加
        output = patch_embeddings + freq_embed

        return output


class ASTModel(nn.Module):
    """
    AST模型 with Frequency Band Embedding (FBE)

    在原始AST模型基础上，增加频段嵌入模块：
    patch_embed + pos_embed + freq_band_embed → Transformer

    这使得模型能够：
    1. 利用频率先验知识引导注意力机制
    2. 学习频段特异性的病理特征
    3. 提高对啰音(低频)和哮鸣音(中高频)的识别能力
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024,
                 imagenet_pretrain=True, audioset_pretrain=False, model_size='base384',
                 verbose=True, mix_beta=None, use_freq_band_embed=True):
        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose:
            print('---------------AST Model with FBE Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain), str(audioset_pretrain)))
            print('Frequency Band Embedding: {:s}'.format(str(use_freq_band_embed)))

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        self.final_feat_dim = 768
        self.mix_beta = mix_beta
        self.use_freq_band_embed = use_freq_band_embed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            self.f_dim = t_dim  # 保存频率维度patch数量（forward中transpose后，t_dim才是频率维）

            if verbose:
                print('frequency stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d} (f_dim={:d}, t_dim={:d})'.format(num_patches, f_dim, t_dim))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

            # ========== 新增：频段嵌入模块 ==========
            if self.use_freq_band_embed:
                self.freq_band_embed = FrequencyBandEmbedding(
                    num_freq_bands=t_dim,
                    embed_dim=self.original_embedding_dim,
                    dropout=0.1
                )
                if verbose:
                    print('Frequency Band Embedding: num_freq_bands={:d}, embed_dim={:d}'.format(t_dim, self.original_embedding_dim))
            # ========================================

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            out_dir = './pretrained_models/'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)

            if os.path.exists(os.path.join(out_dir, 'audioset_10_10_0.4593.pth')) == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out=os.path.join(out_dir, 'audioset_10_10_0.4593.pth'))

            sd = torch.load(os.path.join(out_dir, 'audioset_10_10_0.4593.pth'), map_location=device)
            audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False, use_freq_band_embed=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.f_dim = t_dim  # forward中transpose后，t_dim才是频率维
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print('frequency stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d} (f_dim={:d}, t_dim={:d})'.format(num_patches, f_dim, t_dim))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

            # ========== 新增：频段嵌入模块 (AudioSet预训练模式) ==========
            if self.use_freq_band_embed:
                self.freq_band_embed = FrequencyBandEmbedding(
                    num_freq_bands=t_dim,
                    embed_dim=self.original_embedding_dim,
                    dropout=0.1
                )
                if verbose:
                    print('Frequency Band Embedding: num_freq_bands={:d}, embed_dim={:d}'.format(t_dim, self.original_embedding_dim))
            # ============================================================

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def square_patch(self, patch, hw_num_patch):
        h, w = hw_num_patch
        B, _, dim = patch.size()
        square = patch.reshape(B, h, w, dim)
        return square

    def flatten_patch(self, square):
        B, h, w, dim = square.shape
        patch = square.reshape(B, h * w, dim)
        return patch

    def patch_mix(self, image, target, time_domain=False, hw_num_patch=None):
        if self.mix_beta > 0:
            lam = np.random.beta(self.mix_beta, self.mix_beta)
        else:
            lam = 1

        batch_size, num_patch, dim = image.size()
        device = image.device

        index = torch.randperm(batch_size).to(device)

        if not time_domain:
            num_mask = int(num_patch * (1. - lam))
            mask = torch.randperm(num_patch)[:num_mask].to(device)

            image[:, mask, :] = image[index][:, mask, :]
            lam = 1 - (num_mask / num_patch)
        else:
            squared_1 = self.square_patch(image, hw_num_patch)
            squared_2 = self.square_patch(image[index], hw_num_patch)

            w_size = squared_1.size()[2]
            num_mask = int(w_size * (1. - lam))
            mask = torch.randperm(w_size)[:num_mask].to(device)

            squared_1[:, :, mask, :] = squared_2[:, :, mask, :]
            image = self.flatten_patch(squared_1)
            lam = 1 - (num_mask / w_size)

        y_a, y_b = target, target[index]
        return image, y_a, y_b, lam, index

    @autocast()
    def forward(self, x, y=None, patch_mix=False, time_domain=False):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        h_patch, w_patch = int((x.size()[2] - 16) / 10) + 1, int((x.size()[3] - 16) / 10) + 1

        B = x.shape[0]
        x = self.v.patch_embed(x)  # (B, num_patches, 768)

        if patch_mix:
            x, y_a, y_b, lam, index = self.patch_mix(x, y, time_domain=time_domain, hw_num_patch=[h_patch, w_patch])

        # ========== 新增：注入频段嵌入 ==========
        if self.use_freq_band_embed:
            x = self.freq_band_embed(x, self.f_dim)  # (B, num_patches, 768)
        # ========================================

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for i, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        # x = self.mlp_head(x)

        if not patch_mix:
            return x
        else:
            return x, y_a, y_b, lam, index
