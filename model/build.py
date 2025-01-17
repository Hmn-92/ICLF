from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from .bert import BertConfig, BertForMaskedLM


def build_text_encoder(config, vision_width):
    config_text = BertConfig.from_json_file(config['text_config'])
    config_text.encoder_width = vision_width
    text_encoder, msg = BertForMaskedLM.from_pretrained(config['text_encoder'], config=config_text,
                                                        output_loading_info=True)
    return text_encoder


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim)
    )


class ICLF(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
                                                                      args.stride_size)
        # self.base_model, base_cfg = build_CLIP_from_openai_pretrained('ViT-B/16', (384, 128),stride_size=16)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                             64)
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
        self.itm_head = build_mlp(input_dim=self.embed_dim, output_dim=2)

        # 设置并加载bert
        self.text_encoder = build_text_encoder({'text_config': 'data/config_bert.json',
                                                'text_encoder': 'data/bert-base-uncased'},
                                               vision_width=512)

        self.text_width = self.text_encoder.config.hidden_size  # i.e. cross_width
        # self.itm_head = build_mlp(input_dim=self.text_width, output_dim=2)
        self.temp = nn.Parameter(torch.ones([]) * 0.07)

        # 需要调整clip提取出来的特征到bert的输入
        self.text_feats_2_bert = nn.Linear(512, 768, False)
        # self.image_feats_2_bert = nn.Linear(512, 768, False)

        # 控制两部分loss的权重，要求在0-1之前
        self.weight = args.weight

        # embed_dim = 768
        # self.queue_size = 65536
        self.alpha = 0.4

        # self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        # self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        # self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        # self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
            self.ln_pre_t(q),
            self.ln_pre_i(k),
            self.ln_pre_i(v),
            need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def get_cross_embeds(self, image_embeds, image_atts=None, text_ids=None, text_embeds=None, text_atts=None):
        # assert text_atts is not None
        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder
        if text_embeds is not None:
            return encoder(encoder_embeds=text_embeds,
                           attention_mask=text_atts,
                           encoder_hidden_states=image_embeds,
                           encoder_attention_mask=image_atts,
                           return_dict=True,
                           mode='fusion',
                           ).last_hidden_state
        elif text_ids is not None:
            return encoder(text_ids,
                           attention_mask=text_atts,
                           encoder_hidden_states=image_embeds,
                           encoder_attention_mask=image_atts,
                           return_dict=True,
                           ).last_hidden_state
        else:
            raise ValueError

    # def get_matching_loss(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=None):
    def get_matching_loss(self, image_embeds, text_embeds, image_feat, text_feat, idx=None, weight=1):
        """
        Matching Loss with hard negatives
        """
        bs = image_embeds.size(0)

        image_feat = F.normalize(image_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)

        with torch.no_grad():
            # if 1:
            sim_i2t = image_feat @ text_feat.t() / self.temp
            # sim_i2i = image_feat @ image_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp
            # sim_t2t = text_feat @ text_feat.t() / self.temp

            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5

            if idx is None:
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
            else:
                idx = idx.view(-1, 1)
                assert idx.size(0) == bs
                mask = torch.eq(idx, idx.t())
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)

        image_embeds_neg = []
        # image_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
            # image_atts_neg.append(image_atts[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        # image_atts_neg = torch.stack(image_atts_neg, dim=0)

        text_embeds_neg = []
        # text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            # text_atts_neg.append(text_atts[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        # text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        # text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        # image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0)
        image_atts = text_atts = image_atts_all = text_atts_all = None

        text_embeds1 = self.text_feats_2_bert(text_embeds)
        self.text_embeds1 = text_embeds1
        text_embeds_all1 = self.text_feats_2_bert(text_embeds_all)

        self.cross_embs = self.cross_former(text_embeds, image_embeds, image_embeds) # 8 77 512

        # self.cross_embs = self.get_cross_embeds(image_embeds,
        #                                   image_atts,
        #                                   text_embeds=text_embeds1,
        #                                   text_atts=text_atts
        #                                   )# 8 77 768

        cross_pos = self.cross_embs[:, 0, :]

        cross_neg = self.cross_former(text_embeds_all, image_embeds_all, image_embeds_all) # 8 77 512
        cross_neg = cross_neg[:, 0, :]

        # cross_neg = self.get_cross_embeds(image_embeds_all,
        #                                   image_atts_all,
        #                                   text_embeds=text_embeds_all1,
        #                                   text_atts=text_atts_all
        #                                   )[:, 0, :]

        output = self.itm_head(torch.cat([cross_pos, cross_neg], dim=0))
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long),
                                torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image_embeds.device)
        itm_loss = F.cross_entropy(output, itm_labels)

        ###################################################################################
        ###################################################################################
        ###################################################################################
        alpha = self.alpha

        idx = idx.view(-1, 1)
        idx_all = idx.t()
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        sim_i2i = image_feat @ image_feat.t() / self.temp
        sim_t2t = text_feat @ text_feat.t() / self.temp

        # Contrastive loss
        with torch.no_grad():
            sim_i2i_m = image_feat @ image_feat.t() / self.temp
            sim_t2t_m = text_feat @ text_feat.t() / self.temp

            sim_i2i_targets = alpha * F.softmax(sim_i2i_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2t_targets = alpha * F.softmax(sim_t2t_m, dim=1) + (1 - alpha) * sim_targets

        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1) * sim_i2i_targets, dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1) * sim_t2t_targets, dim=1).mean()
        loss_cl = (loss_i2i + loss_t2t) / 2

        itm_loss = weight * itm_loss + (1 - weight) * loss_cl

        return itm_loss

    def forward(self, batch):
        self.text_embeds1 = None
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)  # 8 193 512 、 8 77 512
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_itc(i_feats, t_feats, logit_scale)})

        if 'itm' in self.current_task:
            # text_feats_bert = self.text_feats_2_bert(text_feats)
            # image_feats_bert = self.image_feats_2_bert(image_feats)
            # i_feats_bert = image_feats_bert[:, 0, :].float()
            # t_feats_bert = text_feats_bert[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

            text_feats_bert = text_feats
            image_feats_bert = image_feats
            i_feats_bert = i_feats
            t_feats_bert = t_feats

            ret.update({'itm_loss': self.get_matching_loss(image_feats_bert,
                                                           text_feats_bert,
                                                           i_feats_bert,
                                                           t_feats_bert,
                                                           batch['pids'],
                                                           weight=self.weight,
                                                           )})

        if 'sdm' in self.current_task:
            ret.update({'sdm_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss': objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})

        if 'id' in self.current_task:
            # image_logits = self.classifier(i_feats.half()).float()
            # text_logits = self.classifier(t_feats.half()).float()
            image_logits = self.classifier(i_feats).float()
            text_logits = self.classifier(t_feats).float()

            ret.update(
                {'id_loss': objectives.compute_id(image_logits, text_logits, batch['pids']) * self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if 'mlm' in self.current_task:
            # mlm_ids = batch['mlm_ids']
            # # 提取特征
            # mlm_feats = self.base_model.encode_text(mlm_ids)
            # # 转为bert需要的维度
            # if self.text_embeds1 is None:
            #     mlm_feats = self.text_feats_2_bert(mlm_feats)
            # else:
            #     mlm_feats = self.text_embeds1
            #
            # # 使用bert计算cross_embed
            # if self.cross_embs is None:
            #     x = self.get_cross_embeds(image_feats, text_embeds=mlm_feats)
            # else:
            #     x = self.cross_embs
            # # 分类
            # x = self.mlm_head(x)  # [batch_size, text_len, num_colors]
            #
            # scores = x.float().reshape(-1, self.args.vocab_size)
            # mlm_labels = batch['mlm_labels'].reshape(-1)
            # ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight})

            # pred = scores.max(1)[1]
            # mlm_label_idx = torch.nonzero(mlm_labels)
            # acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            # ret.update({'mlm_acc': acc})

            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        return ret


def build_model(args, num_classes=11003):
    model = ICLF(args, num_classes)
    # covert model to fp16
    # convert_weights(model)
    return model
