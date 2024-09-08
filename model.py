import os
import json
from pathlib import Path
from torch.nn import functional as F

import clip
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from .Qformer import BertConfig, BertLMHeadModel

from .llama import ModelArgs, Transformer, BERTTransformer
from .tokenizer import Tokenizer
from .utils import sample_top_p, _download
from .dist_utils import download_cached_file, is_url

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LLaMA_adapter(nn.Module):

    def __init__(self, llama_ckpt_dir, llama_tokenizer,
                 max_seq_len=512, max_batch_size=1,
                 query_len=577, query_layer=32, phase="finetune"):
        super().__init__()
        # llama configs
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())
        bias_lora = phase == "finetune"
        # bias_lora = True
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        ) 

        self.query_len = query_len
        self.query_layer = query_layer

        visual_model_args = ModelArgs(dim=1024, n_layers=16, n_heads=8, max_seq_len=577)
        visual_model_args.vocab_size = 1024
        self.Qformer, self.query_tokens = self.init_Qformer(
            576, 768, 2
        )
        self.bev_conv1= nn.Conv2d(in_channels=512, out_channels=768, kernel_size=(1,1), stride = (1, 1) )
        self.bev_proj = nn.Linear(768, 768)
        self.bev_proj_norm = nn.LayerNorm(768)
        self.vision_proj = nn.Linear(768, model_args.dim)
        self.vision_proj_norm = nn.LayerNorm(model_args.dim)
        self.VIEW_RANGE = [
                [[0,35], [325,360]],    # FRONT         FOV:70    MIDDLE:0
                [[270,340]],            # FRONT_LEFT    FOV:70    MIDDLE:-55
                [[20,90]],              # FRONT_RIGHT   FOV:70    MIDDLE:55
                [[125,235]],            # BACK          FOV:110   MIDDLE:180
                [[75,145]],             # BACK_LEFT     FOV:70    MIDDLE:-110
                [[215,285]]            # BACK_RIGHT    FOV:70    MIDDLE:110
            ]
        self.view_masks = self.generate_angles(C=768, H=180,W=180) 
        self.angle_pos_embd = nn.Parameter(
            torch.ones((6, 1, 1, 1, 768))     
            )   

        # 3. adapter query
        self.adapter_query = nn.Embedding(
            query_len * query_layer, model_args.dim)

        # 4. tokenizer
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # 5. llama 
        model_args.w_bias = bias_lora
        model_args.w_lora = bias_lora
        model_args.vocab_size = self.tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.llama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
        for ckpt in ckpts:
            print('load_ckpt_path:', ckpt)
            ckpt = torch.load(ckpt, map_location='cpu')
            self.llama.load_state_dict(ckpt, strict=False)
        

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.phase = phase
        self.get_trainable_params(self.phase)
        for name, param in self.named_parameters():
            if param.requires_grad:
               print(f"Trainable param: {name}, {param.shape}, {param.dtype}")
        print('stop')

    def get_trainable_params(self, phase='finetune'):
        for name, para in self.named_parameters():
            para.requires_grad = False

        if phase == 'finetune':
            for name, para in self.llama.named_parameters():
                if 'norm' in name:
                    para.data = para.data.float()
                    para.requires_grad = True
                if 'bias' in name:
                    para.data = para.data.float()
                    para.requires_grad = True
                if 'lora' in name:
                    para.data = para.data.float()
                    para.requires_grad = True

        elif phase == 'pretrain':
            train_param_name = ['Qformer', 'angle_pos_embd', 'query_tokens', \
                                'vision_proj', 'vision_proj_norm', 'bev', \
                                'gate', 'visual_query', 'visual_blocks', \
                                 'adapter_query']
            for name, para in self.named_parameters():
                for train_name in train_param_name:
                    if train_name in name:
                        para.data = para.data.float()
                        para.requires_grad = True

        else:
            raise ValueError(f"Unknown model phase: {phase}")
     


    def forward_lidar(self, bev_feats, index=None):#imgs is BEV point feature
        bev_feats = self.bev_conv1(bev_feats) 
        bev_feats = self.add_angle_embedding_optimized(bev_feats.float().permute(0,2,3,1))
        bev_feats = self.bev_proj_norm(self.bev_proj(bev_feats))

        bev_feats = bev_feats.float().reshape(bev_feats.shape[0], -1, bev_feats.shape[-1])
        
        bev_atts = torch.ones(bev_feats.size()[:-1], dtype=torch.long).to(bev_feats.device)
    
        query_tokens = self.query_tokens.expand(bev_atts.shape[0], -1, -1)
        
        condition = (index != -1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)#([B, 1, 1, 1, 1])

        selected_angle_embd = torch.where(condition, self.angle_pos_embd[index], #([B, 1, 1, 1, 768])
                                  torch.mean(self.angle_pos_embd, dim=0, keepdim=True))
        
        selected_angle_embd = selected_angle_embd.squeeze(1).squeeze(1).expand_as(query_tokens)#[B, 32, 768]

        query_tokens = query_tokens + selected_angle_embd

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=bev_feats,
            encoder_attention_mask=bev_atts,
            return_dict=True,
        )
        bev_feat_former = torch.cat([torch.mean(query_output.last_hidden_state, dim=1, keepdim = True) , query_output.last_hidden_state], dim=1)
        bev_feat_former = F.normalize(
            self.vision_proj(bev_feat_former), dim=-1
        )

        return bev_feat_former

    def find_angle(self, angle, angle_ranges):
        mask = torch.zeros_like(angle, dtype=torch.bool).to(angle.device)
        for range in angle_ranges:
            mask = mask | (angle >= range[0]) & (angle <= range[1])
        return mask

    def generate_angles(self, C=768, H=60,W=60):
        x_coords = torch.arange(-W // 2, W // 2)
        y_coords = torch.arange(-H // 2, H // 2)
        xx, yy = torch.meshgrid(x_coords, y_coords)
        angles = torch.atan2(xx, yy) * 180 / 3.14159265
        angles = (angles + 360) % 360

        view_masks = torch.zeros((6,H,W,C), dtype=torch.bool)
        for i in range(len(self.VIEW_RANGE)):
            range_ = self.VIEW_RANGE[i]
            mask = self.find_angle(angles, range_)
            mask = mask.unsqueeze(-1).repeat(1,1,C)
            view_masks[i] = mask
        return view_masks
    
    def add_angle_embedding_optimized(self, bev_feat):
        B, H, W, C = bev_feat.shape#self.view_masks.shape: torch.Size([6, 60, 60, 768]) 6 is the view
        view_masks_expanded = self.view_masks.unsqueeze(1).repeat(1, B, 1, 1, 1).to(bev_feat.device)#[6, B, 180, 180, 768]
        angle_pos_embd_expanded = self.angle_pos_embd.expand(-1, B, H, W, C) #([6, B, 60, 60, 768])
        pos_embd_aggregated = view_masks_expanded * angle_pos_embd_expanded 
        bev_feat = bev_feat + pos_embd_aggregated.sum(dim=0)
        return bev_feat #[B, 180, 180, 768]


    def forward(self, tokens, labels, imgs, index):# img is the feature of the bev
        lidar_proj = self.forward_lidar(imgs, index)#([B, 577, 4096])
        del imgs

        _bsz, seqlen = tokens.shape

        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers:
            h = layer(h, 0, freqs_cis, mask, lidar_proj + adapter[adapter_index])
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            assert self.llama.vocab_size == 32000
            c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())

        return c_loss, c_loss

    def forward_inference(self, lidar_proj, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)


        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0

        for layer in self.llama.layers:
            h = layer(h, start_pos, freqs_cis, mask, lidar_proj + adapter[adapter_index].repeat(_bsz, 1, 1))
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float()

    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config, mirror='https://hf-mirror.com'
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        return Qformer, query_tokens
    
    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)


        return msg
    
    def generate(
        self, imgs, prompts,query,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,
    ):
        bsz = len(imgs)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        assert len(imgs) == len(prompts)

        with torch.cuda.amp.autocast():
            visual_query = self.forward_lidar(imgs, query)

        if isinstance(prompts[0], str):
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).cuda().long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                logits = self.forward_inference(visual_query, tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded


_MODELS = {
    "BIAS-7B": "https://github.com/ZrrSkywalker/LLaMA-Adapter/releases/download/v.2.0.0/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth",
    # "LORA16-7B": "",
    # "PARTIAL-7B": ""
}

def available_models():
    return list(_MODELS.keys())

def load(name, llama_dir, device="cuda" if torch.cuda.is_available() else "cpu", download_root='ckpts', max_seq_len=512,
        phase="finetune"):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root)
    elif os.path.isfile(name):
        model_path = name
    else:
        return RuntimeError(f"Model {name} not found; available models = {available_models()}")

    ckpt = torch.load(model_path, map_location='cpu')
    
    llama_type = name.split('.')[0].split('-')[-1]
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')

    print(f'Loading LLaMA-Adapter from {model_path}')
    ckpt = torch.load(model_path, map_location='cpu')

    model = LLaMA_adapter(
        llama_ckpt_dir, llama_tokenzier_path,
        max_seq_len=max_seq_len, max_batch_size=1,
        clip_model='ViT-L/14@336px',
        v_embed_dim=1024, v_depth=16,
        v_num_heads=16, v_mlp_ratio=4.0,
        query_len=577, query_layer=32,
        phase=phase)

    load_result = model.load_state_dict(ckpt['model'], strict=False)

    assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    return model.to(device), model.clip_transform