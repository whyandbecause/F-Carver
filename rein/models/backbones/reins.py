from mmseg.models.builder import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor
from scipy import stats


@MODELS.register_module()
class Reins(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims//8)
        self.mlp_delta_f = nn.Linear(self.embed_dims//8, self.embed_dims)
        
        self.mlp_featd = nn.Linear(self.embed_dims, self.embed_dims//8)
        self.mlp_featu = nn.Linear(self.embed_dims//8, self.embed_dims)
        
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        nn.init.uniform_(self.learnable_tokens.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)
        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)

    def return_auto(self, feats):
        if self.link_token_to_query:
        
            tokens = self.transform(self.get_tokens(-1)).permute(1, 2, 0)
            tokens = torch.cat(
                [
                    F.max_pool1d(tokens, kernel_size=self.num_layers),
                    F.avg_pool1d(tokens, kernel_size=self.num_layers),
                    tokens[:, :, -1].unsqueeze(-1),
                ],
                dim=-1,
            )
            querys = self.merge(tokens.flatten(-2, -1))
            #print('------querys.shape',querys.shape)
            return feats, querys#(num_query, dim=256)
        else:
            return feats

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True, TT=1.0
    ) -> Tensor:
        
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
        tokens = self.get_tokens(layer)
        #print(tokens.shape)
        delta_feat = self.forward_delta_feat(
            feats,
            tokens,
            layer,
            TT,
        )
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int, TT: float) -> Tensor:
        m = self.bernoulli_concrete_sample(self.logits / 0.5,
                                               TT, shape=feats.shape)
        #print( '--------',0.01)
        #m = m.unsqueeze(1)
        
        sel_feats = feats * m
        
        #prepare for patch selection 
        t_feats = self.mlp_featu(F.gelu(self.mlp_featd(feats.permute(1, 0, 2))))#bnc  remap the ori feature
        avg_tf = F.adaptive_avg_pool1d(t_feats,1).squeeze(2)#bn  #avg pool on embeded dimention
        noise = torch.rand(avg_tf.shape).to(feats)#bn for u distribution to add the selected probability for possible patch
        weights = torch.softmax(avg_tf, dim=1)#bn cal the selected weights to choose which patch is needed
        quantitle_scale = torch.quantile(weights, torch.tensor([0.4]).to(feats), dim=1, keepdim=True)
        vv = [i for i in range(weights.shape[0])]
        index= (weights[vv,:]>quantitle_scale[:,vv]).squeeze(0)
        noise[index]+=1
        mask = torch.ones(t_feats.shape).to(feats)
        #get most sure and possible patch(which is contains the most objects to seg and contains the part of objects to seg) 
        _, index = noise.topk(int(noise.shape[1]*0.4), dim=1, largest=False, sorted=False)  #only top k is needed
        source_tensor, index_tensor = torch.chunk(mask, mask.shape[0], dim=0), torch.chunk(index, index.shape[0], dim=0)
        patch_mask = list(map(self.for_value, source_tensor, index_tensor))#other patch is masked
        patch_mask = torch.cat(patch_mask, dim=0).permute(1, 0, 2)
        patch_feats = t_feats.permute(1, 0, 2)*patch_mask
        
        '''
        #prepare for patch selection 
        t_feats = feats.permute(1, 0, 2)#bnc  remap the ori feature
        avg_tf = F.adaptive_avg_pool1d(t_feats,1).squeeze(2)#bn  #avg pool on embeded dimention
        noise = torch.rand(avg_tf.shape).to(feats)#bn for u distribution to add the selected probability for possible patch
        weights = torch.softmax(avg_tf, dim=1)#bn cal the selected weights to choose which patch is needed
        quantitle_scale = torch.quantile(weights, torch.tensor([0.3]).to(feats), dim=1, keepdim=True)
        vv = [i for i in range(weights.shape[0])]
        index= (weights[vv,:]>quantitle_scale[:,vv]).squeeze(0)
        noise[index]+=1
        mask = torch.ones(t_feats.shape).to(feats)
        #get most sure and possible patch(which is contains the most objects to seg and contains the part of objects to seg) 
        _, index = noise.topk(int(noise.shape[1]*0.3), dim=1, largest=False, sorted=False)  #only top k is needed
        source_tensor, index_tensor = torch.chunk(mask, mask.shape[0], dim=0), torch.chunk(index, index.shape[0], dim=0)
        patch_mask = list(map(self.for_value, source_tensor, index_tensor))#other patch is masked
        patch_mask = torch.cat(patch_mask, dim=0).permute(1, 0, 2)
        patch_feats = t_feats.permute(1, 0, 2)*patch_mask
        '''
        
        delta_f = self.mlp_token2feat(sel_feats + patch_feats + feats)
        delta_f = F.gelu(delta_f)
        delta_f = self.mlp_delta_f(delta_f)
        #print('gogogogogogogo')
        return delta_f#delta_f
    
    def for_value(self, matrix,index):
  
      return matrix.index_fill_(1,index[0,:], 0.0)
  


    def clamp_probs(self,probs):
        eps = torch.finfo(probs.dtype).eps
        return torch.clamp(probs, min=eps, max=1-eps)
    
    def bernoulli_concrete_sample(self, logits, temperature, shape=torch.Size([])):
        '''
        Sampling for BinConcrete distribution.
    
        See PyTorch source code, differs from Eq. 16 of Maddison et al., 2017.
        '''
        uniform_shape = torch.Size(shape) 
        #print(uniform_shape,logits.shape)
        u = self.clamp_probs(torch.rand(uniform_shape, dtype=torch.float32,
                                   device=logits.device))
                                   
        #return torch.sigmoid((torch.log(0.5 * (torch.tanh(logits) + 1)) - torch.log(0.5 * (torch.tanh(1-logits) + 1))
        #                      + torch.log(u) - torch.log(1 - u)) / temperature) 
                                                        
        return torch.sigmoid((F.logsigmoid(logits) - F.logsigmoid(-logits)
                              + torch.log(u) - torch.log(1 - u)) / temperature)
        
        #return torch.sigmoid((torch.log(torch.sigmoid(torch.log(1 + torch.exp(logits)))) - torch.log(torch.sigmoid(torch.log(1 + torch.exp(1-logits))))+ torch.log(u) - torch.log(1 - u)) / temperature)
        
       # p = torch.tensor(stats.norm.cdf(self.probit_p.item()))
       #p = torch.sigmoid(torch.log(1 + torch.exp(self.param)))   
       #p = 0.5 * (torch.tanh(self.param) + 1) 
       # p = torch.sigmoid(self.logit_p) 
                      
@MODELS.register_module()
class LoRAReins(Reins):
    def __init__(self, lora_dim=16, **kwargs):
        self.lora_dim = lora_dim
        super().__init__(**kwargs)

    def create_model(self):
        super().create_model()
        del self.learnable_tokens
        self.learnable_tokens_a = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.lora_dim])
        )
        self.learnable_tokens_b = nn.Parameter(
            torch.empty([self.num_layers, self.lora_dim, self.embed_dims])
        )

        init_logit = - torch.log(1 / torch.tensor(0.99) - 1) * 0.5
        self.logits = nn.Parameter(torch.full(
            (self.embed_dims, 1, self.embed_dims), init_logit, dtype=torch.float32, requires_grad=True))

        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + (self.embed_dims * self.lora_dim) ** 0.5
            )
        )
        nn.init.uniform_(self.learnable_tokens_a.data, -val, val)
        nn.init.uniform_(self.learnable_tokens_b.data, -val, val)

    def get_tokens(self, layer):
        if layer == -1:
            return self.learnable_tokens_a @ self.learnable_tokens_b
        else:
            return self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]
