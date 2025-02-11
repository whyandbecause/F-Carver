from mmseg.models.builder import BACKBONES, MODELS
from .reins import Reins
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train
import numpy as np

@BACKBONES.register_module()
class ReinsDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        reins_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reins: Reins = MODELS.build(reins_config)
        self.temperature = 1.0##############
        self.iter_t = 0
        '''
        y_initial = 5.0
        decay_factor = np.power(0.00028 / 5.0, 1.0 / 5000)
        # Arrays to hold x and y values
        x_values = np.arange(0, 10000)
        self.y_values = np.empty_like(x_values, dtype=np.float64)
        # Calculate y for 0 <= x <= 5000
        y = y_initial
        for i in range(5000):
            self.y_values[i] = y
            y *= decay_factor
        
        # Calculate symmetric y for 5000 < x <= 10000
        for i in range(5000, 10000):
            self.y_values[i] = self.y_values[10000 - i-1]
        '''  
         
    def forward_features(self, x, masks=None):
        
        ################AnnealedTempreture-1-001####################
        
        if self.iter_t%10000==0:#cycle change the tempreture
          self.temperature=1.0
        self.iter_t += 1
        self.temperature = self.temperature * np.power(0.0009 / self.temperature,1.0 / 10000)#AnnealedTempreture-1-001
        ##self.temperature = self.temperature * np.power(0.00028 / self.temperature,1.0 / 10000)
        ########################################
        
        
        ########for low2fast, fast2low###############################
        '''
        zzz = self.iter_t%10000
        if zzz<5000:
          self.temperature = 0.01 + (5.01 - 0.01) * (1 - np.exp(-0.00125 * (5000 - zzz)))
        else:     
          self.temperature = 0.01 + (5.01 - 0.01) * (1 - np.exp(-0.00125 * (zzz - 5000)))
        self.iter_t += 1
        '''
        '''
        ##########for fast2low, low2fast###########
        self.temperature = self.y_values[self.iter_t%10000]
        #print(self.temperature)
        self.iter_t+=1
        '''    
        
        #print('----',self.iter_t, self.temperature)
        #print(x.shape, masks)
        B, _, h, w = x.shape
        #print(h,w)
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            #print(x.shape)
            x = self.reins.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
                TT = self.temperature,############
            )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return self.reins.return_auto(outs)

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["reins"])
        set_train(self, ["reins"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "rein" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state
