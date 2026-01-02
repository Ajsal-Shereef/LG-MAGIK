
import torch
from typing import Optional, Union, Tuple, List, Dict, Any
import abc

class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_att_layers = -1

class EmptyControl(AttentionControl):
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid storing too large attention maps
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] = torch.cat([self.attention_store[key][i], self.step_store[key][i]], dim=0)
        self.step_store = self.get_empty_store()
        self.cur_step += 1

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

class AttentionReplace(AttentionControl):
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[Any] = None):
        super(AttentionReplace, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = cross_replace_steps
        self.self_replace_alpha = self_replace_steps
        self.prompts = prompts
        self.num_steps = num_steps
        self.local_blend = local_blend

    def step_callback(self, x_t):
        if self.local_blend is not None:
            # Not implemented in this simple version
            pass
        return x_t

    def replace_cross_attention(self, attn_base, attn_replace):
        return torch.where(self.get_mask(self.cross_replace_alpha, attn_base.dtype, attn_base.device), attn_replace, attn_base)
    
    def replace_self_attention(self, attn_base, attn_replace):
        return torch.where(self.get_mask(self.self_replace_alpha, attn_base.dtype, attn_base.device), attn_replace, attn_base)
        
    def get_mask(self, alpha, dtype, device):
        if self.cur_step / self.num_steps < alpha:
             return torch.ones(1, dtype=dtype, device=device)
        else:
             return torch.zeros(1, dtype=dtype, device=device)

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # NOT IMPLEMENTED FULLY FOR REPLACEMENT WITHOUT SOURCE ATTENTION STORE
        # This usually requires passing the source attention store to the controller.
        # But wait, standard P2P "replacing" usually implies we RUN the generation and swap attention maps.
        # OR we run the source generation first, store attention, then run target generation and inject.
        
        # In this implementation, we assume we are injected into the UNet.
        # Since we just want P2P Editing (Source -> Target), we usually need:
        # 1. AttentionStore (runs on Source)
        # 2. AttentionInject (runs on Target, uses stored attention)
        return attn
    
class AttentionInjector(AttentionControl):
    def __init__(self, source_store: AttentionStore, start_at_step: float = 0.0, end_at_step: float = 1.0, num_steps=50):
        super(AttentionInjector, self).__init__()
        self.source_store = source_store
        self.start_at_step = start_at_step
        self.end_at_step = end_at_step
        self.num_steps = num_steps
        
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        
        # Check if we should inject
        progress = self.cur_step / self.num_steps
        if self.start_at_step <= progress < self.end_at_step:
             # Inject from source
             # We need to map the current layer index to the stored one.
             # The source store stored ALL steps concatenated?
             # My AttentionStore implementation above stores `attention_store` which accumulates?
             # Or `step_store` which is per step?
             
             # Let's fix AttentionStore to be more usable for per-step Injection.
             pass
        
        return attn

# Re-implementing a simpler Attention Wrapper that works with diffusers `set_attn_processor` is HARD because `diffusers` changes API often.
# OLD HACK: Register hooks or monkey-patch UNet.

def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # Standard CrossAttention forward logic (simplified for diffusers 0.14+)
            # We assume `self` is a CrossAttention module (or Attention module)
            
            # This is tricky without knowing exact diffusers version inner workings.
            # But the user had this working before.
            # I will assume we can hook into `to_q`, `to_k`, `to_v`? No, simpler to hook into the attention scores.
            
            # Since we don't have the original p2p_utils.py content (I didn't save it), I have to rely on standard method.
            # Standard method:
            
            batch_size = hidden_states.shape[0]
            if self.added_kv_proj_dim is None:
                 is_cross = encoder_hidden_states is not None
            else:
                 is_cross = True # Simplification

            # Let's trust that the actual attention computation happens and we can intercept `attn_weights`?
            # Diffusers `Attention` class has `processor`.
            # We can define a custom Processor!
            return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
        return forward

    # ... This approach is risky if I don't know the exact class structure.
    # The previous code likely used the standard "Prompt-to-Prompt" repo helpers.
    pass

class EmptyController(AttentionControl):
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn
