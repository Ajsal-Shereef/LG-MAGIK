import math
import torch
from torch.distributions import Categorical, Normal
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from architectures.common_utils import identity, get_activation, get_normalisation_1d
from architectures.math_utils import TanhNormal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_layer_uniform(layer, init_w=3e-3, init_b=0.1):
    """
    Init uniform parameters on the single layer
    layer: nn.Linear
    init_w: float = 3e-3
    :return: nn.Linear
    """
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_b, init_b)

    return layer

def init_layer_orthogonal(layer, gain=torch.sqrt(torch.tensor(2.0))):
    """
    Initialize a linear layer with orthogonal initialization.
    
    Args:
        layer (nn.Linear): The layer to initialize.
        gain (float): Scaling factor (default=1.0, use sqrt(2) for ReLU).
    
    Returns:
        nn.Linear: The layer with orthogonal weights.
    """
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)  # Orthogonal init
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)  # Initialize bias to zero
    return layer



class MLP(nn.Module):
    """
    Baseline of Multilayer perceptron. The layer-norm is not implemented here
    Attributes:
        input_size (int): size of input
        output_size (int): size of output layer
        hidden_sizes (list): sizes of hidden layers
        hidden_activation (function): activation function of hidden layers
        output_activation (function): activation function of output layer
        hidden_layers (list): list containing linear layers
        use_output_layer (bool): whether or not to use the last layer
        n_category (int): category number (-1 if the action is continuous)
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        hidden_activation="relu",
        output_activation="identity",
        linear_layer=nn.Linear,
        norm = "bn",
        use_output_layer=True,
        n_category=-1,
        init_fn=init_layer_orthogonal,
        dropout_prob = 0,
        stabilise_factor = 0
    ):
        """
        Initialize.
        Args:
            input_size (int): size of input
            output_size (int): size of output layer
            hidden_sizes (list): number of hidden layers
            hidden_activation (function): activation function of hidden layers
            output_activation (function): activation function of output layer
            linear_layer (nn.Module): linear layer of mlp
            use_output_layer (bool): whether or not to use the last layer
            n_category (int): category number (-1 if the action is continuous)
            init_fn (Callable): weight initialization function bound for the last layer
        """
        super(MLP, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.output_activation = get_activation(output_activation)
        self.linear_layer = linear_layer
        self.use_output_layer = use_output_layer
        self.n_category = n_category
        self.stabilise_factor = stabilise_factor
        #self.drop_out = torch.nn.Dropout(p=0.5, inplace=False)

        in_size = self.input_size
        self.hidden_layers = nn.Sequential()
        for i, next_size in enumerate(hidden_sizes):
            fc = Linear(in_size, next_size, post_activation = hidden_activation, dropout_prob = dropout_prob, norm = norm)
            in_size = next_size
            self.hidden_layers.add_module("hidden_fc_{}".format(i),fc)

        # set output layers
        if self.use_output_layer:
            self.output_layer = self.linear_layer(in_size, output_size)
            self.output_layer = init_fn(self.output_layer)
        else:
            self.output_layer = identity
            self.output_activation = identity

    def forward(self, x):
        """
        Forward method implementation.
        x: torch.Tensor
        :return: torch.Tensor
        """
        # for hidden_layer in self.hidden_layers:
        #     x = self.hidden_activation(hidden_layer(x))
            #x = self.drop_out(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = self.output_activation(x) + self.stabilise_factor
        return x
    
class MLPResBlock(nn.Module):
    """
    An MLP-based residual block with a skip connection.
    """
    def __init__(self, dim, hidden_dim_ratio, norm, activ, dropout_prob=0):
        super(MLPResBlock, self).__init__()
        hidden_dim = int(dim * hidden_dim_ratio)
        
        self.mlp_block = nn.Sequential(
            Linear(dim, hidden_dim, post_activation=activ, norm=norm, dropout_prob=dropout_prob),
            Linear(hidden_dim, dim, post_activation="identity", norm=norm, dropout_prob=dropout_prob)
        )

    def forward(self, x):
        """
        Apply the MLP block and add the skip connection.
        x_residual = self.mlp_block(x)
        """
        return x + self.mlp_block(x)
    
class MLPEncoder(nn.Module):
    """
    An MLP-based encoder that uses residual blocks.
    """
    def __init__(self, input_dim, hidden_dims, out_dim, num_res_blocks, norm, activ, dropout_prob=0):
        super(MLPEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        #MLP layers
        self.blocks.append(MLP(input_size=input_dim,
                               output_size=out_dim,
                               hidden_sizes=hidden_dims,
                               hidden_activation=activ,
                               norm = norm,
                               dropout_prob=dropout_prob))
        #Resblocks
        for _ in range(num_res_blocks):
            self.blocks.append(MLPResBlock(out_dim, 2, norm, activ))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def get_all_features(self, x):
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features
    
class TransformerCrossAttention(nn.Module):
    def __init__(self, input_dim, text_dim, nhead=8):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj   = nn.Linear(text_dim, input_dim)
        self.value_proj = nn.Linear(text_dim, input_dim)
        self.out_proj   = nn.Linear(input_dim, input_dim)
        self.n_heads = nhead
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Initialize weights as per the original Transformer paper
        nn.init.xavier_uniform_(self.query_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.key_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.value_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.query_proj.bias is not None:
            nn.init.constant_(self.query_proj.bias, 0)
            nn.init.constant_(self.key_proj.bias, 0)
            nn.init.constant_(self.value_proj.bias, 0)
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, feat, text_feat, attention_mask=None):
        """
        feat: [B, F]
        text_feat: [B, T, D]
        attention_mask: [B, T] (1 = keep, 0 = mask)
        """
        B, F = feat.shape
        H = self.n_heads
        d = F // H

        q = self.query_proj(feat).view(B, 1, H, d).transpose(1, 2)  # [B,H,N,d]
        k = self.key_proj(text_feat).view(B, -1, H, d).transpose(1, 2)  # [B,H,T,d]
        v = self.value_proj(text_feat).view(B, -1, H, d).transpose(1, 2)  # [B,H,T,d]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / (d ** 0.5)  # [B,H,N,T]

        if attention_mask is not None:
            # expand mask: [B,T] -> [B,1,1,T]
            mask = attention_mask[:, None, None, :].to(attn.dtype)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)

        out = attn @ v  # [B,H,N,d]
        out = out.transpose(1, 2).contiguous().view(B, F)
        return self.out_proj(out)
    
# class TransformerCrossAttention(nn.Module):
#     """
#     A transformer-based attention module to fuse a feature vector with text embeddings.

#     This module takes a batch of feature vectors and a batch of text embeddings and
#     uses cross-attention to refine the feature vector based on the information
#     in the text. This is a common pattern in multimodal architectures.

#     Args:
#         feature_dim (int): The dimensionality of the input feature vector (F).
#         embed_dim (int): The dimensionality of the text embeddings and the model's hidden state.
#         nhead (int): The number of attention heads in the MultiheadAttention model.
#         num_layers (int): The number of transformer decoder layers to stack.
#         dim_feedforward (int): The dimension of the feedforward network model in the decoder layer.
#         dropout (float): The dropout value.
#     """
#     def __init__(self, feature_dim: int, embed_dim: int = 512, nhead: int = 8, num_layers: int = 1, dim_feedforward: int = 2048, dropout: float = 0.1):
#         super().__init__()

#         self.feature_dim = feature_dim
#         self.embed_dim = embed_dim

#         # 1. Input projection layer
#         # This linear layer projects the input feature vector from its original
#         # dimension (F) to the model's embedding dimension (512), so it can be
#         # processed by the transformer decoder.
#         self.text_proj = nn.Linear(embed_dim, feature_dim)

#         # 2. Transformer Decoder Layer
#         # We use a standard TransformerDecoderLayer, which is perfect for this
#         # use case. It's designed for cross-attention between a 'target' sequence
#         # and a 'memory' sequence.
#         decoder_layer = nn.TransformerDecoderLayer(
#             d_model=feature_dim,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True  # Important: ensures input/output tensors are (batch, seq, feature)
#         )
        
#         # 3. Transformer Decoder
#         # This stacks multiple decoder layers. For many tasks, one layer is sufficient.
#         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

#     def forward(self, feature: torch.Tensor, text_embedding: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass for the attention module.

#         Args:
#             feature (torch.Tensor): The input feature tensor of shape [B, F].
#             text_embedding (torch.Tensor): The text embedding tensor of shape [B, 77, 512].

#         Returns:
#             torch.Tensor: The transformed feature tensor of shape [B, 512].
#         """
#         # --- Input Validation ---
#         if feature.dim() != 2:
#             raise ValueError(f"Expected feature to have 2 dimensions [B, F], but got {feature.dim()}")
#         if text_embedding.dim() != 3:
#             raise ValueError(f"Expected text_embedding to have 3 dimensions [B, 77, 512], but got {text_embedding.dim()}")
#         if text_embedding.shape[2] != self.embed_dim:
#              raise ValueError(f"Text embedding dimension ({text_embedding.shape[2]}) does not match model embed_dim ({self.embed_dim})")


#         # --- Cross-Attention Steps ---

#         # 1. Project the feature to the embedding dimension.
#         # Shape: [B, T, F] -> [B, T, feature_dim]
#         projected_text = self.text_proj(text_embedding)

#         # 2. Add a sequence dimension to the projected feature.
#         # The transformer decoder expects a sequence. We treat our single feature vector
#         # as a sequence of length 1. This will be the 'query' in the attention mechanism.
#         # Shape: [B, feature_dim] -> [B, 1, feature_dim]
#         tgt = feature.unsqueeze(1)

#         # 3. The text embedding serves as the 'memory' (key and value) for the attention.
#         # The model will "attend to" the text to gather relevant information.
#         # Shape: [B, 77, feature_dim]
#         memory = projected_text

#         # 4. Apply the transformer decoder.
#         # The `tgt` (our feature) attends to the `memory` (the text).
#         # The output will have the same shape as the `tgt`.
#         # Shape: [B, 1, feature_dim] -> [B, 1, feature_dim]
#         output = self.transformer_decoder(tgt=tgt, memory=memory, memory_key_padding_mask=text_mask)

#         # 5. Remove the sequence dimension to get the final transformed feature.
#         # Shape: [B, 1, feature_dim] -> [B, feature_dim]
#         transformed_feature = output.squeeze(1)

#         return transformed_feature
    
class FiLM(nn.Module):
    def __init__(self, num_features, cond_dim):
        """
        Args:
            num_features (int): Number of feature channels to condition.
            cond_dim (int): Dimension of the condition vector.
        """
        super(FiLM, self).__init__()
        # Learn a scaling (gamma) and shifting (beta) for each feature channel.
        self.gamma = nn.Linear(cond_dim, num_features)
        self.beta  = nn.Linear(cond_dim, num_features)

    def forward(self, cond):
        """
        Args:
            x (Tensor): Feature map of shape (B, F).
            cond (Tensor): Condition vector of shape (B, cond_dim).
        Returns:
            Tensor: FiLM-modulated feature map.
        """
        # Compute scale and shift parameters and unsqueeze them to (B, C, 1, 1)
        gamma = self.gamma(cond)
        beta  = self.beta(cond)
        return gamma, beta
    
class CrossAttentionFiLM(nn.Module):
    """
    FiLM + Residual Cross-Attention block with learnable gate,
    adapted for feature latent code z.
    """
    def __init__(self, input_dim, latent_dim, text_dim, norm="ln"):
        super().__init__()


        # FiLM: gamma, beta from latent
        self.film = FiLM(input_dim, latent_dim)

        # Local conv processing
        self.linear = Linear(input_dim, input_dim, norm=norm)

        # Cross-attention
        self.cross = TransformerCrossAttention(input_dim, text_dim)
        self.cross_norm = get_normalisation_1d(norm, input_dim)

        # Activation
        self.act = nn.GELU()

    def forward(self, x, z, text_feat, attention):

        # --- FiLM modulation ---
        gamma, beta = self.film(z)

        out = self.linear(x)

        out = out * (1 + gamma) + beta

        # --- Cross-attention residual ---
        attn_out = self.cross(out, text_feat, attention)  # (B, F)

        #residual connection
        out = out + self.cross_norm(attn_out)

        return self.act(out)
    
class FinalTextConditionedOutput(nn.Module):
    def __init__(self, input_dim, output_dim, text_dim, n_heads=8):
        super().__init__()
        self.cross_attention = TransformerCrossAttention(input_dim, text_dim, n_heads)

        # stronger head: residual conv stack
        # self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.linear = MLP(input_dim, output_dim, [input_dim//2], "gelu")

    def forward(self, x, text_feat, attention_mask):
        """
        x: [B, F] - feature map
        text_feat: [B, T, D] - Text feature (from the text encoder)
        """

        # cross-attn
        refined = self.cross_attention(x, text_feat, attention_mask)

        # Pass the refined features through the final convolution
        return self.linear(refined)
    
class MLPTextConditionedDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, decoder_hidden_dims, clip_model):
        super(MLPTextConditionedDecoder, self).__init__()

        #Conv layer to map latent channel to dim
        self.mapping_linear = Linear(latent_dim, decoder_hidden_dims[0])
        #Text encoder and tockenizer
        self.tokenizer=CLIPTokenizer.from_pretrained(clip_model)
        self.text_encoder=CLIPTextModel.from_pretrained(clip_model)
        #Freeze the CLIP model parameters
        for params in self.text_encoder.parameters():
            params.requires_grad = False
        self.text_encoder.eval()
        self.text_dim=self.text_encoder.config.hidden_size
        self.text_adapter = MLP(self.text_dim, self.text_dim, [64, 256], hidden_activation = 'gelu', norm='ln')
        self.attention = TransformerCrossAttention(latent_dim, self.text_dim, nhead=2)
        # self.text_to_latent = nn.Linear(self.text_dim, latent_dim)
        
        self.attention_film_blocks=nn.ModuleList()
        self.linear_blocks = nn.ModuleList()
        start_dim = decoder_hidden_dims[0]
        for hidden_dim in decoder_hidden_dims:
            self.attention_film_blocks.append(CrossAttentionFiLM(start_dim, latent_dim, self.text_dim))
            self.linear_blocks.append(Linear(start_dim, hidden_dim, "gelu"))
            start_dim = hidden_dim
            
        self.final = FinalTextConditionedOutput(decoder_hidden_dims[-1], output_dim, self.text_dim)
        
    def forward(self,z,text_tockens, attention_mask):
        self.text_feats=self.text_encoder(text_tockens, return_dict=False)[0] # (B,T,D)
        self.text_feats = self.text_adapter(self.text_feats)
        # Expand text_feat to spatial (broadcast over H_z, W_z)
        # text_global = self.text_feats.mean(dim=1)  # [B, D]
        # text_proj = self.text_to_latent(text_global)  # [B, F]
        # z = torch.cat([z, text_proj], dim=1)  # concat along channel

        z = self.attention(z, self.text_feats, attention_mask)
        x = self.mapping_linear(z)
        for blk,linear in zip(self.attention_film_blocks, self.linear_blocks):
            x=blk(x,z,self.text_feats, attention_mask)
            x=linear(x)
        return self.final(x, self.text_feats, attention_mask)
    
class Embed(nn.Module):
    def __init__(
        self,
        embed_dim
        ):
        super(Embed, self).__init__()

        self.embed_layer = nn.Embedding(5, embed_dim)
        nn.init.xavier_uniform_(self.embed_layer.weight)

    def forward(self, x):
        """
        Forward method implementation.
        x: torch.Tensor
        :return: torch.Tensor
        """
        x_int = x.type(torch.int64)
        x = self.embed_layer(x_int)
        return x
    
class Linear(nn.Module):
    def __init__(
        self,
        in_dim, 
        out_dim,
        post_activation = "identity",
        dropout_prob = 0,
        norm = "bn",
        ):
        super(Linear, self).__init__()

        self.linear_layer = nn.Linear(in_dim, out_dim)
        self.batch_norm = get_normalisation_1d(norm, out_dim)
        nn.init.orthogonal_(self.linear_layer.weight)
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.post_activation = get_activation(post_activation)

    def forward(self, x):
        """
        Forward method implementation.
        x: torch.Tensor
        :return: torch.Tensor
        """
        x = self.linear_layer(x)
        x = self.batch_norm(x)
        x = self.dropout_layer(x)
        x = self.post_activation(x)
        return x    


class FlattenMLP(MLP):
    """
    Baseline of Multi-layered perceptron for Flatten input.
    """

    def forward(self, *args):
        """
        Forward method implementation.
        states is assume to be Tensor containing list of flatten states
        """
        states, actions = args

        if len(states.size()) == 1:
            states = states.unsqueeze(0)
        if len(actions.size()) == 1:
            actions = actions.unsqueeze(0)
        flat_inputs = torch.cat((states, actions), dim=-1)
        return super(FlattenMLP, self).forward(flat_inputs)


class GaussianDist(MLP):
    """
    Multilayer perceptron with Gaussian distribution output.
    the Mean

    Attributes:
        mean_activation (function): bounding function for mean
        log_std_min (float): lower bound of log std
        log_std_max (float): upper bound of log std
        mean_layer (nn.Linear): output layer for mean
        log_std_layer (nn.Linear): output layer for log std
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        hidden_activation=F.relu,
        mean_activation=torch.tanh,
        std_activation=torch.tanh,
        log_std_min=-20,
        log_std_max=2,
        init_fn=init_layer_uniform,
        std=None
    ):
        """
        Initialize
        If std is not None, then use fixed std value given by argument std, otherwise use std layer
        """
        super(GaussianDist, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            use_output_layer=False,
        )
        self.std_activation = std_activation
        self.mean_activation = mean_activation
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        in_size = hidden_sizes[-1]  # std layer is the last layer

        # set log_std layer
        self.std = std
        self.log_std = None
        if std is None:
            self.log_std_layer = nn.Linear(in_size, output_size)
            self.log_std_layer = init_fn(self.log_std_layer)
        else:
            self.log_std = np.log(std)
            assert log_std_min <= self.log_std <= log_std_max

        # set mean layer
        self.mean_layer = nn.Linear(in_size, output_size)
        self.mean_layer = init_fn(self.mean_layer)

    def get_dist_params(self, x):
        """
        Return gaussian distribution parameters.
        x (torch.Tensor)
        :return: mu, log_std, std
        """
        hidden = super(GaussianDist, self).forward(x)

        # get mean
        mu = self.mean_activation(self.mean_layer(hidden))

        # get std
        if self.std is None:
            uncentered_log_std = self.std_activation(self.log_std_layer(hidden))
            log_std = uncentered_log_std.clamp(min=self.log_std_min, max=self.log_std_max)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        return mu, log_std, std

    def forward(self, x):
        """
        Forward method implementation.
        x (torch.Tensor)
        :return: action (torch.Tensor) and dist
        """
        mu, _, std = self.get_dist_params(x)

        # get normal distribution and action
        dist = Normal(mu, std)
        action = dist.sample()

        return action, dist


class TanhGaussianDistParams(GaussianDist):
    """
    Multilayer perceptron with Gaussian distribution output.
    """

    def __init__(self, **kwargs):
        """Initialize."""
        super(TanhGaussianDistParams, self).__init__(**kwargs, mean_activation=identity)

    def forward(self, x, epsilon=1e-6, deterministic=False, reparameterize=True):
        """
        Forward method implementation.
        x: torch.Tensor
        epsilon: float = 1e-6
        deterministic: bool = False, if deterministic is True, action = tanh(mean).
        :return: Tuple[torch.Tensor, ...]
        """
        mean, _, std = super(TanhGaussianDistParams, self).get_dist_params(x)

        # sampling actions
        if deterministic:
            action = torch.tanh(mean)
            return action, None, None, mean, std
        else:
            tanh_normal = TanhNormal(mean, std, epsilon=epsilon)
            if reparameterize:
                action, z = tanh_normal.rsample()
                log_prob = tanh_normal.log_prob(value=action, pre_tanh_value=z)
            else:
                action, z = tanh_normal.sample()
                log_prob = tanh_normal.log_prob(value=action, pre_tanh_value=z)

            return action, log_prob, z, mean, std


class CategoricalDist(MLP):
    """
    Multilayer perceptron with categorical distribution output (for discrete domains)
    Attributes:
        last_layer (nn.Linear): output layer for softmax
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes,
        hidden_activation=F.relu,
        init_fn=init_layer_uniform,
    ):
        """Initialize."""
        super(CategoricalDist, self).__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
            use_output_layer=False,
        )

        in_size = hidden_sizes[-1]

        self.last_layer = nn.Linear(in_size, output_size)
        self.last_layer = init_fn(self.last_layer)

    def forward(self, x):
        """ Forward method implementation."""
        hidden = super(CategoricalDist, self).forward(x)
        action_probs = F.softmax(self.last_layer(hidden), dim=-1)

        dist = Categorical(action_probs)
        selected_action = dist.sample()
        selected_action = selected_action.unsqueeze(-1)

        return selected_action, action_probs, dist


class CategoricalDistParams(CategoricalDist):
    """ Multilayer perceptron with Categorical distribution output."""

    def __init__(self, epsilon=1e-8, **kwargs):
        """Initialize."""
        super(CategoricalDistParams, self).__init__(**kwargs)
        self.epsilon = epsilon

    def forward(self, x, deterministic=False):
        """ Forward method implementation."""
        selected_action, action_probs, dist = super(CategoricalDistParams, self).forward(x)

        if deterministic:
            selected_action = torch.argmax(action_probs, dim=-1, keepdim=True)

        z = (action_probs == 0.0)
        z = z.float() * self.epsilon
        log_probs = torch.log(action_probs + z)

        return selected_action, action_probs, log_probs, dist