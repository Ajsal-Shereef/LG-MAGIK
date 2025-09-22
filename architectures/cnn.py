import torch
import math
import torch.nn as nn
import math
import numpy as np

# from architectures.m2_vae.vae import FiLM
from architectures.common_utils import identity, get_activation, get_normalisation_2d
from architectures.mlp import MLP, GaussianDist, CategoricalDistParams, TanhGaussianDistParams
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNLayer(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride=1,
        padding=0,
        conv_layer = nn.Conv2d,
        pre_activation_fn=identity,
        activation_fn=nn.LeakyReLU(),
        post_activation_fn=identity,
        gain = math.sqrt(2)
    ):
        super(CNNLayer, self).__init__()
        self.cnn = conv_layer(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        nn.init.orthogonal_(self.cnn.weight, gain)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.pre_activation_fn = pre_activation_fn
        self.activation_fn = activation_fn
        self.post_activation_fn = post_activation_fn

    def forward(self, x):
        x = self.cnn(x)
        x = self.pre_activation_fn(x)
        x = self.activation_fn(x)
        x = self.post_activation_fn(x)
        x = self.batch_norm(x)
        return x


class CNN(nn.Module):
    """ Baseline of Convolution neural network. """
    def __init__(self, cnn_layers, fc_layers):
        """
        cnn_layers: List[CNNLayer]
        fc_layers: MLP
        """
        super(CNN, self).__init__()

        self.cnn_layers = cnn_layers
        self.fc_layers = fc_layers

        self.cnn = nn.Sequential()
        for i, cnn_layer in enumerate(self.cnn_layers):
            self.cnn.add_module("cnn_{}".format(i), cnn_layer)

    def get_cnn_features(self, x, is_flatten=True):
        """
        Get the output of CNN.
        """
        if len(x.size()) == 3:
            x = x.unsqueeze(0)
        x = self.cnn(x)
        # flatten x
        if is_flatten:
            x = x.reshape(x.size(0), -1)
        return x
    
    def get_cnn_feature(self, input, cnn_layer, is_flatten=True):
        if len(input.size()) == 3:
            input = input.unsqueeze(0)
            output = cnn_layer(input)
        if len(input.size()) == 5:
            b,t,c,h,w = input.size()
            input = input.view(b*t, c,h,w)
            output = cnn_layer(input)
            output = output.view(b,t,output.size()[1],output.size()[2],output.size()[3])
        return output

    def forward(self, x, is_flatten = True, **fc_kwargs):
        """
        Forward method implementation.
        x: torch.Tensor
        :return: torch.Tensor
        """
        x = self.get_cnn_features(x, is_flatten)
        if is_flatten:
            if self.fc_layers:
                fc_out = self.fc_layers(x, **fc_kwargs)
            return fc_out, x
        else:
            return None, x
        
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
    
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

    def forward(self, x, cond):
        """
        Args:
            x (Tensor): Feature map of shape (B, C, H, W).
            cond (Tensor): Condition vector of shape (B, cond_dim).
        Returns:
            Tensor: FiLM-modulated feature map.
        """
        # Compute scale and shift parameters and unsqueeze them to (B, C, 1, 1)
        gamma = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)
        beta  = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta
    
class CNNEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(CNNEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type))
        
        for _ in range(n_downsample):
            self.blocks.append(Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type))
            dim *= 2
        
        self.blocks.append(ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type))
        self.output_dim = dim

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
    
class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_k, heads=4, dim_head=64):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner = heads * dim_head
        self.to_q = nn.Linear(dim_q, inner, bias=False)
        self.to_k = nn.Linear(dim_k, inner, bias=False)
        self.to_v = nn.Linear(dim_k, inner, bias=False)
        self.to_out = nn.Linear(inner, dim_q)

    def forward(self, feat, text_feat):
        # feat: (B,C,H,W), text_feat: (B,T,D)
        b,c,h,w = feat.shape
        q = feat.flatten(2).transpose(1,2)  # (B,HW,C)
        k,v = text_feat,text_feat
        q = self.to_q(q); k = self.to_k(k); v = self.to_v(v)

        def reshape(x): return x.view(b,-1,self.heads,q.size(-1)//self.heads).transpose(1,2)
        qh,kh,vh = map(reshape,(q,k,v))
        attn = (qh @ kh.transpose(-2,-1))*self.scale
        attn = attn.softmax(-1)
        out = attn @ vh
        out = out.transpose(1,2).contiguous().view(b,-1,q.size(-1))
        out = self.to_out(out).transpose(1,2).view(b,c,h,w)
        return out
    
class TransformerCrossAttention(nn.Module):
    def __init__(self, channels, text_dim, n_heads=8):
        super().__init__()
        self.query_proj = nn.Linear(channels, channels)
        self.key_proj   = nn.Linear(text_dim, channels)
        self.value_proj = nn.Linear(text_dim, channels)
        self.out_proj   = nn.Linear(channels, channels)
        self.n_heads = n_heads

    def forward(self, img_feat, text_feat):
        """
        img_feat: [B, H*W, C]
        text_feat: [B, T, D]
        """
        B, N, C = img_feat.shape
        H = self.n_heads
        q = self.query_proj(img_feat).view(B, N, H, C // H).transpose(1, 2)  # [B,H,N,d]
        k = self.key_proj(text_feat).view(B, -1, H, C // H).transpose(1, 2)  # [B,H,T,d]
        v = self.value_proj(text_feat).view(B, -1, H, C // H).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ v  # [B,H,N,d]
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(out)

class CrossAttentionFiLM(nn.Module):
    def __init__(self, channels, film_dim, text_dim):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3,1,1)
        self.norm = nn.BatchNorm2d(channels)
        self.film = nn.Linear(film_dim, 2*channels)
        self.cross = CrossAttention(channels, text_dim)
        self.act = nn.GELU()

    def forward(self,x,z,text_feat):
        out = self.norm(self.conv(x))
        gamma,beta = self.film(z).chunk(2,dim=1)
        out = out*(1+gamma) + beta
        out = out + self.cross(out,text_feat)
        return self.act(out)
    
class CrossAttentionFiLMSpatial(nn.Module):
    """
    FiLM and Cross-Attention block adapted for a SPATIAL latent code z.
    """
    def __init__(self, channels, latent_channels, text_dim, layer_idx):
        super().__init__()

        # 1x1 Convolution to match latent channels to feature map channels
        self.z_conv = nn.Conv2d(latent_channels, channels, 1)
        
        # Dynamically set kernel size, stride, padding based on layer index (layer_idx)
        kernel_size = 2 ** (layer_idx)  # 2^(i) for the kernel size
        stride = 2 ** (layer_idx)      # 2^(i) for stride
        padding = 0                    # No padding

        # Define deconvolution layer with dynamic parameters
        self.z_deconv = nn.ConvTranspose2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)
        
        # FiLM: Linear layer to generate gamma and beta
        self.film = nn.Linear(channels, 2 * channels)
        
        # Standard convolution and batch normalization
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm = nn.BatchNorm2d(channels)
        
        # Cross-attention layer
        self.cross = TransformerCrossAttention(channels, text_dim)
        
        # Activation function (GELU)
        self.act = nn.GELU()

    def forward(self, x, z, text_feat):
        """
        Args:
            x (torch.Tensor): Main feature map (B, C, H, W)
            z (torch.Tensor): SPATIAL latent code (B, latent_channels, H_z, W_z)
            text_feat (torch.Tensor): Text features (B, T, D)
        """
        B, C, H, W = x.shape  # Feature map size
        _, _, H_z, W_z = z.shape  # Latent code size

        # Step 1: Apply convolution to the latent code z to match the channels of x
        z_proj = self.z_conv(z)  # (B, C, H_z, W_z)
        z_proj = self.z_deconv(z_proj)  # Upscale z_proj to match x spatial dimensions
        # Step 3: Compute FiLM modulation parameters (gamma, beta)
        z_flat = z_proj.flatten(2).permute(0, 2, 1)  # Shape: (B, H*W, C)
        
        # Generate gamma and beta from the latent code
        gamma, beta = self.film(z_flat).chunk(2, dim=-1)  # Shape: (B, H*W, C)
        
        # Step 4: Apply FiLM modulation to the feature map
        out = self.norm(self.conv(x))
        
        # Unsqueeze gamma and beta to match spatial dimensions for broadcasting
        gamma = gamma.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        beta = beta.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Apply FiLM modulation: out = out * (1 + gamma) + beta
        out = out * (1 + gamma) + beta
        
        # --- Cross-Attention ---
        B, C, H, W = out.shape
        # Flatten spatial dims: [B, C, H, W] -> [B, H*W, C]
        out_flat = out.view(B, C, H*W).permute(0, 2, 1)  # [B, N=H*W, C]

        # Apply transformer-style cross-attention
        attn_out = self.cross(out_flat, text_feat)  # [B, N, C]

        # Reshape back to [B, C, H, W]
        out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        
        return self.act(out)


class CNNDecoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, z_dim, res_norm='adain', activ='relu', pad_type='zero', fc_input_dim=[10,10], fc_hidden_dim = [512], mlp_act='identity'):
        super(CNNDecoder, self).__init__()
        self.dim = dim
        self.fc_input = fc_input_dim
        #fc layers
        self.mlp = MLP(z_dim, dim*np.prod(fc_input_dim), fc_hidden_dim, hidden_activation=mlp_act, output_activation=mlp_act)
        # AdaIN residual blocks
        self.res_layers = ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)
        # upsampling blocks
        self.upsample_layers = nn.Sequential()
        for i in range(n_upsample):
            self.upsample_layers.add_module("UpSampling_{}".format(i), nn.Upsample(scale_factor=2))
            self.upsample_layers.add_module("Conv2dBlock_{}".format(i), Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type))
            dim //= 2
        # use reflection padding in the last conv layer
        self.upsample_layers.add_module("Conv2dBlock_{}".format(i+1), Conv2dBlock(dim, output_dim, 3, 1, 1, norm='none', activation='none', pad_type=pad_type)) # Earlier 7,1,3

    def forward(self, x):
        x = self.mlp(x)
        x = x.view(x.shape[0], self.dim, self.fc_input[0], self.fc_input[1])
        x = self.res_layers(x)
        x = self.upsample_layers(x)
        return (torch.tanh(x) + 1) / 2 #using tanh activation + scalling
    
    def get_all_features(self, x):
        features = []
        x = self.mlp(x)
        x = x.view(x.shape[0], self.dim, self.fc_input[0], self.fc_input[1])
        features.append(x)

        x = self.res_layers(x)
        features.append(x)

        for block in self.upsample_layers:
            x = block(x)
            features.append(x)

        features.append(x)
        return features
    
class FinalTextConditionedOutput(nn.Module):
    def __init__(self, in_channels, out_channels, text_dim, n_heads=8):
        super().__init__()
        self.cross_attention = TransformerCrossAttention(in_channels, text_dim, n_heads)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x, text_feat):
        """
        x: [B, C, H, W] - Image feature map
        text_feat: [B, T, D] - Text feature (from the text encoder)
        """
        B, C, H, W = x.shape

        # Flatten the image feature map for cross-attention (B, C, H, W) -> (B, H*W, C)
        img_feat = x.view(B, C, H * W).transpose(1, 2)  # Shape: [B, H*W, C]

        # Apply the cross-attention to refine the image feature map using text
        refined_feat = self.cross_attention(img_feat, text_feat)  # Shape: [B, H*W, C]

        # Reshape it back to the image spatial dimensions [B, C, H, W]
        refined_feat = refined_feat.view(B, C, H, W)

        # Pass the refined features through the final convolution
        return torch.tanh(self.conv(refined_feat))
    
class CNNTextConditionedDecoder(nn.Module):
    def __init__(self, n_upsample, dim, output_dim, clip_model, latent_channel):
        super(CNNTextConditionedDecoder, self).__init__()
        self.dim = dim
        #Conv layer to map latent channel to dim
        self.mapping_conv = nn.Conv2d(2*latent_channel, dim, 1)
        #Text encoder and tockenizer
        self.tokenizer=CLIPTokenizer.from_pretrained(clip_model)
        self.text_encoder=CLIPTextModel.from_pretrained(clip_model)
        #Freeze the CLIP model parameters
        for params in self.text_encoder.parameters():
            params.requires_grad = False
        self.text_encoder.eval()
        self.text_dim=self.text_encoder.config.hidden_size
        self.text_adapter = MLP(self.text_dim, self.text_dim, [64, 256], hidden_activation = 'gelu', norm='ln')
        self.text_to_latent = nn.Linear(self.text_dim, latent_channel)
        
        self.blocks=nn.ModuleList()
        self.ups=nn.ModuleList()
        
        for i in range(n_upsample):
            self.blocks.append(CrossAttentionFiLMSpatial(dim,2*latent_channel,self.text_dim, i))
            self.ups.append(nn.ConvTranspose2d(dim, dim//2,4,2,1))
            dim //= 2
            
        self.final = FinalTextConditionedOutput(dim, output_dim, self.text_dim)
        
    def forward(self,z,text_tockens, attention_mask):
        self.text_feats=self.text_encoder(text_tockens, return_dict=False)[0] # (B,T,D)
        self.text_feats = self.text_adapter(self.text_feats)
        # Expand text_feat to spatial (broadcast over H_z, W_z)
        text_global = self.text_feats.mean(dim=1)  # [B, D]
        text_proj = self.text_to_latent(text_global).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        text_proj = text_proj.expand(-1, -1, z.size(2), z.size(3))  # match z spatial size

        z = torch.cat([z, text_proj], dim=1)  # concat along channel
        x = self.mapping_conv(z)
        for blk,up in zip(self.blocks,self.ups):
            x=blk(x,z,self.text_feats)
            x=up(x)
        return self.final(x, self.text_feats)
    
class CNNTwoLatentDecoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, z_dim, y_dim, res_norm='adain', activ='relu', pad_type='zero', fc_input_dim=[10,10]):
        super(CNNTwoLatentDecoder, self).__init__()
        self.dim = dim
        self.fc_input = fc_input_dim
        #fc layers
        self.mlp = MLP(z_dim, dim*np.prod(fc_input_dim), [128, 512, 2048])
        # AdaIN residual blocks
        self.res_layers = ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)
        #FiLM layer initialisation
        self.film1 = FiLM(dim, y_dim)
        # upsampling blocks
        self.upsample_layers = nn.Sequential()
        for i in range(n_upsample):
            self.upsample_layers.add_module("UpSampling_{}".format(i), nn.Upsample(scale_factor=2))
            self.upsample_layers.add_module("Conv2dBlock_{}".format(i), Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type))
            dim //= 2
        # use reflection padding in the last conv layer
        self.upsample_layers.add_module("Conv2dBlock_{}".format(i+1), Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='none', pad_type=pad_type))
        #Final FiLM layer
        self.flim2 = FiLM(output_dim, y_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x, y):
        x = self.mlp(x)
        x = x.view(x.shape[0], self.dim, self.fc_input[0], self.fc_input[1])
        x = self.res_layers(x)
        x = self.film1(x, y)
        x = self.upsample_layers(x)
        x = self.flim2(x, y)
        return self.output_activation(x)
        
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = nn.Sequential()
        for i in range(num_blocks):
            self.model.add_module("ResBlocks_{}".format(i), ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type))
    def forward(self, x):
        return self.model(x)
        
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        self.model = nn.Sequential()
        self.model.add_module("Conv2dBlock_{}".format(0), Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type))
        self.model.add_module("Conv2dBlock_{}".format(1), Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type))

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out
        
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        self.norm = get_normalisation_2d(norm, norm_dim)
        self.activation = get_activation(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
    
class Conv2d_MLP_Model(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu,
                 fc_output_activation=identity,
                 dropout_prob = 0
                 ):
        super(Conv2d_MLP_Model, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]
        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        fc_layers = MLP(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation,
            output_activation=fc_output_activation,
            dropout_prob = dropout_prob
        )

        self.conv_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x, is_flatten = True):
        return self.conv_mlp.forward(x, is_flatten)


class MLP_Model(nn.Module):
    """ Fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self, hidden_units):
        super(MLP_Model, self).__init__()
        self.fc_layers = MLP(
            input_size=25,
            output_size=4,
            hidden_sizes=hidden_units,
            hidden_activation=torch.relu,
            output_activation=identity
        )

    def forward(self, x):
        return self.fc_layers(x)
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class VAE(nn.Module):
    def __init__(self, latent_size):
        super(VAE, self).__init__()

        self.latent_size = latent_size

        # Encoder layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(256 * 5 * 5, 512)
        self.fc21 = nn.Linear(512, latent_size)
        self.fc22 = nn.Linear(512, latent_size)

        # Decoder layers
        self.fc3 = nn.Linear(latent_size, 512)
        self.fc4 = nn.Linear(512, 256 * 5 * 5)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1, padding=0)
        self.deconv5 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=0)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
        else:
            eps = torch.zeros_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = z.view(-1, 256, 5, 5)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        z = F.relu(self.deconv4(z))
        z = torch.sigmoid(self.deconv5(z))
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar
  
class Conv2d_MLP_Gaussian(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu
                 ):
        super(Conv2d_MLP_Gaussian, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        fc_layers = GaussianDist(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation
        )

        self.conv_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x):
        return self.conv_mlp.forward(x)


class Conv2d_MLP_Categorical(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu
                 ):
        super(Conv2d_MLP_Categorical, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        fc_layers = CategoricalDistParams(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation
        )

        self.conv_categorical_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x, deterministic=False):
        return self.conv_categorical_mlp.forward(x, deterministic=deterministic)


class Conv2d_MLP_TanhGaussian(nn.Module):
    """ Default convolution neural network composed of conv2d layer followed by fully-connected MLP models """

    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu
                 ):
        super(Conv2d_MLP_TanhGaussian, self).__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = [input_channels] + channels[:-1]

        post_activation_fns = [identity for _ in range(len(strides))]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            post_activation_fns = [torch.nn.MaxPool2d(max_pool_stride) for max_pool_stride in strides]
            strides = ones
        activation_fns = [nonlinearity for _ in range(len(strides))]

        conv_layers = [CNNLayer(input_channels=ic, output_channels=oc,
                                kernel_size=k, stride=s, padding=p, activation_fn=a_fn, post_activation_fn=p_fn)
                       for (ic, oc, k, s, p, a_fn, p_fn) in zip(in_channels, channels, kernel_sizes, strides, paddings,
                                                                activation_fns, post_activation_fns)]
        fc_layers = TanhGaussianDistParams(
            input_size=fc_input_size,
            output_size=fc_output_size,
            hidden_sizes=fc_hidden_sizes,
            hidden_activation=fc_hidden_activation
        )

        self.conv_tanh_gaussian_mlp = CNN(cnn_layers=conv_layers, fc_layers=fc_layers)

    def forward(self, x, epsilon=1e-6, deterministic=False, reparameterize=True):
        return self.conv_tanh_gaussian_mlp.forward(x, epsilon=1e-6, deterministic=False, reparameterize=True)


class Conv2d_Flatten_MLP(Conv2d_MLP_Model):
    """
    Augmented convolution neural network, in which a feature vector will be appended to
        the features extracted by CNN before entering mlp
    """
    # noinspection PyDefaultArgument
    def __init__(self,
                 # conv2d layer arguments
                 input_channels,
                 # fc layer arguments
                 fc_input_size,
                 fc_output_size,
                 # conv2d optional arguments
                 channels=[32, 32, 32],
                 kernel_sizes=[8, 4, 3],
                 strides=[4, 2, 1],
                 paddings=[0, 1, 1],
                 nonlinearity=torch.relu,
                 use_maxpool=False,
                 # fc layer optional arguments
                 fc_hidden_sizes=[100, 100],
                 fc_hidden_activation=torch.relu,
                 fc_output_activation=identity
                 ):
        super(Conv2d_Flatten_MLP, self).__init__(input_channels=input_channels,
                                                 fc_input_size=fc_input_size,
                                                 fc_output_size=fc_output_size,
                                                 channels=channels, kernel_sizes=kernel_sizes, strides=strides,
                                                 paddings=paddings, nonlinearity=nonlinearity,
                                                 use_maxpool=use_maxpool, fc_hidden_sizes=fc_hidden_sizes,
                                                 fc_hidden_activation=fc_hidden_activation,
                                                 fc_output_activation=fc_output_activation)

    def forward(self, *args):
        obs_x, augment_features = args
        cnn_features = self.conv_mlp.get_cnn_features(obs_x)
        features = torch.cat((cnn_features, augment_features), dim=1)
        return self.conv_mlp.fc_layers(features)








