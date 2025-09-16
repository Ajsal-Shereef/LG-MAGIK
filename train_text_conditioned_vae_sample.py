import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer, CLIPTextModel

# ---------------- Caption Tokenizer Helper ---------------- #
def tokenize_captions(tokenizer, examples, is_train=True):
    captions = []
    for caption in examples:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError("Captions must be string or list[str]")
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs

# ---------------- Dataset ---------------- #
class RandomCaptionDataset(Dataset):
    def __init__(self, n=200, H=40, W=40):
        self.images = torch.rand(n, 3, H, W)  # random RGB images in [0,1]
        colors = ["red", "green", "blue", "yellow"]
        objs = ["ball", "cube"]
        self.captions = [f"agent sees a {random.choice(colors)} {random.choice(objs)}" for _ in range(n)]

    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return self.images[idx], self.captions[idx]
    
class CaptionDiscriminator(nn.Module):
    def __init__(self, z_dim=64, text_dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, text_dim)   # regress toward text embedding
        )
    def forward(self, z):
        return self.net(z)
    
# ---------- Gradient Reversal ----------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

# ---------- Cross Attention ----------
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

# ---------- FiLM Block ----------
class FiLMBlock(nn.Module):
    def __init__(self, channels, film_dim, text_dim):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3,1,1)
        self.norm = nn.BatchNorm2d(channels)
        self.film = nn.Linear(film_dim, 2*channels)
        self.cross = CrossAttention(channels, text_dim)
        self.act = nn.GELU()

    def forward(self,x,z,text_feat):
        out = self.norm(self.conv(x))
        gamma,beta = self.film(z).chunk(2,dim=-1)
        out = out*(1+gamma.unsqueeze(-1).unsqueeze(-1)) + beta.unsqueeze(-1).unsqueeze(-1)
        out = out + self.cross(out,text_feat)
        return self.act(out)

# ---------- Symmetric Encoder ----------
class Encoder(nn.Module):
    def __init__(self,in_ch=3,base_ch=64,depth=4,z_dim=128):
        super().__init__()
        chans=[base_ch*(2**i) for i in range(depth)]
        self.convs=nn.ModuleList()
        prev=in_ch
        for ch in chans:
            self.convs.append(nn.Conv2d(prev,ch,4,2,1))
            prev=ch
        self.act=nn.ReLU()
        self.fc_mu=nn.Linear(chans[-1],z_dim)
        self.fc_logvar=nn.Linear(chans[-1],z_dim)

    def forward(self,x):
        b=x.size(0)
        for conv in self.convs:
            x=self.act(conv(x))
        g=F.adaptive_avg_pool2d(x,1).view(b,-1)
        mu = self.fc_mu(g)
        logvar = self.fc_logvar(g)
        return mu, logvar, x  # return feature map for decoder size

# ---------- Symmetric Decoder (dynamic size) ----------
class Decoder(nn.Module):
    def __init__(self,out_ch=3,base_ch=64,depth=4,z_dim=128,text_dim=512):
        super().__init__()
        self.depth = depth
        chans=[base_ch*(2**i) for i in range(depth)]
        self.fc=nn.Linear(z_dim,chans[-1])
        self.blocks=nn.ModuleList()
        self.ups=nn.ModuleList()
        for i in reversed(range(depth)):
            self.blocks.append(FiLMBlock(chans[i],z_dim,text_dim))
            self.ups.append(
                nn.ConvTranspose2d(chans[i],chans[i-1] if i>0 else base_ch,4,2,1)
            )
        self.final=nn.Conv2d(base_ch,out_ch,3,1,1)

    def forward(self,z,text_feat,enc_feat_shape):
        b=z.size(0)
        _,C,H,W = enc_feat_shape
        x=self.fc(z).view(b,C,1,1).expand(b,C,H,W)  # start from encoder’s last feature map size
        for blk,up in zip(self.blocks,self.ups):
            x=blk(x,z,text_feat)
            x=up(x)
        return torch.sigmoid(self.final(x))

# ---------- Discriminator ----------
class CaptionDiscriminator(nn.Module):
    def __init__(self,z_dim=128,text_dim=512,hid=256):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(z_dim,hid),nn.ReLU(),
            nn.Linear(hid,hid),nn.ReLU(),
            nn.Linear(hid,text_dim)
        )
    def forward(self,z): return self.net(z)

# ---------- Full Model ----------
class TextConditionedVAE(nn.Module):
    def __init__(self,text_encoder_path="openai/clip-vit-base-patch32",
                 z_dim=128,base_ch=64,depth=4,adv_lambda=1.0):
        super().__init__()
        self.tokenizer=CLIPTokenizer.from_pretrained(text_encoder_path)
        self.text_encoder=CLIPTextModel.from_pretrained(text_encoder_path)
        text_dim=self.text_encoder.config.hidden_size
        self.encoder=Encoder(3,base_ch,depth,z_dim)
        self.decoder=Decoder(3,base_ch,depth,z_dim,text_dim)
        self.disc=CaptionDiscriminator(z_dim,text_dim)
        self.adv_lambda=adv_lambda

    def reparam(self,mu,logvar):
        std=torch.exp(0.5*logvar);eps=torch.randn_like(std)
        return mu+eps*std

    def forward(self,imgs,captions):
        device=imgs.device
        tokens=self.tokenizer(captions,max_length=self.tokenizer.model_max_length,
                              padding="max_length",truncation=True,return_tensors="pt").to(device)
        text_feats=self.text_encoder(tokens.input_ids, return_dict=False)[0] # (B,T,D)
        pooled=text_feats.mean(1)
        mu,logvar,enc_feat=self.encoder(imgs)
        z=self.reparam(mu,logvar)
        # reconstruction
        x_rec=self.decoder(z,text_feats,enc_feat.shape)
        # adversarial disentanglement
        z_grl=grad_reverse(z,self.adv_lambda)
        pred=self.disc(z_grl)
        disc_loss=F.mse_loss(pred,pooled.detach())
        return x_rec,mu,logvar,disc_loss


# ---------------- Loss ---------------- #
def vae_loss(x_rec, x, mu, logvar, recon_weight=1.0, kl_weight=1.0):
    recon = F.l1_loss(x_rec, x, reduction="mean")
    kl = -0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon, kl

# ---------------- Train/Test Script ---------------- #
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = RandomCaptionDataset(n=100)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = TextConditionedVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    # train few steps
    model.train()
    for epoch in range(2):
        for imgs, caps in loader:
            imgs = imgs.to(device)
            x_rec, mu, logvar, disc_loss = model(imgs, caps)
            rloss, kl = vae_loss(x_rec, imgs, mu, logvar, kl_weight=1e-3)
            # total = VAE loss + adversarial disentanglement
            loss = rloss + 0.1*kl + model.adv_lambda * disc_loss
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"Epoch {epoch} loss={loss.item():.4f} recon={rloss.item():.4f} kl={kl.item():.6f}")

    # test reconstruction + caption swap
    model.eval()
    imgs, caps = next(iter(loader))
    imgs = imgs.to(device)
    with torch.no_grad():
        mu, logvar = model.img_enc(imgs)
        z = model.reparameterize(mu, logvar)
        rec = model(imgs, caps)[0]
        # swap captions (e.g. "red ball" -> "green ball")
        new_caps = [c.replace("red","green") for c in caps]
        swapped = model.decoder(z, model.text_encoder(
            model.tokenizer(new_caps, return_tensors="pt").to(device).input_ids,
            return_dict=False
        )[0])
    print("Reconstruction shape:", rec.shape)
    print("Swapped-gen shape:", swapped.shape)

if __name__ == "__main__":
    main()
