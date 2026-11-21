import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch import nn
from architectures.mlp import MLP
from architectures.m2_vae.vae_utils import ContentEncoderMLP, Decoder


### Gradient Reversal Layer
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

### Discriminator (D)   
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(Discriminator, self).__init__()
        self.model = ContentEncoderMLP(input_size, hidden_sizes, num_classes + 1)

    def forward(self, x):
        return self.model(x)
    
### Auxiliary Classifier (C)
class AuxiliaryClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, z):
        return self.model(z)
    

class SSGAN(nn.Module):
    def __init__(self, params, num_step_within_epoch, device):
        super(SSGAN, self).__init__()

        self.encoder = ContentEncoderMLP(params["input_dim"], params["hidden_dim"], params["latent_dim"]) 
        self.decoder = Decoder(params["latent_dim"], params["decoder_hidden_dims"], params["input_dim"], params["y_dim"])
        self.discriminator = Discriminator(params["input_dim"], params["hidden_dim"], params["y_dim"])
        self.aux_classifier = AuxiliaryClassifier(params["latent_dim"], params["y_dim"])
        # self.mapper = Mapper(params["latent_dim"], params["y_dim"], params["mapper_hidden_layers"], activation)

        self.device = device
        self.num_classes = params["y_dim"]
        self.batch_size = params["batch_size"]
        self.lambda_disentangle = params["lambda_disentangle"]
        self.aux_reconstruction_loss_weight = params["aux_reconstruction_loss_weight"]

        # Optimizers
        self.optimizer_EG = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=params["encoder_lr"])
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=params["discriminator_lr"], weight_decay=1e-4)
        self.optimizer_C = optim.Adam(self.aux_classifier.parameters(), lr=params["aux_classifier_lr"])


    def get_classify_logits(self):
        return self.logits_real_labeled[:,:-1], self.logits_real_unlabeled[:,:-1]
    
    def classify(self, images):
        return self.discriminator(images)[:,:-1]
    
    def classify_with_aux_classifier(self, images):
        z_agnostic_labeled_aux = self.encoder(images)
        logits_C = self.aux_classifier(z_agnostic_labeled_aux.detach())
        return logits_C

    def train_model(self,x,y,u, epoch):
        # self.update += 1 
        labeled_images = x
        labeled_labels = y
        unlabeled_images = u
        #Stacking full images
        real_images = torch.cat([unlabeled_images, labeled_images], dim=0)

        ### Train Discriminator
        self.optimizer_D.zero_grad()

        if len(labeled_images) !=0:
            self.logits_real_labeled = self.discriminator(labeled_images)
            loss_D_labeled = F.cross_entropy(self.logits_real_labeled[:, :-1], labeled_labels)
        else:
            loss_D_labeled = 0.0

        # Real unlabeled data
        self.logits_real_unlabeled = self.discriminator(real_images)
        loss_D_unlabeled = -torch.logsumexp(torch.softmax(self.logits_real_unlabeled, dim=-1)[:, :-1], dim=1).mean()

        # Fake data
        z_agnostic_disc = self.encoder(real_images)
        y_random_disc = torch.randint(0, self.num_classes, (real_images.shape[0],), device=self.device)
        y_random_disc = F.one_hot(y_random_disc, num_classes=self.num_classes).float()
        fake_images = self.decoder(z_agnostic_disc, y_random_disc)
        logits_fake = self.discriminator(fake_images.detach())
        # Compute log-sum-exp for real classes
        loss_D_fake = -torch.log(torch.softmax(logits_fake, dim=-1)[:, -1]).mean()


        # Total Discriminator loss
        loss_D = loss_D_labeled + loss_D_fake + loss_D_unlabeled
        loss_D.backward()
        self.optimizer_D.step()

        ### Train Auxiliary Classifier
        self.optimizer_C.zero_grad()
        if len(labeled_images) !=0:
            z_agnostic_labeled_aux = self.encoder(labeled_images)
            logits_C = self.aux_classifier(z_agnostic_labeled_aux.detach())
            loss_C = F.cross_entropy(logits_C, labeled_labels)
            loss_C.backward()
            self.optimizer_C.step()

        ### Train Encoder and decoder
        self.optimizer_EG.zero_grad()

        # Reconstruction loss for labeled data
        if len(labeled_images) !=0:
            z_agnostic_labeled_eg = self.encoder(labeled_images)
            recon_labeled = self.decoder(z_agnostic_labeled_eg, labeled_labels)
            loss_recon_labeled = F.mse_loss(recon_labeled, labeled_images)
        else:
            loss_recon_labeled = 0.0

        if len(unlabeled_images) !=0:
            with torch.no_grad():
                logits_unlabeled = self.discriminator(unlabeled_images)
                pseudo_labels = torch.argmax(logits_unlabeled[:, :-1].detach(), dim=1)
                pseudo_labels = F.one_hot(pseudo_labels, num_classes=self.num_classes).float()
            z_agnostic_unlabeled_eg = self.encoder(unlabeled_images)
            recon_unlabeled = self.decoder(z_agnostic_unlabeled_eg, pseudo_labels)
            loss_recon_unlabeled = F.mse_loss(recon_unlabeled, unlabeled_images)
        else:
            loss_recon_unlabeled = 0.0

        # Combine reconstruction losses
        total_recon_loss = loss_recon_labeled + loss_recon_unlabeled

        # Adversarial loss
        z_agnostic_eg = self.encoder(real_images)
        y_random = torch.randint(0, self.num_classes, (real_images.shape[0],), device=self.device)
        y_random = F.one_hot(y_random, num_classes=self.num_classes).float()
        fake_images = self.decoder(z_agnostic_eg, y_random)
        logits_fake = self.discriminator(fake_images)
        loss_adv = -torch.logsumexp(torch.softmax(logits_fake, dim=-1)[:, :-1], dim=1).mean()

        # Disentanglement loss via gradient reversal
        if len(labeled_images) !=0:
            z_agnostic_reversed = GradientReversalLayer.apply(z_agnostic_labeled_eg, self.lambda_disentangle)
            logits_C_reversed = self.aux_classifier(z_agnostic_reversed)
            loss_disentangle = F.cross_entropy(logits_C_reversed, labeled_labels)
        else:
            loss_disentangle = 0.0

        # Total Encoder-decoder loss
        loss_EG = total_recon_loss + loss_adv + self.lambda_disentangle * loss_disentangle
        loss_EG.backward()
        self.optimizer_EG.step()
        # self.scheduler_EG.step()

        return total_recon_loss.item(), loss_disentangle.item(), loss_D.item(), loss_D_labeled.item(), loss_D_unlabeled.item(), \
            loss_D_fake.item(), loss_C.item(), loss_adv.item()
    

    def compute_aux_reconstruction_loss(self, real_images, fake_images, y_random):
        batch_size, channels, height, width = real_images.shape
        device = real_images.device
    
        # Predict classes for real images
        with torch.no_grad():
            logits_real = self.discriminator(real_images)
            pred_classes = torch.argmax(logits_real[:, :-1], dim=1)  # Shape: (batch_size,)
        target_classes = torch.argmax(y_random, dim=1)  # Shape: (batch_size,)
    
        # Scale images from [0, 1] to [0, 255] and ensure uint8 precision
        real_images_scaled = (real_images * 255).to(torch.uint8)
    
        # Define color ranges in RGB (matching your [0, 255] ranges)
        color_ranges = {
            0: (torch.tensor([199, 75, 75], dtype=torch.uint8, device=device), 
                torch.tensor([255, 77, 77], dtype=torch.uint8, device=device)),  # Red
            1: (torch.tensor([75, 199, 75], dtype=torch.uint8, device=device), 
                torch.tensor([77, 255, 77], dtype=torch.uint8, device=device)),  # Green
            2: (torch.tensor([75, 75, 199], dtype=torch.uint8, device=device), 
                torch.tensor([77, 77, 255], dtype=torch.uint8, device=device)),  # Blue
            3: (torch.tensor([199, 199, 75], dtype=torch.uint8, device=device), 
                torch.tensor([255, 255, 77], dtype=torch.uint8, device=device)),  # Yellow
        }
        
        # Quadrant bounds
        quad_bounds = {
            1: ((0, height//2), (width//2, width)),  # Top-right
            2: ((0, height//2), (0, width//2)),      # Top-left
            3: ((height//2, height), (0, width//2)),  # Bottom-left
            4: ((height//2, height), (width//2, width))  # Bottom-right
        }
    
        # Detect quadrant colors across batch
        quadrant_colors = torch.full((batch_size, 4), -1, dtype=torch.long, device=device)  # 4 quadrants
        for quad_idx, ((y_start, y_end), (x_start, x_end)) in quad_bounds.items():
            quadrant = real_images_scaled[:, :, y_start:y_end, x_start:x_end]  # (batch, 3, h/2, w/2)
            
            # Check each color range
            for color_idx, (lower, upper) in color_ranges.items():
                lower = lower.view(1, 3, 1, 1)  # Broadcastable shape
                upper = upper.view(1, 3, 1, 1)
                mask = torch.all((quadrant >= lower) & (quadrant <= upper), dim=1)  # (batch, h/2, w/2)
                presence = mask.sum(dim=(1, 2)) > 0  # (batch,)
                quadrant_colors[:, quad_idx-1] = torch.where(presence, color_idx, quadrant_colors[:, quad_idx-1])

        # Infer missing color for unmatched quadrants
        for b in range(batch_size):
            assigned_colors = set(quadrant_colors[b].tolist()) - {-1}  # Exclude -1
            if len(assigned_colors) <= 3:  # Three colors detected, one missing
                missing_color = next(i for i in range(self.num_classes) if i not in assigned_colors)
                unmatched = (quadrant_colors[b] == -1).nonzero(as_tuple=True)[0]
                if unmatched.numel() > 0:
                    quadrant_colors[b, unmatched[0]] = missing_color
        
        # Initialize mask for the batch
        mask = torch.ones(batch_size, channels, height, width, dtype=torch.float32, device=device)

        # Vectorized masking
        pred_mask = (quadrant_colors == pred_classes.unsqueeze(1)).float()  # (batch_size, 4)
        target_mask = (quadrant_colors == target_classes.unsqueeze(1)).float()  # (batch_size, 4)
        combined_mask = (pred_mask + target_mask).clamp(max=1.0)  # (batch_size, 4)
    
        for quad_idx, ((y_start, y_end), (x_start, x_end)) in quad_bounds.items():
            quad_mask = combined_mask[:, quad_idx-1].view(batch_size, 1, 1, 1)  # (batch, 1, 1, 1)
            mask[:, :, y_start:y_end, x_start:x_end] = mask[:, :, y_start:y_end, x_start:x_end] * (1 - quad_mask)
    
        # Compute MSE on masked regions
        diff = (fake_images - real_images) * mask
        aux_recon_loss = torch.mean(diff ** 2)
    
        return aux_recon_loss
    
    def get_latent(self, x):
        z = self.encoder(x)
        logits = self.classify(x)
        c = F.one_hot(torch.argmax(logits[:, :-1], dim=1), num_classes=self.num_classes).float()
        return z, c
    
    def edit_latent(self, z, c):
        return self.mapper(z, c)

    def generate(self, x, cls):
        if cls is None:
            if len(x.shape) <= 2:
                raise ValueError ("For reconstruction, please pass the image. Seems like you passed the latent")
            logits = self.classify(x)
            labels = torch.argmax(logits[:, :-1], dim=1)
            cls = F.one_hot(labels, num_classes=self.num_classes).float()
        z_agnostic = self.encoder(x)
        return self.decoder(z_agnostic, cls)
    
    def save(self, save_path, epochs=0):
        torch.save(self.state_dict(), f'{save_path}/model.pt')
        print(f"[INFO] model saved after {epochs} at, {save_path}/model.pt")
        
    def load(self, model_dir):
        params = torch.load(model_dir, map_location=self.device, weights_only=True)
        # self.load_state_dict(params, strict=False)
        self.load_state_dict(params)
        print("[INFO] GAN model loaded from: ", model_dir)


        









        





