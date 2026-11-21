import cv2
import wandb
import json
import hydra
import torch
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from itertools import cycle
from torch.autograd import Variable
from helper_functions.utils import *
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
# from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from architectures.m2_vae.dgm import DeepGenerativeModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@hydra.main(version_base=None, config_path="config", config_name="SimplePickup")
def main(args: DictConfig) -> None:
    # wandb.init(project=args.General.env)
    # Initialize dataset and dataloader
    #Loading the dataset
    datasets = get_data(f'{args.General.datapath}/{args.General.env}/Random/data.pkl')
    print("Total data: ", len(datasets))
    class_probs = get_data(f'{args.General.datapath}/{args.General.env}/Random/class_prob.pkl')
    
    unlabelled_data, labelled_data, unused_labels, labels = train_test_split(datasets, class_probs, test_size=args.General.human_label_ratio, random_state=42)
    print("Number of labelled data: ", len(labelled_data))
    print("Labelled data ratio: ", args.General.human_label_ratio)
    #Create a model dump directory
    model_dir = create_dump_directory("models/m2_vae")
    print("Dump dir: ", model_dir)

    unlabelled_data = TwoListDataset(unlabelled_data, unused_labels)
    labelled_data = TwoListDataset(labelled_data, labels)
    unlabelled_data_loader = DataLoader(unlabelled_data, batch_size=args.Network.batch_size, shuffle=True)
    labelled_data_loader = DataLoader(labelled_data, batch_size=args.Network.batch_size, shuffle=True)
    #Creating the model and the optimizer
    model = DeepGenerativeModel(args.Network).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.Network.lr_model, betas=(0.9, 0.999))

    # scheduler_model = CosineAnnealingLR(optimizer, T_max=args.Network.epochs)
    
    # wandb.watch(model)
    # wandb.config.update(OmegaConf.to_container(args, resolve=True))
    epoch_bar = tqdm(range(args.Network.epochs), desc="Training Progress", unit="epoch")
    for epoch in epoch_bar:
        total_loss, accuracy_labeled, accuracy_unlabeled, hsic_losses = 0, 0, 0, 0
        classification_losses, reconstruction_errors, z_kl_losses, u_kl_losses  = 0, 0, 0, 0
        L_losses, U_losses = 0, 0
        iter = 0
        for (unlabelled, unused_labels), (labelled, labels) in zip(unlabelled_data_loader, cycle(labelled_data_loader)):
            x, y, u, uy = Variable(labelled).to(device).float(), Variable(labels).to(device).float(), Variable(unlabelled).to(device).float(), Variable(unused_labels).to(device).float()

            x_reconstructed, x_z, x_z_mu, x_z_log_var, x_c_logits, x_c = model(x, y)
            # kl_divergence_weight = anneal_coefficient(epoch, args.Network.epochs, args.Network.kl_weight_start, args.Network.kl_weight_end, 150, True)
            kl_divergence_weight = args.Network.kl_divergence_weight

            wessestein_weight = 1
            total_loss_L, cls_loss= model.L(x, x_reconstructed, x_z_mu, x_z_log_var, y, x_c_logits, x_c, wessestein_weight, kl_weight=kl_divergence_weight)

            u_reconstructed, u_z, u_z_mu, u_z_log_var, u_c_logits, u_c = model(u)
            total_loss_U, recon_loss, kl_z, kl_c, hsic_loss = model.U(u, u_reconstructed, u_z_mu, u_z_log_var, u_c_logits, u_c, wessestein_weight, kl_weight=kl_divergence_weight)
            J_alpha = total_loss_L + total_loss_U 
            optimizer.zero_grad()
            J_alpha.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # scheduler_model.step(J_alpha)
            
            total_loss += J_alpha.item()
            classification_losses += cls_loss.item()
            reconstruction_errors += recon_loss.item()
            z_kl_losses += kl_z.item()
            u_kl_losses+= kl_c.item()
            L_losses += total_loss_L.item()
            U_losses += total_loss_U.item()
            hsic_losses += hsic_loss.item()

            accuracy_labeled += torch.mean((torch.max(F.softmax(x_c_logits, dim=-1), 1)[1].data == torch.max(y, 1)[1].data).float())
            accuracy_unlabeled += torch.mean((torch.max(F.softmax(u_c_logits, dim=-1), 1)[1].data == torch.max(uy, 1)[1].data).float())
            iter += 1
        if epoch%args.Network.save_freequency == 0:
            model.save(model_dir, epoch)
        wandb.log({"Loss" : total_loss/len(unlabelled_data_loader),
                   "Classification_loss" : classification_losses/len(unlabelled_data_loader),
                   "Reconstruction_error" : reconstruction_errors/len(unlabelled_data_loader),
                   "Z_kl_loss" : z_kl_losses/len(unlabelled_data_loader),
                   "U_kl_loss" : u_kl_losses/len(unlabelled_data_loader),
                   "L loss" : L_losses/len(unlabelled_data_loader),
                   "U loss" : U_losses/len(unlabelled_data_loader),
                   "HSIC Loss" : hsic_losses/len(unlabelled_data_loader),
                   "Accuracy" : accuracy_labeled/len(unlabelled_data_loader),
                   "Unlabelled Accuracy" : accuracy_unlabeled/len(unlabelled_data_loader),
                   "Current Learning rate" : optimizer.param_groups[0]['lr'],
                   "KL divergence weight" : kl_divergence_weight}, step=epoch)
        epoch_bar.set_description(f"Accuracy {accuracy_labeled/len(unlabelled_data_loader)} Loss {total_loss/len(unlabelled)}")
    model.save(model_dir)
    model.eval()

if __name__ == "__main__":
    main()        
        
        