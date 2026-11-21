import cv2
import yaml
import wandb
import hydra
import torch
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from itertools import cycle
from torch.autograd import Variable
from helper_functions.utils import *
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from architectures.m2_vae.dgm import DeepGenerativeModel
from sklearn.utils.class_weight import compute_class_weight


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="config", config_name="m2")
def main(args: DictConfig) -> None:
    # Initialize dataset and dataloader
    # Initialize dataset and dataloader
    data_dir = f'{args.General.datapath}/{args.General.env}/feature_based/SAC'
    #Loading the dataset
    print("[INFO] Selected env: ", args.General.env)
    datasets = get_data(f'{data_dir}/states.pkl')
    class_probs = get_data(f'{data_dir}/class_prob.pkl')
    print("[INFO] Interaction data loaded from: ", f'{data_dir}/states.pkl')
    print("[INFO] Total data: ", len(datasets))
    
    unlabelled_data, labelled_data, unused_labels, labels = train_test_split(datasets, class_probs, test_size=args.General.human_label_ratio, random_state=42)
    print("[INFO] Number of labelled data: ", len(labelled_data))
    print("[INFO] Labelled data ratio: ", args.General.human_label_ratio)
    #Create a model dump directory
    model_dir = create_dump_directory(f"model_weights/{args.General.env}/MAGIK/VAE")
    print("[INFO] Dump dir: ", model_dir)

    # Compute class weights
    class_weights = compute_class_weight(class_weight="balanced", 
                                         classes=np.arange(args.Network.y_dim), 
                                         y=np.argmax(labels, axis=1))
    print("[INFO] Class weights: ", class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    unlabelled_data = TwoListDataset(unlabelled_data, unused_labels)
    labelled_data = TwoListDataset(labelled_data, labels)
    unlabelled_data_loader = DataLoader(unlabelled_data, batch_size=args.Network.batch_size, shuffle=True)
    labelled_data_loader = DataLoader(labelled_data, batch_size=args.Network.batch_size, shuffle=True)
    #Creating the model and the optimizer
    model = DeepGenerativeModel(args.Network, class_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.Network.lr_model, betas=(0.9, 0.999))

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=5e-4)

    # wandb.watch(model)
    # wandb.config.update(OmegaConf.to_container(args, resolve=True))

    reconstruction_per_feature = []
    reconstruction_per_feature_std = []

    epoch_bar = tqdm(range(args.Network.epochs), desc="Training Progress", unit="epoch")
    for epoch in epoch_bar:
        total_loss, accuracy_labeled, accuracy_unlabeled, accuracy_labeled_eval, accuracy_unlabeled_eval, hsic_losses = 0, 0, 0, 0, 0, 0
        classification_losses, reconstruction_errors, z_kl_losses, u_kl_losses  = 0, 0, 0, 0
        L_losses, U_losses = 0, 0
        iter = 0
        for (unlabelled, unused_labels), (labelled, labels) in zip(unlabelled_data_loader, cycle(labelled_data_loader)):
            x, y, u, uy = Variable(labelled).to(device).float(), Variable(labels).to(device).float(), Variable(unlabelled).to(device).float(), Variable(unused_labels).to(device).float()
            # temperature = anneal_coefficient(epoch, args.M2_Network.epochs, 1, 0.05, 0, False)
            temperature = 1.00
            x_reconstructed, x_z, x_z_mu, x_z_log_var, x_c_logits, x_c = model(x, y, temperature)
            u_reconstructed, u_z, u_z_mu, u_z_log_var, u_c_logits, u_c = model(u, temp=temperature)

            # kl_divergence_weight = anneal_coefficient(epoch, args.M2_Network.epochs, args.M2_Network.kl_weight_start, args.M2_Network.kl_weight_end, 100, True)
            kl_divergence_weight = args.Network.kl_divergence_weight

            total_loss_L, cls_loss = model.L(x, x_reconstructed, x_z_mu, x_z_log_var, y, x_c_logits, x_c, kl_weight=kl_divergence_weight, epoch=epoch)
            total_loss_U, recon_loss, kl_z, kl_c, hsic_loss = model.U(u, u_reconstructed, u_z_mu, u_z_log_var, u_c_logits, u_c, kl_weight=kl_divergence_weight, epoch=epoch)

            J_alpha = total_loss_L + total_loss_U 
            optimizer.zero_grad()
            J_alpha.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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
            model.eval()
            with torch.no_grad():
                x_reconstructed, x_z, x_z_mu, x_z_log_var, x_c_logits, x_c = model(x, y, temperature)
                u_reconstructed, u_z, u_z_mu, u_z_log_var, u_c_logits, u_c = model(u, temp=temperature)
                accuracy_labeled_eval += torch.mean((torch.max(F.softmax(x_c_logits, dim=-1), 1)[1].data == torch.max(y, 1)[1].data).float())
                accuracy_unlabeled_eval += torch.mean((torch.max(F.softmax(u_c_logits, dim=-1), 1)[1].data == torch.max(uy, 1)[1].data).float())
            model.train()
            iter += 1
        scheduler.step(total_loss)
        reconstruction_per_feature.append(np.mean(torch.stack(model.reconstruction_loss_per_feature).detach().cpu().numpy(), axis=0))
        reconstruction_per_feature_std.append(np.std(torch.stack(model.reconstruction_loss_per_feature).detach().cpu().numpy(), axis=0))
        model.reconstruction_loss_per_feature = []
        if epoch%args.Network.save_freequency == 0 and epoch!=0:
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
                   "Accuracy eval" : accuracy_labeled_eval/len(unlabelled_data_loader),
                   "Unlabelled Accuracy eval" : accuracy_unlabeled_eval/len(unlabelled_data_loader),
                   "Current Learning rate" : optimizer.param_groups[0]['lr'],
                   "KL divergence weight" : kl_divergence_weight}, step=epoch)
        epoch_bar.set_description(f"Accuracy {accuracy_labeled/len(unlabelled_data_loader)} Loss {total_loss/len(unlabelled_data_loader)}")
    barplot(np.stack(reconstruction_per_feature)[-5:,:], np.stack(reconstruction_per_feature_std)[-5:,:], "visualisations")
    model.save(model_dir)
    model.eval()
    wandb.finish()
    
if __name__ == "__main__":
    # wandb.init(project="M2_Mujoco_feature_based_final_run")
    main()        
        
        