import hydra
import wandb

from tqdm import tqdm
from PIL import Image
from itertools import cycle
from torch.autograd import Variable
from helper_functions.utils import *
from torch.utils.data import DataLoader
from architectures.gan.gan import SSGAN
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="config", config_name="gan")
def main(args: DictConfig) -> None:
    wandb.init(project="GAN_feature_input")

    # Initialize dataset and dataloader
    data_dir = f'{args.General.datapath}/{args.General.env}/feature_based/Random'
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
    model_dir = create_dump_directory(f"model_weights/{args.General.env}/MAGIK/gan")
    print("[INFO] Dump dir: ", model_dir)

    # Compute class weights
    class_weights = compute_class_weight(class_weight="balanced", 
                                         classes=np.arange(args.Network.y_dim), 
                                         y=np.argmax(labels, axis=1))
    print("[INFO] Class weights: ", class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    num_step_within_epoch = int(np.ceil(len(unlabelled_data)/args.Network.batch_size))

    unlabelled_data = TwoListDataset(unlabelled_data, unused_labels)
    labelled_data = TwoListDataset(labelled_data, labels)
    unlabelled_data_loader = DataLoader(unlabelled_data, batch_size=args.Network.batch_size, shuffle=True)
    labelled_data_loader = DataLoader(labelled_data, batch_size=args.Network.batch_size, shuffle=True)

    model = SSGAN(args.Network, num_step_within_epoch, device).to(device)
    wandb.watch(model)
    wandb.config.update(OmegaConf.to_container(args, resolve=True))
    epoch_bar = tqdm(range(args.Network.epochs), desc="Training Progress", unit="epoch")
    for epoch in epoch_bar:
        total_recon_losses, losses_disentangle, losses_D, losses_D_labeled, losses_D_unlabeled, losses_D_fake, losses_C = 0,0,0,0,0,0,0
        accuracy_labeled, accuracy_unlabeled, accuracy_labeled_eval, accuracy_unlabeled_eval, adversarial_losses, aux_accuracy_labeled, aux_accuracy_unlabeled = 0, 0, 0, 0, 0, 0, 0
        iter = 0
        for (unlabelled, unused_labels), (labelled, labels) in zip(unlabelled_data_loader, cycle(labelled_data_loader)):
            x, y, u, uy = Variable(labelled).to(device).float(), Variable(labels).to(device).float(), Variable(unlabelled).to(device).float(), Variable(unused_labels).to(device).float()
            # total_recon_loss, loss_disentangle, loss_D, loss_D_labeled, loss_D_unlabeled, loss_D_fake, loss_C, adv_loss, mapper_loss, maploss1, maploss2, maploss3, maploss4, aux_reconstrution_loss = model.train_model(x, y, u, epoch)
            total_recon_loss, loss_disentangle, loss_D, loss_D_labeled, loss_D_unlabeled, loss_D_fake, loss_C, adv_loss = model.train_model(x, y, u, epoch)
            total_recon_losses += total_recon_loss
            losses_disentangle += loss_disentangle
            losses_D += loss_D
            losses_D_labeled += loss_D_labeled
            losses_D_unlabeled += loss_D_unlabeled
            losses_D_fake += loss_D_fake
            losses_C += loss_C
            adversarial_losses += adv_loss

            logits_labeled = model.classify(x)
            logits_unlabeled = model.classify(u)
            accuracy_labeled += torch.mean((torch.max(F.softmax(logits_labeled, dim=-1), 1)[1].data == torch.max(y, 1)[1].data).float())
            accuracy_unlabeled += torch.mean((torch.max(F.softmax(logits_unlabeled, dim=-1), 1)[1].data == torch.max(uy, 1)[1].data).float())

            aux_logits_labeled = model.classify_with_aux_classifier(x)
            aux_logits_unlabeled = model.classify_with_aux_classifier(u)
            aux_accuracy_labeled += torch.mean((torch.max(F.softmax(aux_logits_labeled, dim=-1), 1)[1].data == torch.max(y, 1)[1].data).float())
            aux_accuracy_unlabeled += torch.mean((torch.max(F.softmax(aux_logits_unlabeled, dim=-1), 1)[1].data == torch.max(uy, 1)[1].data).float())

            model.eval()
            with torch.no_grad():
                logits_labeled = model.classify(x)
                logits_unlabeled = model.classify(u)
                accuracy_labeled_eval += torch.mean((torch.max(F.softmax(logits_labeled, dim=-1), 1)[1].data == torch.max(y, 1)[1].data).float())
                accuracy_unlabeled_eval += torch.mean((torch.max(F.softmax(logits_unlabeled, dim=-1), 1)[1].data == torch.max(uy, 1)[1].data).float())
            model.train()
            iter += 1
        if epoch%args.Network.save_freequency == 0 and epoch!=0:
            model.save(model_dir, epoch)
        wandb.log({"Accuracy" : accuracy_labeled/len(unlabelled_data_loader),
                   "Unlabelled Accuracy" : accuracy_unlabeled/len(unlabelled_data_loader),
                   "Aux Accuracy" : aux_accuracy_labeled/len(unlabelled_data_loader),
                   "Aux Unlabelled Accuracy" : aux_accuracy_unlabeled/len(unlabelled_data_loader),
                   "Accuracy eval" : accuracy_labeled_eval/len(unlabelled_data_loader),
                   "Unlabelled Accuracy eval" : accuracy_unlabeled_eval/len(unlabelled_data_loader),
                   "Reconstruction_loss" : total_recon_losses/len(unlabelled_data_loader),
                   "Disentagle loss" : losses_disentangle/len(unlabelled_data_loader),
                   "Discriminator loss" : losses_D/len(unlabelled_data_loader),
                   "Disriminator labelled loss" : losses_D_labeled/len(unlabelled_data_loader),
                   "Disriminator unlabelled loss" : losses_D_unlabeled/len(unlabelled_data_loader),
                   "Disriminator fake loss" : losses_D_fake/len(unlabelled_data_loader),
                   "Advesarial loss" : adversarial_losses/len(unlabelled_data_loader),
                   "Aux classifier loss" : losses_C/len(unlabelled_data_loader)}, step=epoch)
        epoch_bar.set_description(f"Accuracy {accuracy_labeled/len(unlabelled_data_loader)}")
        epoch_bar.update(1)
    model.save(model_dir)
    wandb.finish()


if __name__ == "__main__":
    main()  