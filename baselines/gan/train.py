import sys
import cv2
import torch
import hydra
import wandb
import numpy as np

from PIL import Image
from trainer import UNIT_Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from utils import load_images_as_numpy_array, create_dump_directory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Dataset(Dataset):
    def __init__(self, data):
        """
        Custom Dataset for VAE training.

        Args:
            data (list of str): List of data.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return the data at index idx.

        Args:
            idx (int): Index.

        Returns:
            str: data.
        """
        return self.data[idx]

@hydra.main(config_path="../../config", config_name="train_gan", version_base=None)
def train(config):
    max_iter = config['max_iter']
    trainer = UNIT_Trainer(config).to(device)
    
    if config.use_wandb:
        wandb.init(project="P_3_GAN", name=f"GAN_{config.env}", config=OmegaConf.to_container(config, resolve=True))

    checkpoint_directory = create_dump_directory(f"model_weights/GAN/{config.env}")
    print(checkpoint_directory)

    data_a = load_images_as_numpy_array("data/MiniWorld/Random/vae/domain_3/images")
    data_b = load_images_as_numpy_array("data/MiniWorld/Random/vae/domain_2/images")
    data_a = data_a.transpose(0,3,1,2)
    data_b = data_b.transpose(0,3,1,2)
    
    data_a = Dataset(data_a)
    data_b = Dataset(data_b)

    train_loader_a, train_loader_b = DataLoader(data_a, batch_size=64, shuffle=True), DataLoader(data_b, batch_size=150, shuffle=True)
    if config.resume_dir:
        iterations = trainer.resume(config.resume_dir, hyperparameters=config)
    else:
        iterations = 0
    while True:
        for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
            images_a, images_b = images_a.cuda().detach().float()/255, images_b.cuda().detach().float()/255

            # with Timer("Elapsed time in update: %f"):
            # Main training code
            dis_metrics = trainer.dis_update(images_a, images_b, config)
            gen_metrics = trainer.gen_update(images_a, images_b, config)
            if config.use_wandb:
                dis_metrics.update(gen_metrics)
                wandb.log(dis_metrics, step=iterations)
            torch.cuda.synchronize()
            trainer.update_learning_rate()

            iterations += 1
            if iterations >= max_iter:
                trainer.save(checkpoint_directory, iterations)
                sys.exit('Finish training')
            
            if iterations % 10000 == 0:
                trainer.save(checkpoint_directory, iterations)
            # Save network weights
            # if (iterations + 1) % config['snapshot_save_iter'] == 0:
            #     trainer.save(checkpoint_directory, iterations)


            
if __name__ == "__main__":
    train()