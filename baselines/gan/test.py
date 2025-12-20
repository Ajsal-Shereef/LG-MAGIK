import torch

from utils import get_config, get_data, Timer
from trainer import UNIT_Trainer
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image
from architectures.m2_vae.dgm import DeepGenerativeModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Dataset(Dataset):
    def __init__(self, sentences):
        """
        Custom Dataset for VAE training.

        Args:
            sentences (list of str): List of sentences.
        """
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Return the sentence at index idx.

        Args:
            idx (int): Index.

        Returns:
            str: Sentence.
        """
        return self.sentences[idx]

# def test():
#     config = get_config('breakout-diagonals.yaml')
#     size=(10, 10)
#     num_images = size[0] * size[1] // 2
#     trainer = UNIT_Trainer(config).to(device)
#     trainer.load_model("models/B-G/2025-02-18_15-37-50_6P0ENZ", device)
#     trainer.eval()

#     data_a = get_data("data/Both/data.pkl")
#     data_b = get_data("data/Green_ball/data.pkl")

#     data_a, test_a = train_test_split(data_a, test_size=0.10, random_state=42)
#     data_b, test_b = train_test_split(data_b, test_size=0.10, random_state=42)

#     data_a = Dataset(data_a)
#     data_b = Dataset(data_b)

#     train_loader_a, train_loader_b = DataLoader(data_a, batch_size=50, shuffle=True), DataLoader(data_b, batch_size=50, shuffle=True)

#     for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
#         images_a, images_b = images_a.cuda().detach().float()/255, images_b.cuda().detach().float()/255
#         images_a, images_b = images_a[:num_images], images_b[:num_images]
#         break
    

#     h_a, n_a = trainer.gen_a.encode(images_a)
#     h_b, n_b = trainer.gen_b.encode(images_b)
#     # decode (cross domain)
#     x_ba = trainer.gen_a.decode(h_b + n_b)
#     x_ab = trainer.gen_b.decode(h_a + n_a)

#     comparison = torch.empty((2 * num_images,) + images_b.size()[1:])
#     comparison[0::2] = images_b
#     comparison[1::2] = x_ba

#     save_image(comparison.data, 'result/result_b.png', nrow=size[0], pad_value=0.3)

#     comparison = torch.empty((2 * num_images,) + images_a.size()[1:])
#     comparison[0::2] = images_a
#     comparison[1::2] = x_ab

#     save_image(comparison.data, 'result/result_a.png', nrow=size[0], pad_value=0.3)

def test():
    config = get_config('config/breakout-diagonals.yaml')
    size = (10, 4)  # Adjusted for two-row format
    num_images = size[1]  # Number of images per row
    trainer = UNIT_Trainer(config).to(device)
    trainer.load_model("models/B-R/2025-02-17_20-08-15_CX4LDY", device)
    trainer.eval()

    model = DeepGenerativeModel(get_config("config/m2.yaml")["M2_Network"]).to(device)
    model.load("models/vae/2025-02-16_02-11-20_Y9VWZD/model.pt")
    model.to(device)
    model.eval()

    data_a = get_data("data/Both/data.pkl")
    data_b = get_data("data/Red_ball/data.pkl")

    data_a, test_a = train_test_split(data_a, test_size=0.10, random_state=42)
    data_b, test_b = train_test_split(data_b, test_size=0.10, random_state=42)

    data_a = Dataset(data_a)
    data_b = Dataset(data_b)

    train_loader_a, train_loader_b = DataLoader(data_a, batch_size=50, shuffle=True), DataLoader(data_b, batch_size=50, shuffle=True)

    for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
        images_a, images_b = images_a.cuda().detach().float()/255, images_b.cuda().detach().float()/255
        images_a, images_b = images_a[:num_images], images_b[:num_images]
        break
    
    h_a, n_a = trainer.gen_a.encode(images_a)
    h_b, n_b = trainer.gen_b.encode(images_b)
    
    # Decode (cross domain)
    x_ba = trainer.gen_a.decode(h_b + n_b)
    x_ab = trainer.gen_b.decode(h_a + n_a)

    vae_imagined_state = torch.stack([model.generate(image.unsqueeze(0), torch.tensor([0, 1, 0, 0]).to(device).float().unsqueeze(0)) for image in images_b]).squeeze(1)

    # Create comparison grid with two rows
    comparison_b = torch.cat((images_b, x_ba, vae_imagined_state), dim=0)  # First row: original, Second row: imagined
    comparison_a = torch.cat((images_a, x_ab, vae_imagined_state), dim=0)

    save_image(comparison_b, 'result/result_b.png', nrow=num_images, pad_value=0.3)
    save_image(comparison_a, 'result/result_a.png', nrow=num_images, pad_value=0.3)

if __name__ == "__main__":
    test()