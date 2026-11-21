import os
import cv2
import torch
import yaml
import string
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from datetime import datetime
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
from torch.utils.data import Dataset
from torch.autograd import Variable
from sklearn.model_selection import StratifiedShuffleSplit

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def visualize_latent_space(model, dataloader, device, all_z=[], all_labels=[], method='pca', save_path='', checkpoint=''):
    """
    Visualize the latent space using PCA or t-SNE.

    Args:
        model (VAE): Trained VAE model.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computations on.
        method (str): 'pca' or 'tsne'.
        save_path (str): Path to save the visualization plot.
    """
    model.eval()
    if len(all_z) == 0:
        all_z = []
        all_labels = []
        with torch.no_grad():
            for data, indices in tqdm(dataloader, desc="Collecting Latent Vectors"):
                # sentences = list(sentences)
                data = data.float().to(device)
                # Forward pass
                reconstruction, inference_out = model(data)
                # labels = torch.argmax(inference_out['prob_cat'], dim=-1)
                labels = inference_out['prob_cat']
                all_z.append(inference_out['latent'].cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                # Optionally, collect labels or other metadata if available
        all_z = np.concatenate(all_z, axis=0)  # [num_samples, latent_dim]
        all_labels = np.concatenate(all_labels, axis=0)
    if method == 'pca':
        reducer = PCA(n_components=2)
        reduced_z = reducer.fit_transform(all_z)
        fig_save_name = f'{save_path}/latent_space_pca_{checkpoint}.png'
    elif method == 'tsne':
        reducer = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        reduced_z = reducer.fit_transform(all_z)
        fig_save_name = f'{save_path}/latent_space_tsne_{checkpoint}.png'
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")
    
    # fig = plt.figure(figsize=(16,10))
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(reduced_z[:,0], reduced_z[:,1], reduced_z[:,2], c=all_labels, cmap='tab10')
    # ax.set_title(f'Latent Space Visualization using {method.upper()}')
    # ax.set_xlabel('Component 1')
    # ax.set_ylabel('Component 2')
    # ax.set_zlabel('Component 3')
    # ax.grid(True)
    # ax.legend()
    # fig.savefig(fig_save_name)
    # plt.close(fig)
    
    plt.scatter(all_z[:,0], all_z[:,1], c=all_labels[:,1], cmap='coolwarm', edgecolor='k', alpha=0.7)
    plt.colorbar(label='Probability of Class 1')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Class latent")
    plt.savefig(fig_save_name)
    plt.close()
    
    print(f"Latent space visualization saved to {fig_save_name}")
    return all_z, all_labels

def get_activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'lrelu':
        return nn.LeakyReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == "identity":
        return nn.Identity()
    else:
        raise ValueError ("Activation out of scope. Please define above")

def vgg_preprocess(batch):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean)) # subtract mean
    return batch

def plot_two_coordinate_sets(set1, set2, colors=None,
                             dot_marker='o', star_marker='*',
                             dot_size=100, star_size=200,
                             xlabel='X coordinate', ylabel='Y coordinate',
                             title='Two Sets of Coordinates: Dots for Set1, Stars for Set2',
                             dir = ''):
    """
    Plots two sets of coordinates, each containing 4 points.
    
    Parameters:
        set1 (np.ndarray): Array of shape (4, 2) containing coordinates for the first set.
        set2 (np.ndarray): Array of shape (4, 2) containing coordinates for the second set.
        colors (list): List of 4 colors for the points. Defaults to ['red', 'green', 'blue', 'yellow'].
        dot_marker (str): Marker style for set1. Default is 'o' (circle).
        star_marker (str): Marker style for set2. Default is '*' (star).
        dot_size (int): Marker size for set1.
        star_size (int): Marker size for set2.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Plot title.
    """
    if colors is None:
        colors = ['red', 'green', 'blue', 'orange']
    
    if set1.shape[0] != 4 or set2.shape[0] != 4:
        raise ValueError("Both coordinate sets must contain exactly 4 points.")
    
    plt.figure(figsize=(8, 6))
    
    # Plot each of the 4 points for both sets.
    for i in range(4):
        # Plot set1 point (dot)
        plt.scatter(set1[i, 0], set1[i, 1],
                    color=colors[i], marker=dot_marker, s=dot_size,
                    label=f'Set1 {colors[i]}' if i == 0 else "")
        # Plot set2 point (star)
        plt.scatter(set2[i, 0], set2[i, 1],
                    color=colors[i], marker=star_marker, s=star_size,
                    label=f'Set2 {colors[i]}' if i == 0 else "")
    
    
    plt.xlim(-0.60, 0.60)
    plt.ylim(-0.60, 0.60)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    
    # Deduplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    plt.savefig(f"{dir}/points.png")
    plt.close()

def barplot(data, std, save_dir, epochs=None, feature_labels=None,
            xlabel='Epoch', ylabel='Feature Value',
            title='Grouped Bar Plot with Standard Deviation',
            show=False):
    """
    Plots a grouped bar chart with error bars for standard deviation.

    Parameters:
        data (np.ndarray): A 2D array of shape (B, F) containing mean values, 
                           where B is the number of epochs and F is the number of features.
        std (np.ndarray): A 2D array of shape (B, F) containing standard deviations for each feature at each epoch.
        epochs (np.ndarray, optional): A 1D array of epoch indices. Defaults to np.arange(B).
        feature_labels (list of str, optional): Labels for each feature. Defaults to ['Feature 0', 'Feature 1', ..., 'Feature F-1'].
        xlabel (str, optional): Label for the x-axis. Defaults to 'Epoch'.
        ylabel (str, optional): Label for the y-axis. Defaults to 'Feature Value'.
        title (str, optional): Title of the plot. Defaults to 'Grouped Bar Plot with Standard Deviation'.
        show (bool, optional): Whether to display the plot immediately. Defaults to True.

    Returns:
        fig, ax: The matplotlib figure and axes objects.
    """
    B, F = data.shape
    if epochs is None:
        epochs = np.arange(B)
    if feature_labels is None:
        feature_labels = [f'Feature {i}' for i in range(F)]
    
    # Calculate bar width for grouped bars
    width = 0.8 / F
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars with error bars for each feature
    for i in range(F):
        offset = (i - F / 2) * width + width / 2
        ax.bar(epochs + offset, data[:, i], width=width,
               yerr=std[:, i], capsize=3, label=feature_labels[i])
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.legend(title='Features')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/per_feature_reconstruction_loss.png")
    
    if show:
        plt.show()
        
    return fig, ax

def stratified_split(data, onehot_labels, n_samples_per_class=None, random_state=None):
    """
    Splits the data into two parts:
      1. A balanced subset with an equal number of datapoints per class (stratified sample).
      2. The remaining data.
    
    This version assumes that labels are provided as one-hot vectors.
    If the requested n_samples_per_class is more than available in any class,
    it automatically reduces n_samples_per_class to the smallest class count.

    Args:
        data (np.ndarray): Array of data samples, shape (n_samples, ...).
        onehot_labels (np.ndarray): Array of one-hot labels, shape (n_samples, n_classes).
        n_samples_per_class (int, optional): Number of samples to take from each class.
            If None or too high, the function will use the minimum count among classes.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        remaining_data (np.ndarray): The remaining data after the balanced subset.
        remaining_labels (np.ndarray): The corresponding one-hot labels for remaining_data.
        balanced_data (np.ndarray): Stratified balanced subset of data.
        balanced_labels (np.ndarray): Corresponding one-hot labels for balanced_data.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Convert one-hot labels to integer labels for stratification
    int_labels = np.argmax(onehot_labels, axis=1)
    unique_classes = np.unique(int_labels)
    
    if len(unique_classes) != onehot_labels.shape[1]:
        raise ValueError(f"Expected {onehot_labels.shape[1]} classes, but found {len(unique_classes)} unique classes.")
    
    # Determine the minimum count available in any class
    min_count = min((int_labels == c).sum() for c in unique_classes)
    
    # Adjust n_samples_per_class if necessary
    if n_samples_per_class is None or n_samples_per_class > min_count:
        n_samples_per_class = min_count
        print("[INFO] Insuffienceint number of samples, Changing the number of samples to ", min_count)
    
    balanced_indices = []
    all_indices = set(range(len(int_labels)))
    
    # For each class, randomly select n_samples_per_class indices
    for c in unique_classes:
        class_indices = np.where(int_labels == c)[0]
        sampled_indices = np.random.choice(class_indices, size=n_samples_per_class, replace=False)
        balanced_indices.extend(sampled_indices)
    
    balanced_indices = np.array(balanced_indices)
    remaining_indices = np.array(list(all_indices - set(balanced_indices)))
    
    # Shuffle indices to avoid ordering bias
    np.random.shuffle(balanced_indices)
    np.random.shuffle(remaining_indices)
    
    balanced_data = data[balanced_indices]
    balanced_labels = onehot_labels[balanced_indices]
    remaining_data = data[remaining_indices]
    remaining_labels = onehot_labels[remaining_indices]
    
    return remaining_data, balanced_data, remaining_labels, balanced_labels

# StratifiedShuffleSplit implementation
def stratified_shuffle_split(data, onehot_labels, test_size=0.2, random_state=42):
    """
    Perform a stratified shuffle split and train a simple classifier.
    
    Args:
        data (np.ndarray): Feature data, shape (n_samples, n_features).
        onehot_labels (np.ndarray): One-hot encoded labels, shape (n_samples, n_classes).
        test_size (float): Proportion of data to use as test set.
        random_state (int): Seed for reproducibility.
    """
    # Convert one-hot labels to integer labels for StratifiedShuffleSplit
    int_labels = np.argmax(onehot_labels, axis=1)
    
    # Initialize StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    # Perform the split (returns indices)
    for train_index, test_index in sss.split(data, int_labels):
        X_train, X_test = data[train_index], data[test_index]
        y_train_onehot, y_test_onehot = onehot_labels[train_index], onehot_labels[test_index]
        y_train_int, y_test_int = int_labels[train_index], int_labels[test_index]
    
    # Print class distributions to verify stratification
    print("Training set class distribution:", np.bincount(y_train_int))
    print("Test set class distribution:", np.bincount(y_test_int))
    print("Training set size:", X_train.shape)
    print("Test set size:", X_test.shape)

    
    return X_train, X_test, y_train_onehot, y_test_onehot

# def barplot(data, save_dir):
#     B, F = data.shape
#     epochs = np.arange(B)
#     width = 0.8 / F
#     fig, ax = plt.subplots(figsize=(10, 6))
#     # Plot each feature as a set of bars, offset for each feature within an epoch
#     for i in range(F):
#         # Calculate the offset for feature i
#         offset = (i - F / 2) * width + width / 2
#         ax.bar(epochs + offset, data[:, i], width=width, label=f'Feature {i}')

#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Feature Value')
#     ax.set_title('Grouped Bar Plot of Features vs. Epochs')
#     # ax.legend(title='Features')
#     os.makedirs(save_dir, exist_ok=True)
#     plt.savefig(f"{save_dir}/per_feature_reconstruction_loss.png")
#     plt.close()

def create_dump_directory(path):
    str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    dump_dir = os.path.join(path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_{}'.format(str))
    os.makedirs(dump_dir, exist_ok=True)
    return dump_dir

def sample_gumbel_softmax(c_logit, training=True, tau=0.05):
    if training:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(c_logit) + 1e-10) + 1e-10)
        y = F.softmax((c_logit + gumbel_noise) / tau, dim=-1)
        return y
    else:
        return F.softmax((c_logit)/tau, dim=-1)
    
def logsumexp_stable(x, dim):
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    return max_val.squeeze() + torch.log(torch.sum(torch.exp(x - max_val), dim=dim))

def custom_soft_action_encoding(action, num_action=2, dim=24):
    if dim % num_action != 0:
        raise ValueError("Action encoding dimension must be divisible by {}".format(num_action))
    
    if torch.is_tensor(action):
        action_np = action.detach().cpu().numpy()
    else:
        action_np = np.array(action, dtype=np.float32)
    
    num_dims_per_action = dim // num_action
    
    def f(soft_labels):
        encoding = np.zeros((dim,), dtype=np.float32)
        for idx, label in enumerate(soft_labels):
            start_idx = idx * num_dims_per_action
            end_idx = start_idx + num_dims_per_action
            encoding[start_idx:end_idx] = label  # Fill the respective slice with the soft label value
        return encoding

    action_encoded = np.array([f(soft_label) for soft_label in action_np])
    return action_encoded

def compute_entropy(probs):
    """Compute entropy of a probability distribution."""
    probs = np.clip(probs, 1e-12, 1.0)  # Avoid log(0)
    return -np.sum(probs * np.log(probs), axis=1)

def get_data(datapath):
    return np.load(datapath, allow_pickle=True)

def anneal_coefficient(epoch, max_epoch, start_value, end_value, buffer_timestep, is_increase = True):
    """
    Anneals the coefficient with a buffer period where it stays at the start_value,
    and then linearly increases to the end_value after the buffer period.
    """
    if epoch < buffer_timestep:
        return start_value
    else:
        # Linearly anneal after buffer period
        adjusted_epoch = epoch - buffer_timestep
        annealing_duration = max_epoch - buffer_timestep
        if is_increase:
            return start_value + (end_value - start_value) * (adjusted_epoch / annealing_duration)
        else:
            return start_value - (start_value - end_value) * (adjusted_epoch / annealing_duration)

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
        return self.sentences[idx], idx
    
class TwoListDataset(Dataset):
    def __init__(self, list1, list2):
        self.list1 = list1
        self.list2 = list2
    
    def __len__(self):
        return len(self.list1)  # Assume both lists are of the same length
    
    def __getitem__(self, idx):
        return self.list1[idx], self.list2[idx]
    
class ThreeListDataset(Dataset):
    def __init__(self, list1, list2, list3):
        self.list1 = list1
        self.list2 = list2
        self.list3 = list3
    
    def __len__(self):
        return len(self.list1)  # Assume both lists are of the same length
    
    def __getitem__(self, idx):
        return self.list1[idx], self.list2[idx], self.list3[idx]
    
def normalize_data(data):
    """
    Normalize the data using Z-score normalization (mean = 0, std = 1).
    
    Args:
        data (np.array): The data to normalize (N x features).
        
    Returns:
        np.array: The normalized data.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = (data - mean) / (std + 1e-7)  # Small epsilon to prevent division by zero
    return normalized_data

def monte_carlo_entropy(alpha, mu, sigma, num_samples=10000):
    """
    Monte Carlo estimation of entropy for a mixture of Gaussians.
    
    Args:
        alpha (list): Weights of the Gaussian components. Must sum to 1.
        mu (list): Means of the Gaussian components.
        sigma (list): Standard deviations of the Gaussian components.
        num_samples (int): Number of samples to generate for Monte Carlo estimation.

    Returns:
        float: Estimated entropy of the mixture distribution.
    """
    M = len(alpha)  # Number of components in the mixture

    # Step 1: Sample component indices according to the mixture weights alpha
    component_indices = np.random.choice(M, size=num_samples, p=alpha)

    # Step 2: Generate samples from the corresponding Gaussians
    samples = np.array([np.random.normal(mu[i], sigma[i]) for i in component_indices])

    # Step 3: Compute the density of each sample under the full mixture distribution
    mixture_density = np.zeros(num_samples)
    for i in range(M):
        # Compute the contribution of the i-th Gaussian to the density of each sample
        component_density = alpha[i] * norm.pdf(samples, mu[i], sigma[i])
        mixture_density += component_density

    # Step 4: Compute the entropy using the Monte Carlo estimate
    entropy = -np.mean(np.log(mixture_density + 1e-10))  # Adding a small constant for numerical stability

    return entropy

def collect_random(env, dataset, load_model, num_samples=200):
    state, info = env.reset()
    # state = np.transpose(state['image'], (2, 0, 1))/255
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, truncated, terminated, _ = env.step(action)
        if not load_model:
            class_prob = calculate_probabilities(env.unwrapped.agent_pos, 
                                                 env.unwrapped.obs['image'], 
                                                 env.unwrapped.agent_dir, 
                                                 env.unwrapped.red_ball_loc, 
                                                 env.unwrapped.green_ball_loc)
        else:
            class_prob = 0
        # next_state = np.transpose(next_state['image'], (2, 0, 1))/255
        done = truncated + terminated
        dataset.add(np.transpose(state['image']/255, (2,0,1)), action, reward, np.transpose(next_state['image']/255, (2,0,1)), done, class_prob)
        state = next_state
        if done:
            state, info = env.reset()
            # state = np.transpose(state['image'], (2, 0, 1))/255
            
def write_video(frames, episode, dump_dir, frameSize = (224, 224)):
    os.makedirs(dump_dir, exist_ok=True)
    video = cv2.VideoWriter(dump_dir + '/{}.avi'.format(episode),cv2.VideoWriter_fourcc(*'DIVX'), 1, frameSize, isColor=True)
    for img in frames:
        video.write(img)
    video.release()
            
def test(env, agent, save_dir, n_episode=5):
    for i_episode in range(int(n_episode)):
        frame_array = []
        state, info = env.reset()
        frame = env.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_array.append(frame)
        done = False
        score = 0
        episode_step = 0
        while not done:
            action = agent.get_action(np.expand_dims(state['image'], axis=0))
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated + truncated
            frame = env.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_array.append(frame)
            state = next_state

        write_video(frame_array, i_episode, save_dir)
        
def enumerate_discrete(x, y_dim):
    """
    Generates a `torch.Tensor` of size batch_size x n_labels of
    the given label.

    Example: generate_label(2, 1, 3) #=> torch.Tensor([[0, 1, 0],
                                                       [0, 1, 0]])
    :param x: tensor with batch size to mimic
    :param y_dim: number of total labels
    :return variable
    """
    def batch(batch_size, label):
        labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
        y = torch.zeros((batch_size, y_dim))
        y.scatter_(1, labels, 1)
        return y.type(torch.LongTensor)

    batch_size = x.size(0)
    generated = torch.cat([batch(batch_size, i) for i in range(y_dim)])

    if x.is_cuda:
        generated = generated.cuda()

    return Variable(generated.float())

def index_to_one_hot(index, num_classes):
    """
    Converts an index to a one-hot encoded vector.

    Parameters:
        index (int): The index to be converted.
        num_classes (int): The total number of classes.

    Returns:
        numpy.ndarray: A one-hot encoded vector.
    """
    one_hot = np.zeros(num_classes)
    one_hot[index] = 1
    return one_hot


def onehot(k):
    """
    Converts a number to its one-hot or 1-of-k representation
    vector.
    :param k: (int) length of vector
    :return: onehot function
    """
    def encode(label):
        y = torch.zeros(k)
        if label < k:
            y[label] = 1
        return y
    return encode


def log_sum_exp(tensor, dim=-1, sum_op=torch.sum):
    """
    Uses the LogSumExp (LSE) as an approximation for the sum in a log-domain.
    :param tensor: Tensor to compute LSE over
    :param dim: dimension to perform operation over
    :param sum_op: reductive operation to be applied, e.g. torch.sum or torch.mean
    :return: LSE
    """
    max, _ = torch.max(tensor, dim=dim, keepdim=True)
    return torch.log(sum_op(torch.exp(tensor - max), dim=dim, keepdim=True) + 1e-8) + max