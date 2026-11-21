import hydra
import torch
import random
import wandb
from tqdm import tqdm
from helper_functions.utils import *
# from vae.vae import GMVAE
import torch.optim as optim
from dqn.dqn import DQNAgent
from sac_agent.agent import SAC
from partedvae.models import VAE
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from helper_functions.get_llm_output import GetLLMGoals
from train_captioner import FeatureToEmbeddingModel
from sentence_transformers import SentenceTransformer
from imagination.imagination_net import ImaginationNet
from architectures.m2_vae.dgm import DeepGenerativeModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_imagination_net(config,
                          env,
                          agent, 
                          dataloader, 
                          checkpoint_interval, 
                          checkpoint_dir,
                          vae_model,
                          num_goals):
    imagination_net = ImaginationNet(env = env,
                                     config = config,
                                     num_goals = num_goals,
                                     agent = agent,
                                     vae = vae_model).to(device)
    wandb.watch(imagination_net)
    #Creating the optimizer
    optimizer = optim.Adam(imagination_net.parameters(), lr=config['lr'])
    # nn_utils.clip_grad_norm_(imagination_net.parameters(), max_norm=config['max_gradient'])
    epoch_bar = tqdm(range(config.Imagination_Network.epoch), desc="Training Progress", unit="epoch")
    for epoch in epoch_bar:
        total_loss = 0
        total_class_loss = 0
        total_policy_loss = 0
        total_proximity_loss = 0
        for state, ind in dataloader:
            state = state.float().to(device)
            optimizer.zero_grad()
            imagined_state = imagination_net(state)
            loss, class_loss, policy_loss, proximity_loss, weight  = imagination_net.compute_loss(state, imagined_state, epoch)
            loss.backward()
            optimizer.step()
            # Accumulate losses
            total_loss += loss.item()
            total_class_loss += class_loss.item()
            total_policy_loss += policy_loss.item()
            total_proximity_loss += proximity_loss.item()
            epoch_bar.set_postfix({'Loss': loss.item(), 'Class loss': class_loss.item(), 'Policy loss': policy_loss.item(), 'Proximity loss': proximity_loss.item()})
            
        average_loss = total_loss #/ len(dataloader)
        average_class_loss = total_class_loss #/ len(dataloader)
        average_policy_loss = total_policy_loss #/ len(dataloader)
        average_proximity_loss = total_proximity_loss #/ len(dataloader)
        # current_lr = scheduler.get_last_lr()[0]
        wandb.log({"Total loss" : average_loss,
                   "Class loss" : average_class_loss,
                   "Policy loss" : average_policy_loss,
                   "Proximity loss" : average_proximity_loss,
                   "Weight" : weight}, step = epoch)
        # Save checkpoint at specified intervals
        if epoch % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'imagination_net_epoch_{epoch}.tar')
            imagination_net.save(checkpoint_path)
        # print(f"Epoch {epoch}/{epochs} - Loss: {average_loss:.4f}, Recon Loss: {average_recon:.4f}, KL Divergence: {average_kl:.4f}, KL Weight: {kl_weight:.4f}")
    imagination_net.save('models/imagination_net.pth')
    print("Training complete.")

@hydra.main(version_base=None, config_path="config", config_name="master_config")
def main(args: DictConfig) -> None:
    # Log the configuration
    # wandb.config.update(OmegaConf.to_container(args, resolve=True))
    #Loading the dataset
    # states = get_data(f'{args.Imagination_General.datapath}/{args.General.env}/states.pkl')
    next_states = get_data(f'{args.Imagination_General.datapath}/{args.General.env}/data.pkl')
    next_states = [i/255 for i in next_states]
    # captions = get_data(f'{args.Imagination_General.datapath}/{args.General.env}/caption_encoded.pkl')
    # Initialize dataset and dataloader
    dataset = Dataset(next_states)
    dataloader = DataLoader(dataset, batch_size=args.Imagination_Network.batch_size, shuffle=True)
    if args.General.env ==  "SimplePickup":
        from env.env import SimplePickup
        env = SimplePickup(max_steps=args.Imagination_General.max_ep_len, agent_view_size=5, size=7)
    # dataset = normalize_data(dataset)
    # Device configuration
    # llm_goals = GetLLMGoals()
    # goals = llm_goals.llmGoals('')
    # sentenceEncoder = SentenceTransformer(args.Imagination_General.encoder_model, device=device)
    # goals_encoded = sentenceEncoder.encode(goals, convert_to_tensor=True, device=device)
    if args.Imagination_General.agent == 'dqn':
        agent = DQNAgent(env, 
                         args.General, 
                         args.policy_config, 
                         args.policy_network_cfg, 
                         args.policy_network_cfg, '')
    else:
        agent = SAC(args,
                    input_dim = env.observation_space['image'].shape[-1],
                    action_size = env.action_space.n,
                    device=device,
                    buffer_size = args.Imagination_General.buffer_size)
    
    #Loading the pretrained weight of sac agent.
    agent.load_params(args.Imagination_General.agent_checkpoint)
    #Freezing the SAC agent weight to prevent updating
    for params in agent.parameters():
        params.requires_grad = False
        
    # captioner = FeatureToEmbeddingModel(feature_dim=2*env.observation_space.shape[0], 
    #                                     latent_dim = sentenceEncoder.get_sentence_embedding_dimension())
    # captioner.to(device)
    # captioner.load_state_dict(torch.load(args.Imagination_General.captioner_path, map_location=device))
    
    # for params in captioner.parameters():
    #     params.requires_grad = False
    
    model = DeepGenerativeModel([args.M2_Network.input_dim, args.M2_General.y_dim, args.M2_Network.h_dim, \
                                 args.M2_Network.latent_dim, args.M2_Network.classifier_hidden_dim, args.M2_Network.feature_encoder_channel_dim], \
                                 args.M2_Network.label_loss_weight).to(device)
    model.load(model_dir)
    model.to(device)
    model.eval()
    
    for params in vae.parameters():
        params.requires_grad = False
    
    # Create data directory if it doesn't exist
    model_dir = create_dump_directory("models/imagination_net")
    os.makedirs(model_dir, exist_ok=True)
    
    #Train imagination Net
    train_imagination_net(config = args, 
                          env= env,
                          agent = agent, 
                          dataloader = dataloader, 
                          checkpoint_interval = args.Imagination_Network.checkpoint_interval, 
                          checkpoint_dir = model_dir,
                          vae_model = vae,
                          num_goals = args.Imagination_General.num_goals)
    
if __name__ == "__main__":
    wandb.init(project="Imagination-net_training")
    main()
    