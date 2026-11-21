import os
import cv2
import hydra
import torch
import numpy as np
import gymnasium as gym

from sac_agent.agent import SAC
from omegaconf import DictConfig
from helper_functions.utils import *
from torch.utils.data import DataLoader
from helper_functions.visualize import Visualizer
from architectures.m2_vae.dgm import DeepGenerativeModel

is_agent = False
policy_mode = 'single_objects'
num_test_env = 10

@hydra.main(version_base=None, config_path="config", config_name="SimplePickup")
def main(args: DictConfig) -> None:
    if args.General.env ==  "SimplePickup":
        from env.env import MultiObjectMiniGridEnv
        env = MultiObjectMiniGridEnv(max_steps=20, agent_view_size=5, size=7, \
                                     reward_object="Red_ball", is_single_object = False)
        from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper
        env = RGBImgPartialObsWrapper(env)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    agent = SAC(input_dim = env.observation_space['image'].shape[-1],
                action_size = env.action_space.n,
                fc_hidden_size = 576,
                device=device,
                buffer_size = 10_000)
    #Loading the pretrained weight of sac agent.
    agent.load_params(args.Test.agent_checkpoint)
    agent.eval()
    #Freezing the SAC agent weight to prevent updating
    for params in agent.parameters():
        params.requires_grad = False
        
    model = DeepGenerativeModel(args.Network).to(device)
    model.load(args.Test.vae_checkpoint)
    model.to(device)
    model.eval()
    
    for params in model.parameters():
        params.requires_grad = False
        
    # Create data directory if it doesn't exist
    if is_agent:
        video_dir = "Videos/Original"
    else:
        video_dir = "Videos/Imagination"
    cumulative_green_ball_picked = 0
    cumulative_red_ball_picked = 0
    for i in range(0, num_test_env):
        frame_arrays = []
        partial_view_array = []
        state, info = env.reset()
        frame = env.unwrapped.get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_arrays.append(frame)
        episode_steps = 0
        while True:
            p_agent_loc = env.unwrapped.agent_pos
            x = torch.tensor(np.transpose(state['image']/255, (2,0,1))).to(device).float()
            # p_state = env.get_unprocesed_obs()
            if is_agent:
                with torch.no_grad():
                    action = agent.get_action(x)
            else:
                with torch.no_grad():
                    if policy_mode == 'all_objects':
                        if torch.argmax(model(x.unsqueeze(0))[-1]) == 0:
                            transformations = [torch.tensor([0, 1, 0, 0])]
                            need_tranformation = True
                        else:
                            need_tranformation = False
                    else:
                        cls = torch.argmax(model(x.unsqueeze(0))[-1])
                        if  cls == 0:
                            transformations = [torch.tensor([0, 1, 0, 0])]
                            need_tranformation = True
                        elif cls == 1:
                            transformations = [torch.tensor([0, 0, 0, 1])]
                            need_tranformation = True
                        elif cls == 2:
                            transformations = [torch.tensor([1, 0, 0, 0]), torch.tensor([0, 1, 0, 0])]
                            need_tranformation = True
                        else:
                            need_tranformation = False
                    if need_tranformation:
                        imagined_state = x.unsqueeze(0)
                        for transformation in transformations:
                            imagined_state = model.generate(imagined_state, transformation.to(device).float().unsqueeze(0))
                        action = agent.get_action(imagined_state)
                        imagined_state_numpy = imagined_state.squeeze().detach().cpu().numpy()
                        # r_obs_gen = reverse_preprocess_observation(imagined_state_numpy, 5, 5)
                        partial_view_gen = (np.transpose(imagined_state_numpy, (1,2,0))*255).astype('uint8')
                        partial_view = state['image']
                    else:
                        action = agent.get_action(x.unsqueeze(0))
                        # r_obs = state
                        # partial_view = env.render_partial_view_from_features(r_obs['image'], r_obs['direction'], tile_size=100)
                        partial_view_gen = state['image']
                        partial_view = partial_view_gen
                    # Concatenate tensors side by side (along the width)
                    concatenated_image = np.concatenate((partial_view_gen, partial_view), axis=1)  # Shape: (3, 40, 80)
                    # Convert PyTorch tensor to NumPy array for OpenCV (HWC format)
                    # partial_frame = (concatenated_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)  # (40, 80, 3)
                    partial_view_array.append(cv2.cvtColor(concatenated_image, cv2.COLOR_BGR2RGB))
            next_state, reward, terminated, truncated, _ = env.step(action)
            frame = env.get_frame()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_arrays.append(frame)
            done = terminated + truncated
            state = next_state
            episode_steps += 1
            if env.unwrapped.picked == 'Red_ball':
                cumulative_red_ball_picked += 1
            elif env.unwrapped.picked == 'Green_ball':
                cumulative_green_ball_picked += 1
            if done:
                break
        if not is_agent:
            if policy_mode == 'all_objects':
                dir = f'{video_dir}/All_object/'
            else:
                dir = f'{video_dir}/Single_object/'
        else:
            dir = video_dir
        write_video(frame_arrays, f'{str(i)}_full_view', f'{dir}/Full_view')
        if not is_agent:
            write_video(partial_view_array, f'{str(i)}_partial_view', f'{dir}/Partial_view', (80,40))

    print(f"Number of green ball picked {cumulative_green_ball_picked} out of {num_test_env}")
    print(f"Number of red_ball picked {cumulative_red_ball_picked} out of {num_test_env}")


    data_dir = f'visualizations/m2_vae'
    viz = Visualizer(model, device, root=f'{data_dir}/')
    datasets = get_data(f'{args.General.datapath}/{args.General.env}/Random/data.pkl')
    data = Dataset(datasets)
    data_loader = DataLoader(data, batch_size=10, shuffle=True)
    for batch in data_loader:
        break
    viz.swap(images=batch[0], device=device)

    

if __name__ == "__main__":
    print("*******************************************************************")
    if is_agent:
        print("Selected testing mode is: SAC AGENT")
    else:
        print("Selected testing mode is: IMAGINATION NET")
    print("*******************************************************************")
    main()
    