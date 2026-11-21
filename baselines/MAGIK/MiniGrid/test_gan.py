import os
import cv2
import hydra
import torch
import numpy as np
import gymnasium as gym

from dqn.dqn import DQNAgent
from sac_agent.agent import SAC
from omegaconf import DictConfig
from helper_functions.utils import *
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from helper_functions.visualize import Visualizer
from architectures.gan.gan import SSGAN

is_agent = False
policy_mode = 'single_objects'
num_test_env = 10
target_mapping = {"Red" : 0, "Green" : 1, "Both" : 2, "Neither" : 3}

@hydra.main(version_base=None, config_path="config", config_name="gan")
def main(args: DictConfig) -> None:
    if args.General.env ==  "SimplePickup":
        from env.env import MultiObjectMiniGridEnv
        env = MultiObjectMiniGridEnv(max_steps=20, agent_view_size=5, size=7, \
                                     reward_object="Red_ball", is_single_object = False)
        from minigrid.wrappers import RGBImgObsWrapper, RGBImgPartialObsWrapper
        env = RGBImgPartialObsWrapper(env)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if args.Test.agent == 'dqn':
        agent = DQNAgent(env, 
                        args.General, 
                        args.policy_config, 
                        args.policy_network_cfg, 
                        args.policy_network_cfg, '')
    else:
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
        
    model = SSGAN(args.Network, 0, device).to(device)
    model.load(args.Test.gan_checkpoint)
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
    true_classes = []
    predicted_classes = []
    all_misclassified = []
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
                        if torch.argmax(model.classify(x.unsqueeze(0))[-1]) == 0:
                            transformations = [torch.tensor([0, 1, 0, 0])]
                            need_tranformation = True
                        else:
                            need_tranformation = False
                    else:
                        cls = torch.argmax(model.classify(x.unsqueeze(0))[0])
                        true_class = np.argmax(env.get_class())
                        true_classes.append(true_class)
                        predicted_classes.append(cls.item())
                        if true_class != cls.item():
                            all_misclassified.append([env.get_frame(40), cls.item(), true_class])
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
                            # imagined_state = torch.round(imagined_state)
                            # o_obs_gen = reverse_preprocess_observation(model(imagined_state)[0].squeeze(0).detach().cpu().numpy(),5,5)
                            # cv2.imwrite("frame.png", cv2.cvtColor(env.render_partial_view_from_features(o_obs_gen['image'], o_obs_gen['direction'], tile_size=100), cv2.COLOR_BGR2RGB))
                        action = agent.get_action(imagined_state)
                        imagined_state_numpy = imagined_state.squeeze().detach().cpu().numpy()
                        # r_obs_gen = reverse_preprocess_observation(imagined_state_numpy, 5, 5)
                        partial_view_gen = (np.transpose(imagined_state_numpy, (1,2,0))*255).astype('uint8')
                        # gen_image_class = torch.argmax(model(imagined_state)[-1]).detach().cpu().numpy()
                        # org_image_class = torch.argmax(model(x.unsqueeze(0))[-1]).detach().cpu().numpy()
                        # cv2.putText(partial_view_gen, f'Action: {str(action)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        # cv2.putText(partial_view_gen, f'Class: {str(gen_image_class)}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        partial_view = state['image']
                        # partial_view = env.render_partial_view_from_features(r_obs['image'], r_obs['direction'], tile_size=100)
                        # cv2.putText(partial_view, f'Action: {str(agent.get_action(np.expand_dims(state, axis=0)))}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        # cv2.putText(partial_view, f'Class: {str(org_image_class)}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else:
                        action = agent.get_action(x.unsqueeze(0))
                        # r_obs = state
                        # partial_view = env.render_partial_view_from_features(r_obs['image'], r_obs['direction'], tile_size=100)
                        partial_view_gen = state['image']
                        partial_view = partial_view_gen
                        # org_image_class = torch.argmax(model(x.unsqueeze(0))[-1]).detach().cpu().numpy()
                        # cv2.putText(partial_view, f'Action: {str(action)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        # cv2.putText(partial_view, f'Class: {str(org_image_class)}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        # cv2.putText(partial_view_gen, f'Action: {str(action)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        # cv2.putText(partial_view_gen, f'Class: {str(org_image_class)}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
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

    if not is_agent:
        data_dir = f'visualizations/gan'
        os.makedirs(data_dir, exist_ok=True)
        viz = Visualizer(model, device, root=f'{data_dir}/')
        datasets = get_data(f'{args.General.datapath}/{args.General.env}/SAC/data.pkl')
        data = Dataset(datasets)
        data_loader_recon = DataLoader(data, batch_size=500, shuffle=True)
        for batch in data_loader_recon:
            break
        viz.reconstructions(batch[0])
        data_loader_swap = DataLoader(data, batch_size=10, shuffle=True)
        for batch in data_loader_swap:
            break
        viz.swap(images=batch[0], device=device)

        # Print classification summary after all episodes
        print("\nClassification Summary:")
        print(classification_report(true_classes, predicted_classes, target_names=target_mapping.keys()))

        visualisation_dir = 'visualizations'
        # === Display Misclassified Images in a Grid ===
        # Display up to 25 misclassified images
        n_display = min(25, len(all_misclassified))
        if n_display > 0:
            n_cols = 5
            n_rows = (n_display + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
            axes = axes.flatten()
            for i in range(n_display):
                image, pred, true_label = all_misclassified[i]
                # If image is channel-first (C, H, W) then convert to (H, W, C)
                if image.shape[0] == 3 or image.shape[0] == 1:
                    image_disp = np.transpose(image, (1, 2, 0))
                else:
                    image_disp = image
                axes[i].imshow(image_disp)
                axes[i].axis('off')
                axes[i].set_title(f"Pred: {pred}\nTrue: {true_label}", fontsize=10)
            # Turn off any unused subplots
            for j in range(n_display, len(axes)):
                axes[j].axis('off')
            plt.tight_layout()
            plt.savefig(f"{visualisation_dir}/misclassified.png")
        else:
            print("No misclassified samples to display.")

    

if __name__ == "__main__":
    print("*******************************************************************")
    if is_agent:
        print("Selected testing mode is: SAC AGENT")
    else:
        print("Selected testing mode is: IMAGINATION NET")
    print("*******************************************************************")
    main()
    