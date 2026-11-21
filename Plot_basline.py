import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Set publication quality styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 14.5,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.4,
    'lines.linewidth': 2.2,
})

def format_k_steps(x, pos):
    if x >= 1000:
        return f'{int(x/1000)}k'
    return f'{int(x)}'

CONFIGS = [
    # --- MiniWorld ---
    {
        'env_name': 'MiniWorld',
        'target_id': 'Target1',
        'csv': 'Results/baselines/Fintune_learning_data/MiniWorld/Target1.csv',
        'algo': 'PPO',
        'title': 'MiniWorld: Case 1 (Novel Target)',
        'prism_mean': 11.4159,
        'prism_std': 0.7653,
        'pdf_name': 'Fine_tune_MiniWorld_case_1.pdf',
        'ema_span': 35,
        'y_min': -3.0,
        'y_max': 15.5,
        'y_label': 'Episode Return',
        'legend_loc': 'lower right',
    },
    {
        'env_name': 'MiniWorld',
        'target_id': 'Target2',
        'csv': 'Results/baselines/Fintune_learning_data/MiniWorld/Target2.csv',
        'algo': 'PPO',
        'title': 'MiniWorld: Case 2 (Compound Shift)',
        'prism_mean': 11.7906,
        'prism_std': 0.6431,
        'pdf_name': 'Fine_tune_MiniWorld_case_2.pdf',
        'ema_span': 35,
        'y_min': -3.0,
        'y_max': 15.5,
        'y_label': 'Episode Return',
        'legend_loc': 'lower right',
    },
    {
        'env_name': 'MiniWorld',
        'target_id': 'Target3',
        'csv': 'Results/baselines/Fintune_learning_data/MiniWorld/Target3.csv',
        'algo': 'PPO',
        'title': 'MiniWorld: Case 3 (Distractor Shift)',
        'prism_mean': 9.9300,
        'prism_std': 1.3879,
        'pdf_name': 'Fine_tune_MiniWorld_case_3.pdf',
        'ema_span': 35,
        'y_min': -12.0,
        'y_max': 14.5,
        'y_label': 'Episode Return',
        'legend_loc': 'lower right',
    },

    # --- MiniGrid (SimplePickup) ---
    {
        'env_name': 'SimplePickup',
        'target_id': 'Target1',
        'csv': 'Results/baselines/Fintune_learning_data/SimplePickup/Target1.csv',
        'algo': 'DQN',
        'title': 'MiniGrid: Case 1 (Novel Target)',
        'prism_mean': 0.9400,
        'prism_std': 0.0800,
        'pdf_name': 'Fine_tune_MiniGrid_case_1.pdf',
        'ema_span': 60,
        'y_min': -0.2,
        'y_max': 1.18,
        'y_label': 'Score',
        'legend_loc': 'lower right',
    },
    {
        'env_name': 'SimplePickup',
        'target_id': 'Target2',
        'csv': 'Results/baselines/Fintune_learning_data/SimplePickup/Target2.csv',
        'algo': 'DQN',
        'title': 'MiniGrid: Case 2 (Compound Shift)',
        'prism_mean': 0.9200,
        'prism_std': 0.0980,
        'pdf_name': 'Fine_tune_MiniGrid_case_2.pdf',
        'ema_span': 60,
        'y_min': -0.2,
        'y_max': 1.18,
        'y_label': 'Score',
        'legend_loc': 'lower right',
    },
    {
        'env_name': 'SimplePickup',
        'target_id': 'Target3',
        'csv': 'Results/baselines/Fintune_learning_data/SimplePickup/Taregt3.csv',
        'algo': 'DQN',
        'title': 'MiniGrid: Case 3 (Goal Inversion)',
        'prism_mean': 0.9400,
        'prism_std': 0.0800,
        'pdf_name': 'Fine_tune_MiniGrid_case_3.pdf',
        'ema_span': 60,
        'y_min': -0.2,
        'y_max': 1.18,
        'y_label': 'Score',
        'legend_loc': 'lower right',
    },

    # --- Fragile Object Manipulation (PickEnv) ---
    {
        'env_name': 'PickEnv',
        'target_id': 'Target',
        'csv': 'Results/baselines/Fintune_learning_data/PickEnv/Target.csv',
        'algo': 'SAC',
        'title': 'Fragile Object Manipulation',
        'prism_mean': 48.1763,
        'prism_std': 6.6770,
        'pdf_name': 'Fine_tune_manipulation.pdf',
        'ema_span': 80,
        'y_min': -90.0,
        'y_max': 75.0,
        'y_label': 'Episode Return',
        'legend_loc': 'lower right',
    },

    # --- MiniGrid Relational ---
    {
        'env_name': 'MiniGridRelational',
        'target_id': 'Target1',
        'csv': 'Results/baselines/Fintune_learning_data/MiniGridRelational/Target1.csv',
        'algo': 'DQN',
        'title': 'Relational Target 1 (Adjacency)',
        'prism_mean': 10.1800,
        'prism_std': 0.7547,
        'pdf_name': 'Fine_tune_Relational_target_1.pdf',
        'ema_span': 50,
        'y_min': -0.5,
        'y_max': 12.5,
        'y_label': 'Episode Return',
        'legend_loc': 'lower right',
    },
    {
        'env_name': 'MiniGridRelational',
        'target_id': 'Target2',
        'csv': 'Results/baselines/Fintune_learning_data/MiniGridRelational/Target2.csv',
        'algo': 'DQN',
        'title': 'Relational Target 2 (Symmetry)',
        'prism_mean': 9.8000,
        'prism_std': 0.7483,
        'pdf_name': 'Fine_tune_Relational_target_2.pdf',
        'ema_span': 50,
        'y_min': -0.5,
        'y_max': 12.5,
        'y_label': 'Episode Return',
        'legend_loc': 'lower right',
    },
    {
        'env_name': 'MiniGridRelational',
        'target_id': 'Target3',
        'csv': 'Results/baselines/Fintune_learning_data/MiniGridRelational/Target3.csv',
        'algo': 'DQN',
        'title': 'Relational Target 3 (Distance-2)',
        'prism_mean': 9.4600,
        'prism_std': 0.8800,
        'pdf_name': 'Fine_tune_Relational_target_3.pdf',
        'ema_span': 50,
        'y_min': -0.5,
        'y_max': 12.5,
        'y_label': 'Episode Return',
        'legend_loc': 'lower right',
    },
    {
        'env_name': 'MiniGridRelational',
        'target_id': 'Target4',
        'csv': 'Results/baselines/Fintune_learning_data/MiniGridRelational/Target4.csv',
        'algo': 'DQN',
        'title': 'Relational Target 4 (Offset Symmetry)',
        'prism_mean': 9.6000,
        'prism_std': 1.2000,
        'pdf_name': 'Fine_tune_Relational_target_4.pdf',
        'ema_span': 50,
        'y_min': -0.5,
        'y_max': 12.5,
        'y_label': 'Episode Return',
        'legend_loc': 'lower right',
    },
    {
        'env_name': 'MiniGridRelational',
        'target_id': 'Target5',
        'csv': 'Results/baselines/Fintune_learning_data/MiniGridRelational/Target5.csv',
        'algo': 'DQN',
        'title': 'Relational Target 5 (Multi-Pair)',
        'prism_mean': 19.2000,
        'prism_std': 1.0807,
        'pdf_name': 'Fine_tune_Relational_target_5.pdf',
        'ema_span': 50,
        'y_min': -0.5,
        'y_max': 23.5,
        'y_label': 'Episode Return',
        'legend_loc': 'lower right',
    },
]

def generate_figure(cfg, output_pdf, output_png=None):
    df = pd.read_csv(cfg['csv'])
    seeds = [1024, 789, 456, 123, 42]
    max_step = 50000
    grid = np.linspace(0, max_step, 501)
    
    seed_curves = []
    for s in seeds:
        col = [c for c in df.columns if f'seed_{s}' in c and 'episode_reward' in c and not c.endswith('__MIN') and not c.endswith('__MAX')][0]
        data = df[['Step', col]].dropna().sort_values('Step')
        smoothed = data[col].ewm(span=cfg['ema_span'], adjust=False).mean()
        interp = np.interp(grid, data['Step'], smoothed)
        seed_curves.append(interp)
        
    arr = np.array(seed_curves)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    
    fig, ax = plt.subplots(figsize=(5.4, 4.0), dpi=300)
    
    # Color scheme
    ft_color = '#1f77b4'       # Deep professional blue
    prism_color = '#d62728'    # Vivid crimson
    
    # 1. Fine-Tuning Baseline Curve with shaded std region
    ax.plot(grid, mean, color=ft_color, lw=2.4, label=f'{cfg["algo"]} Fine-Tuning (mean ± std)')
    ax.fill_between(grid, mean - std, mean + std, color=ft_color, alpha=0.18)
    
    # 2. PRISM Zero-Shot performance
    p_mean = cfg['prism_mean']
    p_std = cfg['prism_std']
    label_prism = f'PRISM Zero-Shot: {p_mean:.2f} ± {p_std:.2f}' if p_mean >= 1.0 else f'PRISM Zero-Shot: {p_mean:.2f} ± {p_std:.2f}'
    ax.axhline(p_mean, color=prism_color, linestyle='--', lw=2.4, label=label_prism)
    ax.axhspan(p_mean - p_std, p_mean + p_std, color=prism_color, alpha=0.14)
    
    # Aesthetics & Titles
    ax.set_title(cfg['title'], fontweight='bold', pad=9)
    ax.set_xlabel('Training Steps', fontweight='medium')
    ax.set_ylabel(cfg['y_label'], fontweight='medium')
    ax.xaxis.set_major_formatter(FuncFormatter(format_k_steps))
    ax.set_xlim(0, max_step)
    if 'y_min' in cfg and 'y_max' in cfg:
        ax.set_ylim(cfg['y_min'], cfg['y_max'])
        
    ax.grid(True, linestyle='--', alpha=0.45, color='#c8c8c8')
    ax.set_axisbelow(True)
    
    # Clean legend
    ax.legend(loc=cfg.get('legend_loc', 'lower right'),
              framealpha=0.92, facecolor='white', edgecolor='#d0d0d0', fontsize=10.5)
    
    plt.tight_layout()
    if output_pdf:
        plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    if output_png:
        plt.savefig(output_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    os.makedirs('AAMAS/Figures', exist_ok=True)
    os.makedirs('Results/plots/fine_tune', exist_ok=True)
    
    for cfg in CONFIGS:
        pdf_path = os.path.join('AAMAS/Figures', cfg['pdf_name'])
        png_path = os.path.join('Results/plots/fine_tune', cfg['pdf_name'].replace('.pdf', '.png'))
        generate_figure(cfg, output_pdf=pdf_path, output_png=png_path)
        print(f'Generated: {pdf_path} and {png_path}')

if __name__ == '__main__':
    main()
