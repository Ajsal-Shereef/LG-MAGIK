# MAGIK: Mapping to Analogous Goals via Imagination-enabled Knowledge

This directory contains the implementation of **MAGIK**, a zero-shot policy transfer framework for reinforcement learning based on a semi-supervised M2 Variational Autoencoder (`DeepGenerativeModel`). MAGIK enables agents to reuse policies across known object classes by generating source-aligned observations through discrete one-hot class imagination, without requiring interaction with the target environment.

## 🧠 Key Idea

MAGIK leverages a semi-supervised M2-VAE to disentangle task-agnostic spatial geometry ($z$) and task-specific object class presence ($c$). At test time, it classifies the incoming target observation into one of $K$ discrete one-hot classes and re-decodes the latent spatial vector $z$ conditioned on the analogous source class vector $c_{\text{source}}$, allowing zero-shot policy reuse.

MAGIK is implemented for:
- **`MiniGrid/`**: Original MAGIK MiniGrid implementation (`SimplePickup`)
- **`Reacher/`**: Original MAGIK Reacher implementation
- **`MiniWorld/`**: MAGIK baseline for **Case 3 (`target3`) of MiniWorld** as a baseline to **PRISM**

---

## MiniWorld (Case 3 Baseline to PRISM)

In MiniWorld:
- **Source Task (`task_mode="source"`)**: Pick the **blue box** (+10 reward) and avoid the **green ball** (-10 penalty) in a room with grass floor and concrete walls.
- **Case 3 Target Task (`task_mode="target3"`)**: Pick the **green ball** (+10 reward) and avoid the **blue box** (-10 penalty) in the same canonical room.

Because Case 3 involves a role reversal between two **known** objects from the source environment (`green ball` and `blue box`), MAGIK defines 4 discrete one-hot observation classes ($y \in \{0, 1\}^4$):
- **Class 0 (`[1, 0, 0, 0]`)**: Only `green ball` visible (Case 3 target / Source distractor)
- **Class 1 (`[0, 1, 0, 0]`)**: Only `blue box` visible (Source target / Case 3 distractor)
- **Class 2 (`[0, 0, 1, 0]`)**: Both `green ball` and `blue box` visible
- **Class 3 (`[0, 0, 0, 1]`)**: No objects visible in current view

At test time in Case 3 (`target3`), MAGIK classifies each $80 \times 80 \times 3$ observation and transforms it into the source policy's perceptual space:
- `Class 0` (`green ball` only) $\to$ transform with `[0, 1, 0, 0]` (imagines a `blue box` so the source PPO policy picks it)
- `Class 1` (`blue box` only) $\to$ transform with `[0, 0, 0, 1]` (imagines an empty view so the source PPO policy ignores the `blue box` distractor)
- `Class 2` (both objects visible) $\to$ two-stage transformation `[[1, 0, 0, 0], [0, 1, 0, 0]]` (first isolates the `green ball`, then transforms it into a `blue box`)
- `Class 3` (no objects visible) $\to$ unchanged

### Usage (from project root)

#### 1. (Optional) Prepare or Collect MiniWorld MAGIK Dataset
If `data/MiniWorld` (e.g. `data/MiniWorld/agent/images`) already exists from PRISM training, `m2_vae.py` automatically extracts the 4-way one-hot class labels and caches `data/MiniWorld/Random/data.pkl` and `class_prob.pkl`. You can also explicitly prepare or collect the dataset via:
```bash
python baselines/MAGIK/MiniWorld/helper_functions/collect_vae_training_data.py
```

#### 2. Train MAGIK M2-VAE on MiniWorld
```bash
python baselines/MAGIK/MiniWorld/m2_vae.py General.seed=42
```
Saves the trained M2-VAE checkpoint to `model_weights/MiniWorld/MAGIK/m2_vae/seed_42/model.pt`.

#### 3. Evaluate MAGIK Baseline on MiniWorld Case 3 (`target3`)
Run across all evaluation seeds (`42, 123, 456, 789, 1024`):
```bash
python baselines/MAGIK/MiniWorld/test_imagination_net.py Test.seed=42
```
Or evaluate via PRISM's unified `test_imagination.py` entrypoint:
```bash
python test_imagination.py env=MiniWorld env.task_mode=target3 mapping_strategy=magik seed=42
```
Results and rollout GIFs are automatically saved to `Results/baselines/magik.md`, `Results/baselines/magik_cache.json`, and `result/MAGIK_PPO/MiniWorld/target3/transfer/`.
