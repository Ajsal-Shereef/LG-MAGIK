import subprocess
print("Starting training with features...")
subprocess.run([
    "/home/s222147455/.conda/lg_magik/bin/python", "train_agent.py",
    "env=PandaGym",
    "env.observation_mode=feature",
    "env.total_timestep=500",
    "number_data_to_collect=100"
])
print("Done.")
