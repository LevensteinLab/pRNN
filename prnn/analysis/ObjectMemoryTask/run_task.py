import matplotlib.pyplot as plt
import wandb

from prnn.utils.predictiveNet import PredictiveNet
from prnn.analysis.ObjectMemoryTask import ObjectMemoryTask
from prnn.utils.eg_utils import get_device

NET_NAME = "thRNN18x18thRNN_5win--s8"
PACKAGE = "farama-minigrid"
ENV_NAME = "MiniGrid-LRoom_Goal-18x18-v0"
RESULTS_SAVE_FOLDER = "results"
DEVICE = get_device()

RUN = wandb.init(
    entity="sabrina-du-mila-mila",
    project="curious-george",
)

def main():
    print("Loading pre-trained network...")
    predictiveNet = PredictiveNet.loadNet(NET_NAME)

    # Step 2: Run the Object Memory Task
    print("Running Object Memory Task...")
    print(f"Using device: {DEVICE}")
    omt = ObjectMemoryTask(
        predictiveNet,
        env_novel_name=ENV_NAME,
        num_trials=50,      # Fewer trials for faster testing, otherwise default 100
        trial_duration=500, # Shorter duration for testing, default 1000
        lr_trials=2,        # Learning rate multiplier
        lr_groups=[0, 1, 2], # Which parameter groups to train
        package=PACKAGE,
        device=DEVICE,
    )

    # Step 3: Display results
    print("\nResults:")
    print(f"Goal modulation: {omt.objectLearning['goalmodulation']:.4f}")
    print(f"Control modulation: {omt.objectLearning['ctlmodulation_diffloc']:.4f}")

    # Step 4: Generate plots
    print("Generating plots...")
    plt.figure(figsize=(12, 8))
    omt.ObjectLearningFigure(netname=NET_NAME, savefolder=RESULTS_SAVE_FOLDER)
    print("Done!")

if __name__ == "__main__":
    main()