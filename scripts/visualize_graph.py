import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_win_rates():
    """
    Plots a bar chart comparing the win rates (harmlessness) of PPO and SFT models.
    """
    models = ['Safe RLHF', 'SFT']
    win_rates = [0.51, 0.49]  # Example proportions from the image

    logger.info("Generating bar plot for win rates")

    plt.figure(figsize=(6, 4))
    plt.bar(models, win_rates)
    plt.ylim(0, 1)
    plt.ylabel('Proportion')
    plt.title('Win Rates (Helpfulness): Safe RLHF vs SFT')
    plt.tight_layout()
    plt.savefig('safe_helpfulness.png')

if __name__ == "__main__":
    plot_win_rates()
