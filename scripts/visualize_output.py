import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_slurm_file(file_path: str) -> pd.DataFrame:
    """Parses a SLURM output file to extract loss and reward data.

    Args:
        file_path (str): Path to the SLURM output file.

    Returns:
        pd.DataFrame: DataFrame containing epoch, loss, and reward columns.
    """
    epoch_list: list[int] = []
    loss_list: list[float] = []
    reward_list: list[float] = []

    epoch_pattern = re.compile(r"Epoch:\s*(\d+)", re.IGNORECASE)
    loss_pattern = re.compile(r"Loss:\s*([\d\.e+-]+)", re.IGNORECASE)
    reward_pattern = re.compile(r"Reward:\s*([\d\.\-e+]+)", re.IGNORECASE)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            epoch_match = epoch_pattern.search(line)
            loss_match = loss_pattern.search(line)
            reward_match = reward_pattern.search(line)
            if epoch_match and loss_match and reward_match:
                epoch = int(epoch_match.group(1))
                loss = float(loss_match.group(1))
                reward = float(reward_match.group(1))
                epoch_list.append(epoch)
                loss_list.append(loss)
                reward_list.append(reward)

    df = pd.DataFrame({
        'epoch': epoch_list,
        'loss': loss_list,
        'reward': reward_list
    })

    print(f"Parsed {len(df)} records from the SLURM output file.")
    return df

def plot_loss_and_reward(df: pd.DataFrame, save: bool = False) -> None:
    """Plots loss and reward over time using matplotlib.

    Args:
        df (pd.DataFrame): DataFrame with 'loss' and 'reward' columns.
    """
    plt.figure()
    plt.plot(df.index, df['loss'])
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Vanilla PPO Training Loss')
    plt.grid(True)
    if save:
        plt.savefig('loss_curve.png')
    else:
        plt.show()

    plt.figure()
    plt.plot(df.index, df['reward'])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Vanilla PPO Training Reward')
    plt.grid(True)
    if save:
        plt.savefig('reward_curve.png')
    else:
        plt.show()


if __name__ == "__main__":
    file_path = "C:/dev/ece435-final-project/scripts/slurm-63787119.out"
    df_parsed = parse_slurm_file(file_path)
    plot_loss_and_reward(df_parsed, save=True)
