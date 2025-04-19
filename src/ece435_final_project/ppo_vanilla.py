import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from transformers import AutoModelForCausalLM
from safe_rlhf.models import AutoModelForScore
from dataloader import RLHFDatasetLoader

# ACTOR (policy)
# Tested with PKU-Alignment/alpaca-7b-reproduced and PKU-Alignment/beaver-7b-v3.0-reward (see below)
class Actor(nn.Module): # LM to be updated/fine-tuned; our policy.
    def __init__(self, model_name):
        super(Actor, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    @torch.no_grad()
    def generate(self, input_ids, attention_mask):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=True, max_length=512, num_return_sequences=1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        return log_probs

class RewardModel(nn.Module): # used for computing reward only
    def __init__(self, model_name):
        super(RewardModel, self).__init__()
        self.model = AutoModelForScore.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        return outputs.scores

# CRITIC (value)
class RewardCriticModel(nn.Module): # initialized based on reward model, optimized using advantage estimates
    def __init__(self, model_name):
        super(RewardCriticModel, self).__init__()
        self.model = AutoModelForScore.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        return outputs.scores
    
class ReferenceModel(nn.Module): # LM (not to be updated - just for kl div computation)
    def __init__(self, model_name):
        super(ReferenceModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        return log_probs

class PPO:
    def __init__(self, actor, reward_critic, reward_model, ref_model, sft_dataset, gamma, beta, epsilon, alpha, lr, gae_lambda, avg_cost, critic_loss_wt):
        """
        gamma: PTX loss weight
        beta: KL loss weight
        epsilon: PPO clipping parameter
        lr: learning rate
        lambda_init: initial value for lambda
        avg_cost: average cost
        """
        self.actor = actor
        self.reward_critic = reward_critic
        self.reward_model = reward_model
        self.ref_model = ref_model
        self.sft_dataset = sft_dataset
        self.critic_loss_wt = critic_loss_wt

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.alpha = alpha
        self.avg_cost = avg_cost

        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.reward_critic.parameters()), lr=lr)

    # Compute the KL penalty
    def kl_penalty(self, actor_logprobs, ref_logprobs):
        kl_penalty = torch.mean(torch.nn.functional.kl_div(actor_logprobs, ref_logprobs, reduction='batchmean'))
        return kl_penalty

    # Compute the reward
    def reward(self, input_ids, attention_mask, actor_logprobs, ref_logprobs):
        r_rm = self.reward_model(input_ids, attention_mask)
        kl_penalty = self.kl_penalty(actor_logprobs, ref_logprobs)
        r_hat = r_rm + (self.beta / 2) * kl_penalty

        return r_hat

    # Calculate the GAE
    def gae(self, rewards, values, gamma=0.99, lam=0.95):
        next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)
        deltas = rewards + gamma * next_values - values
        adv = torch.zeros_like(deltas)
        last_gae = 0
        for t in reversed(range(deltas.size(1))):
            last_gae = deltas[:, t] + gamma * lam * last_gae
            adv[:, t] = last_gae
        returns = adv + values
        return adv, returns

    def actor_loss(self, old_log_probs, new_log_probs, advantages):
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        return -torch.mean(torch.min(surr1, surr2))

    def critic_loss(self, values, returns):
        return torch.mean((values - returns) ** 2)
    
    # Update the actor
    def ppo_update(self, input_ids, attention_mask):
        # Generate response
        response = self.actor.generate(input_ids, attention_mask)
        # Generate old logprobs
        old_logprobs = self.actor.forward(input_ids, attention_mask)
        # Concatenate prompt and response
        full_ids = torch.cat((input_ids, response), dim=1)
        resp_mask = torch.ones_like(response)
        full_mask = torch.cat([attention_mask, resp_mask], dim=-1)
        # Compute logprobs
        logprobs = self.actor.forward(full_ids, full_mask)
        # Compute reference logits
        ref_logprobs = self.ref_model.forward(full_ids, full_mask)

        # Compute advantage for reward and cost
        rewards = self.reward(full_ids, full_mask, logprobs, ref_logprobs)
        reward_values = self.reward_critic_model(input_ids, attention_mask)
        advantage_reward, returns = self.gae(rewards, reward_values)

        # Compute the losses
        actor_loss = self.actor_loss(old_logprobs, logprobs, advantage_reward)
        critic_loss = self.critic_loss(reward_values, returns) # rewards or returns?
        total_loss = self.critic_loss_wt * critic_loss + actor_loss

        # Update the actor
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
    
    def train(self, num_epochs):
        # Initialize average cost
        steps = 0
        for epoch in range(num_epochs):
            for batch in self.sft_dataset:
                print("BATCH: ")
                print(batch)
                loss = self.ppo_update(**batch)
                steps += 1
                print(f"Epoch: {epoch}, Loss: {loss}")

if __name__ == "__main__":
    actor = Actor("PKU-Alignment/alpaca-7b-reproduced")
    reward_critic = RewardCriticModel("PKU-Alignment/beaver-7b-v3.0-reward")
    ref_model = ReferenceModel("PKU-Alignment/alpaca-7b-reproduced")
    reward_model = RewardModel("PKU-Alignment/beaver-7b-v3.0-reward")
    dataloader = RLHFDatasetLoader()
    sft_dataset = dataloader.get_dataloader()
    ppo = PPO(actor, reward_critic, reward_model, ref_model, sft_dataset, 0.99, 0.99, 0.1, 0.1, 0.001, 0.95, 0, 0.5)

    ppo.train(5)