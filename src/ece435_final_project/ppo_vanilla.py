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
        response = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=True, max_length=512, num_return_sequences=1)
        full_mask = torch.ones_like(response)
        return response, full_mask
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        logits = outputs.logits
        logprobs = torch.log_softmax(logits, dim=-1)
        return logits, logprobs

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
        logits = outputs.logits
        logprobs = torch.log_softmax(logits, dim=-1)
        return logits, logprobs
    
class PPO:
    def __init__(self, actor, reward_critic, reward_model, ref_model, sft_dataset, gamma, beta, epsilon, alpha, lr, gae_lambda, avg_cost, critic_loss_wt):
        """
        gamma: discount
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

        self.actor.optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
        self.reward_critic.optimizer = optim.AdamW(self.reward_critic.parameters(), lr=lr)

    # Compute the KL penalty
    def kl_penalty(self, actor_logprobs, ref_logprobs):
        actor_probs = torch.exp(actor_logprobs)
        kl_penalty = torch.sum(actor_probs * (actor_logprobs - ref_logprobs), dim=-1)
        return kl_penalty

    # Gather log probs
    def gather_log_probs(self, logprobs, tokens):
        return logprobs.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

    # Compute the reward
    def reward(self, input_ids, attention_mask, actor_logprobs, ref_logprobs):
        r_rm = self.reward_model(input_ids, attention_mask).squeeze(-1)
        kl_penalty = self.kl_penalty(actor_logprobs, ref_logprobs)
        r_hat = r_rm - (self.beta / 2) * kl_penalty

        return r_hat

    # Calculate the GAE
    def gae(self, rewards, values, gamma=0.99, lam=0.95):
        B, T = rewards.shape()
        adv = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(T - 1)):
            delta = rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
            gae   = delta + gamma * lam * gae
            adv[:, t] = gae
        returns = adv + values[: , :-1]
        return adv.detach(), returns.detach()

    def actor_loss(self, old_logprobs, new_log_probs, advantages):
        ratio = torch.exp(new_log_probs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        return -torch.mean(torch.min(surr1, surr2))

    def critic_loss(self, values, returns):
        return torch.mean((values - returns) ** 2)
    
    # Update the actor
    def ppo_update(self, input_ids, attention_mask):
        with torch.no_grad():
            # Generate response
            response, masks = self.actor.generate(input_ids, attention_mask)
            # Compute actor and reference logprobs
            old_actor_logprobs, _ = self.actor.forward(response, masks)
            old_logprobs = self.gather_log_probs(old_actor_logprobs, response).detach()
        
        # Compute new logprobs
        actor_logprobs, _ = self.actor.forward(response, masks)
        ref_logprobs, _ = self.ref_model.forward(response, masks)
        new_logprobs = self.gather_log_probs(actor_logprobs, response)

        # Compute advantage for reward
        rewards = self.reward(response, masks, actor_logprobs, ref_logprobs)
        R = torch.zeros_like(response, dtype=actor_logprobs.dtype)
        R[:, -1] = rewards
        reward_values = self.reward_critic(response, masks).squeeze(-1)
        V = torch.zeros_like(response, dtype=reward_values.dtype)
        V[:, -1]  = reward_values
        advantage_reward, returns = self.gae(R, V)

        # Compute the losses
        actor_loss = self.actor_loss(old_logprobs, new_logprobs, advantage_reward)
        critic_loss = self.critic_loss(reward_values, returns)
        total_loss = self.critic_loss_wt * critic_loss + actor_loss

        # Update the actor
        self.actor.optimizer.zero_grad()
        self.reward_critic.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        nn.utils.clip_grad_norm_(self.reward_critic.parameters(), 1.0)
        self.actor.optimizer.step()
        self.reward_critic.optimizer.step()

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