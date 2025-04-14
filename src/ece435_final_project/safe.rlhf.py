import torch
import torch.nn as nn
import torch.optim as optim
import math

from transformers import AutoTokenizer
from safe_rlhf.models import AutoModelForScore

class Actor(nn.Module):
    def __init__(self, model_name):
        super(Actor, self).__init__()
        self.model = AutoModelForScore.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')

    def forward(self, prompts):
        outputs = self.model(**prompts)
        return outputs.logits
    
    def compute_log_likelihood(self, prompts):
        outputs = self.model(**prompts)
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        return log_probs
    
class RewardModel(nn.Module):
    def __init__(self, model_name):
        super(RewardModel, self).__init__()
        self.model = AutoModelForScore.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
    def forward(self, prompts):
        outputs = self.model(**prompts)
        return outputs.logits
    
class CostModel(nn.Module):
    def __init__(self, model_name):
        super(CostModel, self).__init__()
        self.model = AutoModelForScore.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')

    def forward(self, prompts):
        outputs = self.model(**prompts)
        return outputs.logits

class ReferenceModel(nn.Module):
    def __init__(self, model_name):
        super(ReferenceModel, self).__init__()
        self.model = AutoModelForScore.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')

    def forward(self, prompts):
        outputs = self.model(**prompts)
        return outputs.logits

class SafeRLHF:
    def __init__(self, actor, reward_model, cost_model, ref_model, sft_dataset, prompt_dataset, gamma, beta, epsilon, alpha, lr, lambda_init, avg_cost):
        """
        gamma: PTX loss weight
        beta: KL loss weight
        epsilon: PPO clipping parameter
        lr: learning rate
        lambda_init: initial value for lambda
        avg_cost: average cost
        """
        self.actor = actor
        self.reward_model = reward_model
        self.cost_model = cost_model
        self.ref_model = ref_model
        self.sft_dataset = sft_dataset
        self.prompt_dataset = prompt_dataset

        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.alpha = alpha
        self.avg_cost = avg_cost

        self.lambda_param = torch.tensor(lambda_init, requires_grad=True)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)

    # Compute the KL penalty
    def kl_penalty(self, prompt, responses):
        with torch.no_grad():
            ref_logits = self.ref_model(prompt)
            actor_logits = self.actor(prompt, responses)
        kl_penalty = torch.mean(torch.nn.functional.kl_div(actor_logits, ref_logits, reduction='batchmean'))

        return kl_penalty

    # Compute the reward
    def reward(self, prompt, response):
        r_rm = self.reward_model(prompt)
        kl_penalty = self.kl_penalty(prompt, response)
        r_hat = r_rm + (self.beta / 2) * kl_penalty

        return r_hat
    
    # Compute the cost
    def cost(self, prompt, response):
        c_rm = self.cost_model(prompt)
        kl_penalty = self.kl_penalty(prompt, response)
        c_hat = c_rm - (self.beta / 2) * kl_penalty

        return c_hat
    
    # Compute the PTX loss
    def ptx_loss(self, prompt, target_responses):
        pass

    # Calculate the GAE
    def gae(self, rewards, values, next_values, masks):
        deltas = rewards + (1 - masks) * next_values - values
        advantages = torch.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            advantages[t] = deltas[t] + (1 - masks[t]) * self.gamma * advantages[t + 1]
        return advantages

    def ppo_loss(self, old_log_probs, new_log_probs, advantages):
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        return -torch.mean(torch.min(surr1, surr2))

    # Update the actor
    def update_step(self, prompt, avg_cost):
        # Generate response
        response = self.actor.forward(prompt)

        # Compute advantage for reward and cost
        rewards = self.reward(prompt, response)
        costs = self.cost(prompt, response)
        values = self.actor(prompt)
        next_values = self.actor(prompt)
        masks = torch.ones_like(rewards)
        advantage_reward = self.gae(rewards, values, next_values, masks)
        advantage_cost = self.gae(costs, values, next_values, masks)

        # Compute the losses
        ptx_loss = self.ptx_loss(prompt, response)
        safe_rl_reward_loss = self.ppo_loss(prompt, response, advantage_reward)
        safe_rl_cost_loss = self.ppo_loss(prompt, response, advantage_cost)
        safe_rl_loss = (1 / (1 + self.lambda_param)) * (safe_rl_reward_loss - self.lambda_param * safe_rl_cost_loss)

        total_loss = safe_rl_loss + self.gamma * ptx_loss

        # Update the actor
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Update lambda
        new_lambda = self.lambda_param * torch.exp(self.alpha * avg_cost)
        self.lambda_param.data = new_lambda.data

        return total_loss.item(), costs.item()
    
    def train(self, num_epochs):
        # Initialize average cost
        avg_cost = 0
        steps = 0
        for epoch in range(num_epochs):
            for batch in self.sft_dataset:
                prompt = batch['prompt']
                loss, cost = self.update_step(prompt, avg_cost)
                steps += 1
                # Update average cost
                avg_cost = (avg_cost * (steps - 1) + cost) / steps
                print(f"Epoch: {epoch}, Loss: {loss}")





    

        



    
