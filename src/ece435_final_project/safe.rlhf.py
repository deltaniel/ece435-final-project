import torch
import torch.nn as nn
import torch.optim as optim
import math
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataloader import RLHFDatasetLoader

class Actor(nn.Module):
    def __init__(self, model_name, lora_rank, lora_alpha, lora_dropout, target_modules):
        super(Actor, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

        if lora_rank > 0:
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask):
        return self.model.generate(input_ids, attention_mask, do_sample=True, max_length=512, num_return_sequences=1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        return log_probs
    
class RewardCritic(nn.Module):
    def __init__(self, model_name):
        super(RewardCritic, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        return outputs.logits
    
class RewardModel(nn.Module):
    def __init__(self, model_name):
        super(RewardModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        return outputs.logits
    
class CostCritic(nn.Module):
    def __init__(self, model_name):
        super(RewardCritic, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        return outputs.logits
    
class CostModel(nn.Module):
    def __init__(self, model_name):
        super(CostModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        return outputs.logits

class ReferenceModel(nn.Module):
    def __init__(self, model_name):
        super(ReferenceModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        return outputs.logits

    def compute_log_likelihood(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask)
        log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        return log_probs

class SafeRLHF:
    def __init__(self, actor, reward_model, reward_critic_model, cost_model, cost_critic_model, ref_model, sft_dataset, prompt_dataset, gamma, beta, epsilon, alpha, lr, lambda_init, avg_cost):
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
        self.reward_critic_model = reward_critic_model
        self.cost_critic_model = cost_critic_model
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

    # Tokenize the output
    def tokenize_output(self, output):
        tokenizer = AutoTokenizer.from_pretrained(self.actor.model.config._name_or_path)
        tokenized_output = tokenizer(output, return_tensors='pt')
        return tokenized_output

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
    def gae(self, rewards, values, gamma=0.99, lam=0.95):
        next_values = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)
        deltas = rewards + gamma * next_values - values
        adv = torch.zeros_like(deltas)
        last_gae = 0
        for t in reversed(range(deltas.size(1))):
            last_gae = deltas[:, t] + gamma * lam * last_gae
            adv[:, t] = last_gae
        returns = adv + values

    # Compute the actor loss
    def actor_loss(self, old_log_probs, new_log_probs, advantages):
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        return -torch.mean(torch.min(surr1, surr2))

    # Compute the value loss
    def value_loss(self, values, returns):
        return torch.mean((values - returns) ** 2)
    
    # Update the actor
    def update_step(self, input_ids, attention_mask, avg_cost):
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
        ref_logprobs = self.ref_model(full_ids, full_ids)

        # Compute advantage for reward and cost
        rewards = self.reward(full_ids, full_mask, logprobs, ref_logprobs)
        costs = self.cost(full_ids, full_mask, logprobs, ref_logprobs)
        reward_values = self.reward_critic_model(input_ids, attention_mask)
        cost_values = self.cost_critic_model(input_ids, attention_mask)
        advantage_reward = self.gae(rewards, reward_values)
        advantage_cost = self.gae(costs, cost_values)

        # Compute the losses
        ptx_loss = self.ptx_loss(input_ids, attention_mask, response)
        safe_rl_reward_loss = self.actor_loss(logprobs, old_logprobs, advantage_reward)
        safe_rl_cost_loss = self.actor_loss(logprobs, old_logprobs, advantage_cost)
        safe_rl_loss = (1 / (1 + self.lambda_param)) * (safe_rl_reward_loss - self.lambda_param * safe_rl_cost_loss)
        reward_critic_loss = self.value_loss(reward_values, rewards)
        cost_critic_loss = self.value_loss(cost_values, costs)

        total_loss = safe_rl_loss + self.gamma * ptx_loss

        # Update the critics
        self.reward_critic_model.optimizer.zero_grad()
        reward_critic_loss.backward()
        self.reward_critic_model.optimizer.step()

        self.cost_critic_model.optimizer.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_model.optimizer.step()

        # Update the actor
        self.actor.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()

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





    

        



    
