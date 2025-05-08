import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import RLHFDatasetLoader
from safe_rlhf.models import AutoModelForScore
from tqdm import tqdm
from transformers import AutoModelForCausalLM

CACHE_DIR = os.getenv("HF_HOME")
OUTPUT_DIR = os.getenv("ECE435_OUTPUT")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class PPO:
    def __init__(self, actor, reward_critic, reward_model, ref_model, sft_dataset, critic_loss_wt, gamma, beta, epsilon, gae_lambda, lr):
        """
        critic_loss_wt: weight for the critic loss
        gamma: discount
        beta: KL loss weight
        epsilon: PPO clipping parameter
        gae_lambda: lambda for GAE
        lr: learning rate
        """
        max_mem = {
            0: "20GiB",
            1: "20GiB",
            2: "20GiB",
            3: "20GiB",
        }
        self.actor = AutoModelForCausalLM.from_pretrained(actor, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR, device_map="auto", max_memory=max_mem)
        self.reward_critic = AutoModelForScore.from_pretrained(reward_critic, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR, device_map="auto", max_memory=max_mem)
        self.reward_model = AutoModelForScore.from_pretrained(reward_model, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR, device_map="auto", max_memory=max_mem)
        self.ref_model = AutoModelForCausalLM.from_pretrained(ref_model, torch_dtype=torch.bfloat16, cache_dir=CACHE_DIR, device_map="auto", max_memory=max_mem)
        self.sft_dataset = sft_dataset
        self.critic_loss_wt = critic_loss_wt
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.gae_lambda = gae_lambda

        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.AdamW(self.reward_critic.parameters(), lr=lr)

        self.reward_model.eval()
        for p in self.reward_model.parameters():
            p.requires_grad = False

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def move_to_device(self, tensor, model):
        return tensor.to(next(model.parameters()).device)

    # Gather log probabilities
    def gather_log_probs(self, logprobs, response, attention_mask):
        prompt_lens = attention_mask.sum(dim=1).long()
        batch_logp = []
        for i, L in enumerate(prompt_lens):
            lp_i = logprobs[i, L:, :].gather(-1,
                        response[i].unsqueeze(-1)).squeeze(-1)
            batch_logp.append(lp_i)

        return torch.stack(batch_logp, dim=0)

    # Compute the KL penalty
    def kl_penalty(self, actor_logprobs, ref_logprobs):
        kl_step = actor_logprobs - ref_logprobs
        return - self.beta * kl_step

    # Compute the reward
    @torch.no_grad
    def reward(self, input_ids, attention_mask, output_mask, actor_logprobs, ref_logprobs):
        input_ids = self.move_to_device(input_ids, self.reward_model)
        attention_mask = self.move_to_device(attention_mask, self.reward_model)
        r_rm = self.reward_model(input_ids, attention_mask).end_scores.squeeze(-1).to(actor_logprobs.device)
        kl_penalty = self.kl_penalty(actor_logprobs, ref_logprobs)
        rewards = kl_penalty.clone()
        end_idx = output_mask.long().sum(dim=1) - 1
        batch_idx = torch.arange(rewards.size(0), device=rewards.device)
        rewards[batch_idx, end_idx] += r_rm[batch_idx]

        return rewards

    # Calculate the GAE
    def gae(self, rewards, values, gamma=0.99, lam=0.95):
        B, T = rewards.shape
        adv = torch.zeros_like(rewards)
        gae = torch.zeros(B, device=values.device)
        for t in reversed(range(T)):
            delta = rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
            gae = delta + gamma * lam * gae
            adv[:, t] = gae
        returns = adv + values[:, :-1]
        return adv.detach(), returns.detach()

    def actor_loss(self, old_logprobs, new_log_probs, advantages):
        ratio = torch.exp(new_log_probs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        return -torch.mean(torch.min(surr1, surr2))

    def critic_loss(self, values, returns):
        return torch.mean((values - returns) ** 2)

    # Generate a rollout and calculate the advantage
    def rollout(self, input_ids, attention_mask):
        input_ids = self.move_to_device(input_ids, self.actor)
        attention_mask = self.move_to_device(attention_mask, self.actor)

        self.actor.eval()
        self.ref_model.eval()
        self.reward_critic.eval()

        with torch.no_grad():
            # Generate response
            sequence = self.actor.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=True, max_new_tokens=256, num_return_sequences=1)

            L_prompt  = input_ids.size(1)
            response  = sequence[:, L_prompt:]
            resp_masks  = torch.ones_like(response)
            full_masks = torch.cat([attention_mask, resp_masks], dim=1)

            old_logits = self.actor(sequence, full_masks).logits
            old_lp = torch.log_softmax(old_logits, dim=-1)
            old_logprobs = self.gather_log_probs(old_lp, response, attention_mask)

            sequence = self.move_to_device(sequence, self.ref_model)
            full_masks = self.move_to_device(full_masks, self.ref_model)

            ref_logits = self.ref_model(sequence, full_masks).logits
            ref_lp = torch.log_softmax(ref_logits, dim=-1)
            ref_logprobs = self.gather_log_probs(ref_lp, response, attention_mask)

            # Compute advantage for reward
            rewards = self.reward(sequence, full_masks, resp_masks, old_logprobs, ref_logprobs)

            sequence = self.move_to_device(sequence, self.reward_critic)
            full_masks = self.move_to_device(full_masks, self.reward_critic)

            reward_values = self.reward_critic(sequence, full_masks).scores.squeeze(-1)[:, L_prompt:]
            zero_pad = torch.zeros(reward_values.size(0), 1, device=reward_values.device)
            reward_values_padded = torch.cat([reward_values, zero_pad], dim=1)

            advantage_reward, returns = self.gae(rewards, reward_values_padded, self.gamma, self.gae_lambda)

        return sequence, response, full_masks, attention_mask, old_logprobs, advantage_reward, reward_values, returns

    def ppo_update(self, sequence, response, full_masks, attention_mask, old_logprobs, advantage_reward, reward_values, returns):
        self.actor.train()
        self.reward_critic.train()

        # Compute the new log probabilities
        new_logits = self.actor(sequence, full_masks).logits
        new_lp = torch.log_softmax(new_logits, dim=-1)
        new_logprobs = self.gather_log_probs(new_lp, response, attention_mask)

        reward_values = self.reward_critic(sequence, full_masks).scores.squeeze(-1)[:, sequence.size(1) - response.size(1):]
        mean_reward = reward_values.mean()

        actor_loss = self.actor_loss(old_logprobs, new_logprobs, advantage_reward)
        critic_loss = self.critic_loss(reward_values, returns)
        total_loss = self.critic_loss_wt * critic_loss + actor_loss

        # Update the actor
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        nn.utils.clip_grad_norm_(self.reward_critic.parameters(), 1.0)
        self.actor_optim.step()
        self.critic_optim.step()

        return total_loss.item(), mean_reward.item()

    def train(self, num_epochs: int, save_every: int = 50):
        for epoch in range(num_epochs):
            i = 0
            for batch in self.sft_dataset:
                # logging.info(f"BATCH:\n{batch}")
                (sequence, response, full_masks, attention_mask, old_logprobs, advantage_reward, reward_values, returns) = self.rollout(
                    batch['input_ids'], batch['attention_mask'])
                loss, reward = self.ppo_update(sequence, response, full_masks, attention_mask, old_logprobs, advantage_reward, reward_values, returns)
                logging.info(f"Epoch: {epoch}, Loss: {loss}, Reward: {reward}")

                i += 1
                if i % save_every == 0:
                    logging.info("Saving checkpoint...")
                    torch.save(self.actor.state_dict(), os.path.join(OUTPUT_DIR, "vanilla/actor_current.pt"))
                    torch.save(self.reward_critic.state_dict(), os.path.join(OUTPUT_DIR, "vanilla/reward_current.pt"))

                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            torch.save(self.actor.state_dict(), os.path.join(OUTPUT_DIR, f"vanilla/actor_epoch_{epoch}.pt"))
            torch.save(self.reward_critic.state_dict(), os.path.join(OUTPUT_DIR, f"vanilla/reward_epoch_{epoch}.pt"))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataloader = RLHFDatasetLoader(max_length=128, batch_size=32)
    sft_dataset = dataloader.get_dataloader()
    ppo = PPO(actor="PKU-Alignment/alpaca-7b-reproduced",
              reward_critic="PKU-Alignment/beaver-7b-v1.0-reward",
              reward_model="PKU-Alignment/beaver-7b-v1.0-reward",
              ref_model="PKU-Alignment/alpaca-7b-reproduced",
              sft_dataset=sft_dataset,
              critic_loss_wt=0.5,
              gamma=0.99,
              beta=0.1,
              epsilon=0.1,
              gae_lambda=0.95,
              lr=1e-5)

    ppo.train(5)