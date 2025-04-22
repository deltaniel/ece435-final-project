import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LogitsProcessor
from safe_rlhf.models import AutoModelForScore
from dataloader import RLHFDatasetLoader

class ClampLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # scores is [batch, vocab_size]
        # replace any nan/inf/neg values before softmax
        scores = torch.nan_to_num(
            scores,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4
        )
        return scores

logger = logging.getLogger("ppo_rlhf")
logger.setLevel(logging.DEBUG)

# a) console handler (optional)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)5s %(message)s"))
logger.addHandler(ch)

# b) file handler
fh = logging.FileHandler("ppo_rlhf.log", mode="a")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)5s [%(name)s] %(message)s"
))
logger.addHandler(fh)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,)

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
        self.actor = get_peft_model(AutoModelForCausalLM.from_pretrained(actor,
                                                                        quantization_config=bnb_config, device_map="auto"),
                                    LoraConfig(r=8, 
                                               lora_alpha=16, 
                                               lora_dropout=0.05,
                                               target_modules=["q_proj","v_proj"]))
        self.reward_critic = get_peft_model(AutoModelForScore.from_pretrained(reward_critic, 
                                                               quantization_config=bnb_config, 
                                                               offload_folder="offload_cache",
                                                               offload_state_dict=True,
                                                               device_map="auto"),
                                            LoraConfig(r=8, 
                                               lora_alpha=16, 
                                               lora_dropout=0.05,
                                               target_modules=["q_proj","v_proj"]))
        self.reward_model = AutoModelForScore.from_pretrained(reward_model, 
                                                              quantization_config=bnb_config, 
                                                              offload_folder="offload_cache",
                                                              offload_state_dict=True,
                                                              device_map="auto",)
        self.ref_model = AutoModelForCausalLM.from_pretrained(ref_model, 
                                                              quantization_config=bnb_config, 
                                                              offload_folder="offload_cache",
                                                              offload_state_dict=True,
                                                              device_map="auto",)

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
        
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.actor.train()
        self.reward_critic.train()

    # Gather log probabilities
    def gather_log_probs(self, logprobs, response, attention_mask):
        prompt_lens = attention_mask.sum(dim=1).long()         
        batch_logp = []
        T_resp = response.size(1)
        for i, L in enumerate(prompt_lens):
            logits_slice = logprobs[i, L : L + T_resp, :]
            lp_i = logits_slice.gather(
                    -1,
                    response[i].unsqueeze(-1)
                ).squeeze(-1)
            batch_logp.append(lp_i)

        return torch.stack(batch_logp, dim=0)

    # Compute the KL penalty
    def kl_penalty(self, actor_logprobs, ref_logprobs):
        kl_step = actor_logprobs - ref_logprobs
        return - self.beta * kl_step 
        
    # Compute the reward
    @torch.no_grad
    def reward(self, input_ids, attention_mask, output_mask, actor_logprobs, ref_logprobs):
        r_rm = self.reward_model(input_ids, attention_mask).end_scores.squeeze(-1)
        kl_penalty = self.kl_penalty(actor_logprobs, ref_logprobs)
        rewards = kl_penalty.clone()      
        end_idx = output_mask.long().sum(dim=1) - 1  
        batch_idx = torch.arange(rewards.size(0), device=rewards.device)
        rewards[batch_idx, end_idx] += r_rm 

        return rewards

    # Calculate the GAE
    def gae(self, rewards, values, gamma=0.99, lam=0.95):
        B, T = rewards.shape
        adv = torch.zeros_like(rewards)
        gae = torch.zeros(B, device=values.device)
        for t in reversed(range(T - 1)):
            delta = rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
            gae = delta + gamma * lam * gae
            adv[:, t] = gae
        returns = adv + values
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
        device = next(self.actor.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        logger.info(f"Rollout on device={device}, input_ids.shape={input_ids.shape}, attn.shape={attention_mask.shape}")

        self.actor.eval()
        self.ref_model.eval()

        with torch.no_grad():
            processor = [ClampLogitsProcessor()]
            # Generate response
            sequence = self.actor.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            max_new_tokens=32,
            num_return_sequences=1,
            logits_processor=processor,
            # you can also tame the distribution a bit:
            temperature=0.8,
            top_k=50,
            top_p=0.95,
        )
            logger.info(f"Generated sequence of length {sequence.size(1)}")

            L_prompt  = input_ids.size(1)
            response  = sequence[:, L_prompt:]
            resp_masks  = torch.ones_like(response)
            full_masks = torch.cat([attention_mask, resp_masks], dim=1)
            logger.info(f"Response.shape={response.shape} (should be [B, T_resp])")

            old_logits = self.actor(sequence, full_masks).logits
            old_lp = torch.log_softmax(old_logits, dim=-1)
            old_logprobs = self.gather_log_probs(old_lp, response, attention_mask)

            if torch.isnan(old_logits).any() or torch.isinf(old_logits).any():
                    logger.error(f"old_logits contains NaN/Inf: min={old_logits.min()}, max={old_logits.max()}")
            logger.info(f"old_logprobs.shape={old_logprobs.shape}")

            ref_logits = self.ref_model(sequence, full_masks)
            ref_lp = torch.log_softmax(ref_logits, dim=-1)
            ref_logprobs = self.gather_log_probs(ref_lp, response, attention_mask)

            # Compute advantage for reward
            rewards = self.reward(sequence, full_masks, resp_masks, old_logprobs, ref_logprobs)

            reward_values = self.reward_critic(sequence, full_masks).scores.squeeze(-1)[:, L_prompt:]
            advantage_reward, returns = self.gae(rewards, reward_values, self.gamma, self.gae_lambda)

            logger.info(f"rewards.shape={rewards.shape}, values.shape={reward_values.shape}, adv.shape={advantage_reward.shape}")

        return sequence, L_prompt, response, full_masks, attention_mask, old_logprobs, advantage_reward, returns
    
    def ppo_update(self, sequence, L_prompt, response, full_masks, attention_mask, old_logprobs, advantage_reward, returns):
        self.actor.train()
        self.reward_critic.train()
        logger.debug("Starting PPO update")
        # Compute the new log probabilities
        new_logits = self.actor(sequence, full_masks).logits
        new_lp = torch.log_softmax(new_logits, dim=-1)
        new_logprobs = self.gather_log_probs(new_lp, response, attention_mask)

        new_values = self.reward_critic(sequence, full_masks).scores.squeeze(-1)[:, L_prompt:]

        actor_loss = self.actor_loss(old_logprobs, new_logprobs, advantage_reward)
        critic_loss = self.critic_loss(new_values, returns)
        total_loss = self.critic_loss_wt * critic_loss + actor_loss

        # Update the actor
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.reward_critic.parameters(), 0.5)
        self.actor_optim.step()
        self.critic_optim.step()

        logger.info(f"PPO update loss={total_loss.item():.4f}")
        return total_loss.item()

    def train(self, num_epochs):
        logger.info(f"Begin training for {num_epochs} epochs")
        self.actor.train()
        self.reward_critic.train()
        for epoch in range(num_epochs):
            for step, batch in enumerate(self.sft_dataset):
                logger.info(f"Epoch {epoch} step {step}: batch shapes "
                    f"{batch['input_ids'].shape}")
                (sequence, L_prompt, response, full_masks, attention_mask, old_logprobs, advantage_reward, reward_values, returns) = self.rollout(
                    batch['input_ids'], batch['attention_mask'])
                loss = self.ppo_update(sequence, L_prompt, response, full_masks, attention_mask, old_logprobs, advantage_reward, reward_values, returns)
            logger.info(f"Finished epoch {epoch}")

if __name__ == "__main__":
    dataloader = RLHFDatasetLoader(
        tokenizer_name="PKU-Alignment/alpaca-7b-reproduced",  
        max_length=32,
        batch_size=4,                 
        shuffle=True
    )
    
    sft_dataset = dataloader.get_dataloader()
    ppo = PPO(actor="PKU-Alignment/alpaca-7b-reproduced", 
              reward_critic="PKU-Alignment/beaver-7b-unified-reward",
              reward_model="PKU-Alignment/beaver-7b-unified-reward",
              ref_model="PKU-Alignment/alpaca-7b-reproduced", 
              sft_dataset=sft_dataset, 
              critic_loss_wt=0.5,
              gamma=0.99, 
              beta=0.1, 
              epsilon=0.1, 
              gae_lambda=0.95,
              lr=1e-6)

    ppo.train(5)