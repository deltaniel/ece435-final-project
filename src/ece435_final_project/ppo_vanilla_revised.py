import logging
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                          LogitsProcessor)

from safe_rlhf.models import AutoModelForScore
from dataloader import RLHFDatasetLoader


logger = logging.getLogger("ppo_rlhf")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s %(levelname)5s %(message)s")
ch = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO);  ch.setFormatter(formatter)
fh = logging.FileHandler("ppo_rlhf.log", mode="a"); fh.setLevel(logging.DEBUG); fh.setFormatter(formatter)
logger.addHandler(ch); logger.addHandler(fh)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True,
)

class ClampLogitsProcessor(LogitsProcessor):
    """Clamp or zero‑out bad logits before softmax."""
    def __call__(self, _ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)

class PPOTrainer:
    def __init__(
        self,
        actor_id: str,
        reward_critic_id: str,
        reward_model_id: str,
        ref_model_id: str,
        sft_dataset,
        *,
        critic_loss_wt: float = 0.5,
        gamma: float = 0.99,
        beta: float = 0.1,
        clip_eps: float = 0.1,
        gae_lambda: float = 0.95,
        lr: float = 5e-5,
        lora_r: int = 8,
        lora_alpha: int = 16,
    ):
        
        self.actor = self._load_lora_lm(actor_id, lora_r, lora_alpha)
        self.value_critic = self._load_lora_score(
            reward_critic_id, lora_r, lora_alpha
        )
        self.reward_model = AutoModelForScore.from_pretrained(
            reward_model_id,
            quantization_config=bnb_config,
            device_map="auto",
        ).eval().requires_grad_(False)

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            ref_model_id,
            quantization_config=bnb_config,
            device_map="auto",
        ).eval().requires_grad_(False)

        self.sft_dataset = sft_dataset
        self.critic_loss_wt = critic_loss_wt
        self.gamma = gamma
        self.beta = beta
        self.clip_eps = clip_eps
        self.gae_lambda = gae_lambda

        self.actor_optim  = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.AdamW(self.value_critic.parameters(), lr=lr)

    @staticmethod
    def _load_lora_lm(model_id: str, r: int, alpha: int):
        base = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto"
        )
        lora_cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        return get_peft_model(base, lora_cfg)

    @staticmethod
    def _load_lora_score(model_id: str, r: int, alpha: int):
        base = AutoModelForScore.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto"
        )
        lora_cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        return get_peft_model(base, lora_cfg)

    @staticmethod
    def _gather_log_probs(
        logp: torch.FloatTensor, response: torch.LongTensor, prompt_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        """Extract per‑token log‑probs of generated responses."""
        prompt_lens = prompt_mask.sum(dim=1).long()        # [B]
        T_resp      = response.size(1)
        out = []
        for i, L in enumerate(prompt_lens):
            slice_i = logp[i, L : L + T_resp, :]
            out.append(slice_i.gather(-1, response[i].unsqueeze(-1)).squeeze(-1))
        return torch.stack(out, dim=0)                     # [B, T_resp]

    def _kl_penalty(self, logp_act, logp_ref):
        return -self.beta * (logp_act - logp_ref)

    @staticmethod
    def _gae(
        rewards: torch.Tensor, values: torch.Tensor,
        gamma: float, lam: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = rewards.shape
        adv  = torch.zeros_like(rewards)
        last_gae = torch.zeros(B, device=rewards.device)
        for t in reversed(range(T)):
            next_val = values[:, t + 1] if t + 1 < T else 0.0
            delta = rewards[:, t] + gamma * next_val - values[:, t]
            last_gae = delta + gamma * lam * last_gae
            adv[:, t] = last_gae
        returns = adv + values
        return adv, returns

    @torch.no_grad()
    def rollout(self, input_ids, attention_mask):
        device = next(self.actor.parameters()).device
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        self.actor.eval();  self.value_critic.eval()

        seq = self.actor.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            max_new_tokens=32,
            temperature=0.8, top_k=50, top_p=0.95,
            logits_processor=[ClampLogitsProcessor()],
        )
        L_prompt  = input_ids.size(1)
        response  = seq[:, L_prompt:]
        resp_mask = torch.ones_like(response)
        full_mask = torch.cat([attention_mask, resp_mask], dim=1)

        act_logits = self.actor(seq, attention_mask=full_mask).logits
        ref_logits = self.ref_model(seq, attention_mask=full_mask).logits
        act_logp = torch.log_softmax(act_logits, dim=-1)
        ref_logp = torch.log_softmax(ref_logits, dim=-1)

        act_resp_logp = self._gather_log_probs(act_logp, response, attention_mask)
        ref_resp_logp = self._gather_log_probs(ref_logp, response, attention_mask)

        r_rm = self.reward_model(seq, attention_mask=full_mask).end_scores.squeeze(-1)
        kl_penalty = self._kl_penalty(act_resp_logp, ref_resp_logp)
        rewards = kl_penalty.clone()
        end_idx = resp_mask.long().sum(dim=1) - 1
        rewards[torch.arange(rewards.size(0), device=device), end_idx] += r_rm

        value_tokens = self.value_critic(seq, attention_mask=full_mask).scores.squeeze(-1)
        values_resp  = value_tokens[:, L_prompt:]                          # [B, T]

        adv, rets = self._gae(rewards, values_resp, self.gamma, self.gae_lambda)

        return seq, response, full_mask, attention_mask, act_resp_logp, adv, rets
    
    def _actor_loss(self, old_lp, new_lp, advantages):
        ratio = torch.exp(new_lp - old_lp)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        return -torch.mean(torch.min(surr1, surr2))

    @staticmethod
    def _critic_loss(values, returns):
        return torch.mean((values - returns) ** 2)

    def ppo_update(
        self,
        seq, response, full_mask, attention_mask,
        old_logp, advantages, returns, L_prompt
    ):
        self.actor.train(); self.value_critic.train()

        new_logits = self.actor(seq, attention_mask=full_mask).logits
        new_lp     = torch.log_softmax(new_logits, dim=-1)
        new_resp_lp = self._gather_log_probs(new_lp, response, attention_mask)

        value_tokens = self.value_critic(seq, attention_mask=full_mask).scores.squeeze(-1)
        values_resp  = value_tokens[:, L_prompt:]

        loss_actor  = self._actor_loss(old_logp, new_resp_lp, advantages)
        loss_critic = self._critic_loss(values_resp, returns)
        loss_total  = loss_actor + self.critic_loss_wt * loss_critic

        self.actor_optim.zero_grad(); self.critic_optim.zero_grad()
        loss_total.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.value_critic.parameters(), 0.5)
        self.actor_optim.step(); self.critic_optim.step()

        return loss_total.item()


    def train(self, epochs: int):
        logger.info(f"=== Begin PPO‑RLHF training ({epochs} epochs) ===")
        for epoch in range(epochs):
            for step, batch in enumerate(self.sft_dataset):
                seq, resp, mask, attn, old_lp, adv, rets = self.rollout(
                    batch["input_ids"], batch["attention_mask"]
                )
                loss = self.ppo_update(
                    seq, resp, mask, attn, old_lp, adv, rets,
                    L_prompt=batch["input_ids"].size(1)
                )
                if step % 10 == 0:
                    logger.info(f"Epoch {epoch} | Step {step:04d} | Loss {loss:.4f}")
        logger.info("=== Training complete ===")

if __name__ == "__main__":
    dataloader = RLHFDatasetLoader(
        tokenizer_name="PKU-Alignment/alpaca-7b-reproduced",
        max_length=32,
        batch_size=4,
        shuffle=True
    )
    sft_loader = dataloader.get_dataloader() 

    trainer = PPOTrainer(
        actor_id       ="PKU-Alignment/alpaca-7b-reproduced",
        reward_critic_id="PKU-Alignment/beaver-7b-unified-reward",
        reward_model_id ="PKU-Alignment/beaver-7b-unified-reward",
        ref_model_id    ="PKU-Alignment/alpaca-7b-reproduced",
        sft_dataset     =sft_loader,
        critic_loss_wt  =0.5,
        gamma           =0.99,
        beta            =0.1,
        clip_eps        =0.1,
        gae_lambda      =0.95,
        lr              =5e-5,     
    )

    trainer.train(epochs=5)