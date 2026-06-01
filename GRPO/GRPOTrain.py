import torch
from torch.optim import AdamW
import torch.nn.functional as F
from copy import deepcopy

def GRPOTrain(model, train_dataloader, reward_func, tokenizer, save_path, epochs, lr=1e-5, beta=0.04, epsilon=0.2, G=8, max_tokens=512, temperature=0.3, monit_steps=20, run=None):
    ref_model = deepcopy(model)
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        print(f"EPOCH: {epoch}")
        losses = []
        accuracies = []
        for batch in train_dataloader:
            token_ids = batch["token_ids"].to("cuda")
            attention_masks = batch["attention_masks"].to("cuda")
            prompt_length = token_ids.size(-1)
            answers = batch["answers"]
    
            with torch.no_grad():
                outputs = model.generate(
                    token_ids,
                    attention_mask=attention_masks,
                    max_new_tokens=max_tokens,
                    num_return_sequences=G,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                out_sequences = outputs[:, prompt_length:]              
                full_attention_mask = (outputs != tokenizer.eos_token_id).long()
                completion_mask = full_attention_mask[:, prompt_length:]  
    
                old_logits = model(outputs, attention_mask=full_attention_mask).logits
                old_logits = old_logits[:, prompt_length-1:-1, :]         
                old_per_token_logps = F.log_softmax(old_logits, dim=-1)
                old_per_token_logps = old_per_token_logps.gather(
                    2, out_sequences.unsqueeze(-1)
                ).squeeze(-1)                                             
                old_per_token_logps = old_per_token_logps.detach()
                
                ref_logits = ref_model(outputs, attention_mask=full_attention_mask).logits
                ref_logits = ref_logits[:, prompt_length-1:-1, :]         
                ref_per_token_logps = F.log_softmax(ref_logits, dim=-1)
                ref_per_token_logps = ref_per_token_logps.gather(
                    2, out_sequences.unsqueeze(-1)
                ).squeeze(-1)                                             
                
                rewards = reward_func(outputs, answers, G)            
                reward_mean = rewards.mean(dim=1, keepdim=True)
                reward_std = rewards.std(dim=1, keepdim=True)
                advantages = (rewards - reward_mean) / (reward_std + 1e-8)
                advantages = advantages.reshape(-1, 1)                   
    
            new_logits = model(outputs, attention_mask=full_attention_mask).logits
            new_logits = new_logits[:, prompt_length-1:-1, :]             
            per_token_logps = F.log_softmax(new_logits, dim=-1)
            per_token_logps = per_token_logps.gather(
                2, out_sequences.unsqueeze(-1)
            ).squeeze(-1)                                                
    
            policy_ratio = torch.exp(per_token_logps - old_per_token_logps)  
            clip_policy_ratio = torch.clamp(policy_ratio, 1 - epsilon, 1 + epsilon)
    
            loss = torch.min(
                advantages * policy_ratio,
                advantages * clip_policy_ratio,
            )                                                             
    
            kl_div = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )                                                             
    
            loss = -loss + beta * kl_div
    
            loss = ((loss * completion_mask).sum(dim=-1) / 
                    completion_mask.sum(dim=-1).clamp(min=1)).mean()
    
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
    
            losses.append(loss.item())
    
            accuracy = (rewards == 1.0).float().mean().item()
            accuracies.append(accuracy)
            
            if len(losses) % monit_steps == 0:
                step_loss = sum(losses[-monit_steps:])/monit_steps
                avg_accuracy = sum(accuracies)/len(accuracies)
                reward_std = rewards.std(dim=1).mean().item()
                kl = kl_div.mean().item()
                ratio = policy_ratio.mean().item()
                
                print(
                    f"Batch: {len(losses)} | "
                    f"Loss: {step_loss:.6f} | "
                    f"Avg Acc: {avg_accuracy:.2%} | "
                    f"Reward Std: {reward_std:.3f} | "
                    f"KL: {kl:.4f} | "
                    f"Ratio: {ratio:.4f}"
                )

                if run:
                    run.log({
                        "loss": step_loss,
                        "accuracy": avg_accuracy,
                        "reward_std": reward_std,
                        "kl": kl,
                        "ratio": ratio,
                    })
        
        torch.save(model.state_dict(), f"{save_path}/model_{epoch}.pth")
        print(f"Epoch {epoch} done | Mean Loss: {sum(losses)/len(losses):.6f} | Mean Acc: {sum(accuracies)/len(accuracies):.2%}")
    
    if run:
        run.finish()
