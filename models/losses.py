import torch
import torch.nn.functional as F


def hierarchical_contrastive_loss(z1, z2, contrast_weight):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        loss += instance_contrastive_loss(z1, z2, contrast_weight)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        loss += instance_contrastive_loss(z1, z2, contrast_weight)
        d += 1
    return loss / d


def instance_contrastive_loss(z1, z2, contrast_weight, temperature=1):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    
    w = torch.cat([contrast_weight, contrast_weight], dim=0)
    bars_weight = -(torch.abs(w.reshape(-1, 1) - w.reshape(-1, 1).T))
    bars_weight = torch.softmax(bars_weight, dim=-1)
        
    logits_weight = torch.tril(bars_weight, diagonal=-1)[:, :-1]
    logits_weight += torch.triu(bars_weight, diagonal=1)[:, 1:] # 2B x (2B-1)
    
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    
    loss = (-F.log_softmax(logits / temperature, dim=-1) * logits_weight).mean()
    return loss




