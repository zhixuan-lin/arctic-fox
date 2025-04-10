# Adaptive Computation Pruning (ACP) for the Forgetting Transformer

Official implementation of "[Adaptive Computation Pruning for the Forgetting Transformer](https://arxiv.org/abs/2504.06949)". This implementation is a preview version and it will be integrated into the [main FoX repository](https://github.com/zhixuan-lin/forgetting-transformer) in the future.

## Dependencies

Install the following. We pin the versions to ensure that this works, but you don't have to.

```
pip install pytest einops numpy
pip install torch==2.4.0  # This also installs triton==3.0.0
```

## Usage

The core API change compared to the original Forgetting Attention kernel is the `adaptive_threshold` argument, which corresponds to the $\delta$ threshold in the paper. Here is an example demonstrating how you should set the threshold.  You can read the paper to understand the rationale behind it. Note currently we only support dynamicaly setting the threshold for models that use QK-norm. Support for models without QK-norm will be added after further experiments.

```python
import torch
from forgetting_attention import forgetting_attention
import math
from torch import nn
from einops import rearrange

batch_size = 4
num_heads = 12
seq_len = 512
head_dim = 64
dtype = torch.bfloat16
device = "cuda"

q = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=dtype, device=device, requires_grad=True)
k = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=dtype, device=device, requires_grad=True)
v = torch.randn((batch_size, seq_len, num_heads, head_dim), dtype=dtype, device=device, requires_grad=True)
# You can use a tiny linear layer to get `fgate_logit`.
# For example, let `x` be the attention input with shape (batch_size, seq_len, hidden_size) 
# which is also used to compute `q`, `k` and `v`. You can get `fgate_logit` as follows
#     In your model's `__init__`: `self.fgate_proj = nn.Linear(hidden_size, num_heads, bias=True)`
#     In your model's `forward`:  `fgate_logit = self.fgate_proj(x)`
fgate_logit = torch.randn((batch_size, seq_len, num_heads), dtype=dtype, device=device, requires_grad=True)
log_fgate = torch.nn.functional.logsigmoid(fgate_logit.float())


class GroupRMSNorm(nn.Module):
    """Naive implementation of grouped RMSNorm"""
    def __init__(self, hidden_size: int, num_groups: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x):
        assert x.size(-1) == self.hidden_size, x.size(-1)
        x = rearrange(x, '... (g d) -> ... g d', g=self.num_groups)
        weight = rearrange(self.weight, '(g d) -> g d', g=self.num_groups)
        rstd = x.float().square().mean(dim=-1, keepdim=True).sqrt()
        out = x / rstd * weight
        out = rearrange(out, '... g d -> ... (g d)')
        out = out.to(x.dtype)
        return out

# exp(log_epsilon) bounds the maximum total attention weights that could be pruned
log_epsilon = -10

with torch.no_grad():
    # Calculate upper bounds of attention logits
    q_norm = GroupRMSNorm(hidden_size=num_heads * head_dim, num_groups=num_heads).to(device)
    k_norm = GroupRMSNorm(hidden_size=num_heads * head_dim, num_groups=num_heads).to(device)
    q, k = [rearrange(entry, '... h d -> ... (h d)') for entry in (q, k)]
    q = q_norm(q)
    k = k_norm(k)
    q, k = [rearrange(entry, '... (h d) -> ... h d', h=num_heads) for entry in (q, k)]
    # If we use QK-norm, it is easily to get an upper bound of q/k L2-norm
    max_q_norm = q_norm.weight.view(num_heads, head_dim).abs().max(dim=-1).values * math.sqrt(head_dim)
    max_k_norm = k_norm.weight.view(num_heads, head_dim).abs().max(dim=-1).values * math.sqrt(head_dim)

    logit_upper_bound = max_q_norm * max_k_norm / math.sqrt(head_dim)
    adaptive_threshold = -(2 * logit_upper_bound + math.log(seq_len)) + log_epsilon


out = forgetting_attention(q, k, v, log_fgate, adaptive_threshold=adaptive_threshold)
assert out.size() == (batch_size, seq_len, num_heads, head_dim)

```

## Citation

If you use this code, please consider citing the following:

```
@article{lin2025adaptive,
  title={Adaptive Computation Pruning for the Forgetting Transformer},
  author={Lin, Zhixuan and Obando-Ceron, Johan and He, Xu Owen and Courville, Aaron},
  journal={arXiv preprint arXiv:2504.06949},
  year={2025}
}
```
