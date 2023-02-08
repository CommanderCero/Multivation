import torch
import torch.nn as nn

body = torch.nn.Sequential(
    nn.Linear(10, 256),
    nn.Linear(256, 256),
)

num_heads = 20
heads = [nn.Sequential(nn.Linear(256, 4)) for _ in range(num_heads)]


batch_inp = torch.rand(64, 10)
batch_out = torch.rand(64, 4)

body_out = body(batch_inp)
head_outputs = [head(body_out) for head in heads]
losses = [nn.functional.mse_loss(out, batch_out) for out in head_outputs]

loss = torch.sum(torch.stack(losses))
loss.backward()
    
    
for l in body:
    print(f"--- {l} ---")
    for name, p in l.named_parameters():
        print(name, p.norm())
        
for i, head in enumerate(heads):
    for l in body:
        print(f"--- Head {i} {l} ---")
        for name, p in l.named_parameters():
            print(name, p.norm())