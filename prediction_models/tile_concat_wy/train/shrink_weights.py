import torch
weights = [f'./weights/Resnext50_reg/Resnext50_reg_{i}_ckpt.pth.tar' for i in range(4)]
for i in range(5):
    state_dict = torch.load(weights[i])
    weights_state = state_dict['state_dict']
    torch.save(weights_state, f'./weights/Resnext50_reg/Resnext50_reg_{i}_best.pth')