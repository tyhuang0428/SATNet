import torch
import os
from collections import OrderedDict


datasets = ['msd', 'pmd', 'rgbd']
for dataset in datasets:
    path = os.path.join('./best', dataset, 'iter_20000.pth')
    ckpt = torch.load(path)
    state_dict = ckpt['state_dict']
    new = OrderedDict()
    for key in state_dict:
        name = key.replace('diff', 'ccl') if 'diff' in key else key
        new[name] = state_dict[key]
    ckpt['state_dict'] = new
    torch.save(ckpt, './ckpts/' + dataset + '.pth')
