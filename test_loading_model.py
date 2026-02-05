import sys
sys.path.append('scripts')
from Retrosynthesis import init_LocalRetro, retrosnythesis
import torch

dataset = 'USPTO_50K'
device = torch.device('cuda:0')
model_path = 'models/LocalRetro_USPTO_50K_ORG.pth' #% dataset
config_path = 'data/configs/default_config.json'
data_dir = 'data/%s' % dataset

args = {'data_dir': data_dir, 'model_path': model_path, 'config_path': config_path, 'device': device}
model, graph_function, atom_templates, bond_templates, template_infos = init_LocalRetro(args)


ckpt = torch.load("models/LocalRetro_USPTO_50K_ORG.pth", map_location="gpu")
print("loaded checkpoint type:", type(ckpt))

# If your checkpoint is a dict, it might be either a state_dict or a training checkpoint
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    state = ckpt["model_state_dict"]
else:
    state = ckpt  # assume it's directly a state_dict

# This line should NOT involve any indexing
missing, unexpected = model.load_state_dict(state, strict=False)
print("missing keys:", len(missing), "unexpected keys:", len(unexpected))
print("Load_state_dict finished")
