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

target_smiles = {'Lenalidomide': 'O=C1NC(=O)CCC1N3C(=O)c2cccc(c2C3)N',
                 'Salmeterol': 'OCc1cc(ccc1O)[C@H](O)CNCCCCCCOCCCCc2ccccc2',
                 '5-HT6 receptor ligand': 'O=S(=O)(Nc4cc2CCC1(CCC1)Oc2c(N3CCNCC3)c4)c5ccccc5F', 
                 'DDR1_037': 'O=C(Nc4cccc(C(=O)N3CCN(c1ccnc2[nH]ccc12)C3)c4)c5cccc(C(F)(F)F)c5',
                 'DDR1_032': 'Cc3cc2[nH]c(c1cc(CN(C)C)cc(C(F)(F)F)c1)nc2cc3C#Cc4cncnc4'}

lenal = target_smiles['Lenalidomide']
result_lenal = retrosnythesis(lenal, model, graph_function, device, atom_templates, bond_templates, template_infos)
print('Lenalidomide\n\n')
print(result_lenal)
Sal = target_smiles['Salmeterol']
result_Sal = retrosnythesis(Sal, model, graph_function, device, atom_templates, bond_templates, template_infos)
print('Salmeterol\n\n')
print(result_Sal)
HT6 = target_smiles['5-HT6 receptor ligand']
result_HT6 = retrosnythesis(HT6, model, graph_function, device, atom_templates, bond_templates, template_infos)
print('5-HT6 receptor ligand\n\n')
print(result_HT6)
DDR1_037 = target_smiles['DDR1_037']
result_DDR1_037 = retrosnythesis(DDR1_037, model, graph_function, device, atom_templates, bond_templates, template_infos)
print('DDR1_037\n\n')
print(result_DDR1_037)
DDR1_032 = target_smiles['DDR1_032']
result_DDR1_032 = retrosnythesis(DDR1_032, model, graph_function, device, atom_templates, bond_templates, template_infos)
print('DDR1_032\n\n')
print(result_DDR1_032)
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
IPythonConsole.drawOptions.addAtomIndices = True
IPythonConsole.molSize = 300,300

if 1 != len(result_lenal):
	mol = Chem.MolFromSmiles(result_lenal.SMILES[1])
	Draw.MolToFile(mol,'T_gps1_lenal.png')

if 1 != len(result_Sal):
	mol = Chem.MolFromSmiles(result_Sal.SMILES[1])
	Draw.MolToFile(mol,'T_gps1_Sal.png')

if 1 != len(result_HT6):
	mol = Chem.MolFromSmiles(result_HT6.SMILES[1])
	Draw.MolToFile(mol,'T_gps1_HT6.png')
	
if 1 != len(result_DDR1_037):
	mol = Chem.MolFromSmiles(result_DDR1_037.SMILES[1])
	Draw.MolToFile(mol,'T_gps1_DDR1_037.png')

if 1 != len(result_DDR1_032):
	mol = Chem.MolFromSmiles(result_DDR1_032.SMILES[1])
	Draw.MolToFile(mol,'T_gps1_DDR1_032.png')


