import torch
import torch.nn as nn

import sklearn
import dgl
import dgllife
from dgllife.model import MPNNGNN

<<<<<<< HEAD
from model_utils import pair_atom_feats, unbatch_mask, unbatch_feats, Global_Reactivity_Attention, GELU
=======
from model_utils import pair_atom_feats, unbatch_mask, unbatch_feats, Global_Reactivity_Attention, Global_Reactivity_Attention_2, unbatch_mask_2, unbatch_feats_2

class GraphGPS_edit(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats,
                 edge_hidden_feats,
                 num_step_message_passing,
                 attention_heads,
                 attention_layers,
                 AtomTemplate_n, 
                 BondTemplate_n):
        super(GraphGPS_edit, self).__init__() 
        
        
        self.mpnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        
        self.linearB = nn.Linear(node_out_feats*2, node_out_feats)

        self.lin_new = nn.Linear(node_in_feats, node_out_feats)
        
        self.att = Global_Reactivity_Attention_2(node_out_feats, attention_heads, attention_layers) # this should only give out node
        
        self.atom_linear =  nn.Sequential(
                            nn.Linear(node_out_feats, node_out_feats), 
                            nn.ReLU(), 
                            nn.Dropout(0.2),
                            nn.Linear(node_out_feats, AtomTemplate_n+1))
        
        self.bond_linear =  nn.Sequential(
                            nn.Linear(node_out_feats, node_out_feats), 
                            nn.ReLU(), 
                            nn.Dropout(0.2),
                            nn.Linear(node_out_feats, BondTemplate_n+1))
                            
        self.mlp =  nn.Sequential(
                            nn.Linear(node_out_feats, node_out_feats*2), 
                            nn.ReLU(), 
                            nn.Linear(node_out_feats*2, node_out_feats),
                            #nn.ReLU(), 
                            )                
        
        
    def forward(self, g, node_feats, edge_feats):
        #print(node_feats.size())
        atom_feats_mpnn = self.mpnn(g, node_feats, edge_feats)
        #print(atom_feats_mpnn.size())

        node_feats_new = self.lin_new(node_feats)
        #print(node_feats_new.size())
        #exit()
        #atom_feats = node_feats
        bond_feats = self.linearB(pair_atom_feats(g, node_feats_new))
        #print(bond_feats.size())
        #print(edge_feats.size())
        #print("This is edge")
        #exit()
        edit_node_feats, mask = unbatch_mask_2(g, node_feats_new, bond_feats)
        
        attention_score, edit_node_feats = self.att(edit_node_feats, mask)
        
        atom_feats_att = unbatch_feats_2(g, edit_node_feats)
        #print(atom_feats_att.size())
        atom_feats=self.mlp(torch.add(atom_feats_mpnn,atom_feats_att))
        #atom_outs = self.atom_linear(atom_feats) 
        #bond_outs = self.bond_linear(bond_feats) 
        
        #print("atom out", atom_outs.size())
        #print("bond out", bond_outs.size())
        #print(len(attention_score[0]))
        #print(len(atom_outs))
        #print(len(bond_outs))
        #exit()
        #print(atom_feats.size())
        #print("This is output")
        return atom_feats, edge_feats, attention_score

        
        
"""        
        gps_stack = []
        #pff_stack = []
        for _ in range(n_layers):
            att_stack.append(MultiHeadAttention(heads, d_model, dropout))
            pff_stack.append(FeedForward(d_model, dropout))
        self.att_stack = nn.ModuleList(att_stack)
        self.pff_stack = nn.ModuleList(pff_stack)
        
    def forward(self, x, mask):zcc
        scores = []
        for n in range(self.n_layers):
            score, x = self.att_stack[n](x, mask)
            x = self.pff_stack[n](x)
            scores.append(score)
        return scores, x


Global_Reactivity_Attention(node_out_feats, attention_heads, attention_layers)
"""
>>>>>>> da5ded7 (Initial commit: LocalRetro_GraphGPS code)

class LocalRetro_model(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats,
                 edge_hidden_feats,
                 num_step_message_passing,
                 attention_heads,
                 attention_layers,
                 AtomTemplate_n, 
                 BondTemplate_n,
                 activation = 'gelu'):
        super(LocalRetro_model, self).__init__()
                
        if activation in ['GELU', 'gelu']:
            self.activation = GELU()
        elif activation in ['ReLU', 'relu']:
            self.activation = nn.ReLU()
            
        self.mpnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        
        self.linearB = nn.Linear(node_out_feats*2, node_out_feats)

        self.att = Global_Reactivity_Attention(node_out_feats, attention_heads, attention_layers, activation=self.activation)
        
        self.atom_linear =  nn.Sequential(
                            nn.Linear(node_out_feats, node_out_feats), 
                            self.activation, 
                            nn.Dropout(0.2),
                            nn.Linear(node_out_feats, AtomTemplate_n+1))
        self.bond_linear =  nn.Sequential(
                            nn.Linear(node_out_feats, node_out_feats), 
                            self.activation,
                            nn.Dropout(0.2),
                            nn.Linear(node_out_feats, BondTemplate_n+1))
                            

        
        GPS_stack =[]
        #GraphGPS
        for _ in range(3):
            GPS_stack.append(GraphGPS_edit(node_in_feats if _ == 0 else node_out_feats, 
                                           edge_in_feats,
                                           node_out_feats,
                                           edge_hidden_feats,
                                           num_step_message_passing,
                                           attention_heads,
                                           attention_layers,
                                           AtomTemplate_n, 
                                           BondTemplate_n))
        
        self.GPS_stack = nn.ModuleList(GPS_stack)                            

    def forward(self, g, node_feats, edge_feats):

        #GPS
        
        for _ in range(1):
            node_feats, edge_feats, attention_score = self.GPS_stack[_](g, node_feats, edge_feats)
        
        atom_feats = node_feats
        bond_feats = self.linearB(pair_atom_feats(g, node_feats))
        atom_outs = self.atom_linear(atom_feats) 
        bond_outs = self.bond_linear(bond_feats) 
        
        #ORIGINAL
        """
        node_feats = self.mpnn(g, node_feats, edge_feats)
      
        atom_feats = node_feats
        bond_feats = self.linearB(pair_atom_feats(g, node_feats))
       
        edit_feats, mask = unbatch_mask(g, atom_feats, bond_feats)

        attention_score, edit_feats = self.att(edit_feats, mask)

        atom_feats, bond_feats = unbatch_feats(g, edit_feats)
        atom_outs = self.atom_linear(atom_feats) 
        bond_outs = self.bond_linear(bond_feats) 
        """

        
        #print("atom out", atom_outs.size())
        #print("bond out", bond_outs.size())
        #print(len(attention_score[0]))
        #print(len(atom_outs))
        #print(len(bond_outs))
        #exit()
        
        return atom_outs, bond_outs, attention_score


