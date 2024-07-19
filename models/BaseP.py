import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F

def type_mask(sample_type, target = 'recommend'):  # True for recommend sample, False for search sample
    if target == 'recommend':
        return sample_type.eq(0).type(torch.float).unsqueeze(-1)  # 0:recommend, 1:search
    else:
        return sample_type.eq(1).type(torch.float).unsqueeze(-1)  # 0:recommend, 1:search  
class Attn(torch.nn.Module):
    
    def __init__(self, hidden_size):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size
        self.linear = torch.nn.Linear(self.hidden_size, hidden_size)
        

    def general_score(self, hidden, encoder_output):
        energy = self.linear(encoder_output)
        return torch.sum(hidden * energy, dim=2)
    def forward(self, hidden, ori_encoder_outputs,mask=None):
        '''
        input: hidden: [N,dim]
        encoder_output: [N,T,dim]
        mask: [N,T]
        output:N,T
        '''
        encoder_outputs = ori_encoder_outputs.transpose(0,1)  #[T,N,dim]
        attn_energies = self.general_score(hidden, encoder_outputs)        
        attn_energies = attn_energies.t()  #[N,T]
        
        if mask!=None:
            A=F.softmax(attn_energies, dim=1) #N,T
            
            A = A*mask #N,T
            A_sum=torch.sum(A, dim=1) #N
            threshold=torch.ones_like(A_sum)*1e-5 
            A_sum = torch.max(A_sum, threshold).unsqueeze(1) #[N,1]
            
            A = A / A_sum #[N,T]
            attn_energies =A.unsqueeze(1) #[N,1,T]

        else:
            attn_energies=F.softmax(attn_energies, dim=1).unsqueeze(1)  #[N,1,T]
        
        context = attn_energies.bmm(ori_encoder_outputs) #[N,1,T]*[N,T,dim]
        
        return context.squeeze(1),attn_energies.squeeze(1) #[N,dim]
class IntentGraph(nn.Module):

    def __init__(self,config,demand_mlp=None,item_query_attn = None,linear_K = None):
        super(IntentGraph, self).__init__()
        self.sample_graph = config.sample_graph
        self.device =config.device
        self.demand_mlp = demand_mlp
        self.item_query_attn = item_query_attn
        self.linear_K = linear_K

    def edge_query(self,edges):
        
        recommend_mask = type_mask(edges.data['type'], target='recommend')
        search_mask = type_mask(edges.data['type'], target='search')

        #search
        edge_num = edges.data['query'].shape[0] 
        query_embs = self.query_emb(edges.data['query'])  #[N,per_query_num,dim]
        tmp_search_query = torch.sum(query_embs,dim=-2)

        #recommend
        user_query_embs = self.query_emb(edges.data['userquery'])
        
        item_query_r = self.query_emb(edges.data['itemquery'])
        if self.u_to_i:  #src: user nodes; dst: item nodes
            
            key = self.linear_K(torch.concat([edges.src['h'],user_query_embs.reshape([edge_num,-1])],dim=1)) #[N,dim]
            

            item_query_emb,_= self.item_query_attn(key,item_query_r)
            tmp_recommend_query = self.demand_mlp(torch.cat([edges.src['h'],edges.dst['h'],torch.sum(user_query_embs,dim=1),item_query_emb],dim=1))
        else: #src: item nodes; dst: user nodes
            
            key = self.linear_K(torch.concat([edges.dst['h'],user_query_embs.reshape([edge_num,-1])],dim=1)) #[N,dim]
            
            item_query_emb,_= self.item_query_attn(key,item_query_r)
            
            tmp_recommend_query = self.demand_mlp(torch.cat([edges.dst['h'],edges.src['h'],torch.sum(user_query_embs,dim=1),item_query_emb],dim=1))

        q_e = recommend_mask*tmp_recommend_query+search_mask*tmp_search_query

        return {'q_e':q_e}
    
    
    
    def forward(self, g, user_emb,item_emb,query_emb=None,first = True):

        self.query_emb = query_emb   
        

        g.nodes['user'].data['h']=user_emb
        g.nodes['item'].data['h']=item_emb
        
        if first:
            if self.sample_graph:
                self.u_to_i = True
                g['uqi'].apply_edges(self.edge_query)
                self.u_to_i = False
                g['iqu'].apply_edges(self.edge_query)
            else:
                self.u_to_i = True
                g['uqi'].apply_edges(self.edge_query)
                #Edge attributes are independent of edge direction.
                g['iqu'].edata['q_e']=g['uqi'].edata['q_e'] 


            
        dic= {
            'uqi': (fn.u_add_e('h', 'q_e','m'),fn.mean('m','h')),
            'iqu': (fn.u_sub_e('h', 'q_e','m'),fn.mean('m','h')),
            
        }

        
        g.multi_update_all( 
            dic,
            "mean"
        ) 
        return g.nodes['user'].data['h'],g.nodes['item'].data['h']
   