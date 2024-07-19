
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from models.BaseP import  IntentGraph,type_mask,Attn
  
class UDITSR(nn.Module):
    def __init__(self,config) -> None:
        super(UDITSR, self).__init__()
        self.config = config
        self.device = config.device
        emb_dim = config.emb_dim
        layer_num = config.layer_num
        self.init_weight(config)
        self.graphlayers = nn.ModuleList()
        print("dropout",config.dropout)
        self.demand_mlp = nn.Sequential(
            nn.Linear(4*emb_dim,int(2*emb_dim)),
            nn.Dropout(p=config.dropout),
            nn.Tanh(),
            nn.Linear(int(2*emb_dim),int(2*emb_dim)),
            nn.Dropout(p=config.dropout),
            nn.Tanh(),
            nn.Linear(int(2*emb_dim),emb_dim)
        )
        self.item_query_attn = Attn(config.emb_dim)
        self.linear_K = nn.Linear((1+config.user_per_query_num)*emb_dim,emb_dim)
        for _ in range(layer_num):
            self.graphlayers.append(IntentGraph(config,demand_mlp=self.demand_mlp,item_query_attn = self.item_query_attn,linear_K = self.linear_K)) 
        
        self.sg_lambda=config.sg_lambda
        
        self.cst_lambda = config.cst_lambda
        self.logsigmoid = nn.LogSigmoid()

        input_size = emb_dim*3

        self.search_super_criterion = nn.MSELoss(reduction='none')
            
        self.search_score_predictor= nn.Sequential(
                nn.Linear(input_size,input_size//2),
                nn.LeakyReLU(),
                nn.Linear(input_size//2,input_size//4),
                nn.LeakyReLU(),
                nn.Linear(input_size//4,1),
                nn.LeakyReLU(),
        )

        self.recommend_score_predictor = nn.Sequential(
                nn.Linear(input_size,input_size//2),
                nn.LeakyReLU(),
                nn.Linear(input_size//2,input_size//4),
                nn.LeakyReLU(),
                nn.Linear(input_size//4,1),
                nn.LeakyReLU(),
        )  

    def init_weight(self,config):
        user_num = config.user_num
        item_num = config.item_num
        word_num = config.word_num
        emb_dim = config.emb_dim   
        
        initializer = nn.init.xavier_uniform_
        self.query_embedding = nn.Embedding(word_num, emb_dim)
        self.query_embedding.weight=initializer(self.query_embedding.weight)
        
        self.embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(user_num, emb_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(item_num, emb_dim))),
        })
        
            

    def graph_aggr(self,train_graph):

        prev_user_embedding = self.embedding_dict['user_emb']
        prev_item_embedding = self.embedding_dict['item_emb']   
        for  i, layer in enumerate(self.graphlayers):
            #first True: calculate the demand intention attribute on the edge.
            if i==0:
                user_emb, item_emb = layer(train_graph,prev_user_embedding,prev_item_embedding,self.query_embedding,first=True)
            else:
                user_emb, item_emb = layer(train_graph,user_emb,item_emb,self.query_embedding,first=False)  
            prev_user_embedding = prev_user_embedding+ user_emb*(1/(i+2))
            prev_item_embedding = prev_item_embedding+ item_emb*(1/(i+2))
        self.graph_user_emb_matrix= prev_user_embedding
        self.graph_item_emb_matrix= prev_item_embedding
        
        




    def get_loss(self,data_dict,global_graph=None,train=True,case=False):
        
        if train:
            self.graph_aggr(global_graph)
        graph_user_emb = self.graph_user_emb_matrix[data_dict['USER']]
        
        graph_pos_item_emb = self.graph_item_emb_matrix[data_dict['ITEM']]
        
        search_mask = type_mask(data_dict['TYPE'], target='search') #[batch,1]
        recommend_mask = type_mask(data_dict['TYPE'], target='recommend')
        
        #------search scene: the ground-truth query--------
        tmp_query_emb = self.query_embedding(data_dict['QUERY']) 
        query_emb = torch.sum(tmp_query_emb,dim=1)
        inputs = torch.cat([graph_user_emb,graph_pos_item_emb ,query_emb],dim=1)
        # predict the score for search 
        pos_score_s = self.search_score_predictor(inputs)


        #------recommend scene: Search-Supervised Demand Intent Generator--------
        init_user_emb = self.embedding_dict['user_emb'][data_dict['USER']]
        init_item_emb = self.embedding_dict['item_emb'][data_dict['ITEM']]
        # 1. sum-pooling operation for the user queries
        user_queries_emb = self.query_embedding(data_dict['USERTOPKQUERY']) #[-1,self.config.per_query_num,self.config.emb_dim]
        reshape_user_queries_emb = user_queries_emb.reshape([-1,self.config.user_per_query_num*self.config.emb_dim]) #[N,per_query_num*emb_dim]
        user_queries_emb = torch.sum(user_queries_emb,dim=1)

        # 2. gate operation for the item queries 
        item_queries_key = self.linear_K(torch.concat([init_user_emb,reshape_user_queries_emb],dim=1)) #[N,dim]
        
        item_queries_emb = self.query_embedding(data_dict['ITEMTOPKQUERY'])
        
        item_queries_emb,_= self.item_query_attn(item_queries_key,item_queries_emb) 
        
        # 3. generate the demand intent for (user,pos_item) pair
        mlp_query_emb =  self.demand_mlp(torch.cat([init_user_emb,init_item_emb,user_queries_emb,item_queries_emb],dim=1))#+top_query_emb
        
        # predict the score for recommendation    
        inputs = torch.cat([graph_user_emb,graph_pos_item_emb,mlp_query_emb],dim=1)
        pos_score_r = self.recommend_score_predictor(inputs)

        # search & recommend for pos triplet
        pos_score = pos_score_s*search_mask + pos_score_r*recommend_mask  
        
        sq_search_mask = search_mask.squeeze(1) #[batch] 
        # L_{SG}: search-supervised generation loss
        search_super_loss = torch.mean(self.search_super_criterion(mlp_query_emb,query_emb),dim=1) #[batch]
        search_super_loss = torch.mean(self.sg_lambda*sq_search_mask*search_super_loss)#[1]

        #the final query_emb for search and recommend postive sample
        query_emb = search_mask*query_emb + recommend_mask*mlp_query_emb
        
        if train:
            
            
            # the ground truth interactive result $\mathbf{e}_i^*$ should be near to the translated intent $\mathbf{e}_u^*+\widetilde{\mathbf{e}}_q$
            pos_cst = torch.mean(torch.pow(graph_user_emb+query_emb-graph_pos_item_emb , 2),dim=1)
            graph_neg_item_emb = self.graph_item_emb_matrix[data_dict['NEG_ITEM']]
            # negative $\mathbf{e}_{i'}^*$ should be away from $\mathbf{e}_u^*+\widetilde{\mathbf{e}}_q$
            neg_cst = torch.mean(torch.pow(graph_user_emb+query_emb-graph_neg_item_emb, 2),dim=1)
            #L_{CL}: intent translation contrastive loss
            cst_loss = -torch.mean(self.cst_lambda*self.logsigmoid(neg_cst-pos_cst))  #The bigger the negative sample, the better.
            
            #search: neg triple
            inputs = torch.cat([graph_user_emb,graph_neg_item_emb,query_emb],dim=1)
            neg_score_s = self.search_score_predictor(inputs)

            #recommend: neg triple
            init_item_emb = self.embedding_dict['item_emb'][data_dict['NEG_ITEM']]
            # generate the demand intent for (user,neg_item) pair
            item_queries_emb,_ = self.item_query_attn(item_queries_key,self.query_embedding(data_dict['NEG_ITEMTOPKQUERY']))
            mlp_query_emb =  self.demand_mlp(torch.cat([init_user_emb,init_item_emb,user_queries_emb,item_queries_emb],dim=1))#+top_query_emb
            inputs = torch.cat([graph_user_emb,graph_neg_item_emb,mlp_query_emb],dim=1)
            neg_score_r = self.recommend_score_predictor(inputs)

            # search & recommend for neg triplet
            neg_score = neg_score_s*search_mask  + neg_score_r*recommend_mask
            
            # the bpr loss
            bpr_loss = -torch.mean(self.logsigmoid(pos_score-neg_score))
            
            return {"bpr_loss":bpr_loss,"seach_supervised_loss":search_super_loss,"intent_translate_loss":cst_loss}
        else:
            return search_super_loss,pos_score


    


        


    
    
    