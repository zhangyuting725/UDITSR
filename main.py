
from conf import Config
from utils import now_time, build_graph,load_data,Dataset,test,EarlyStopping

from models.UDITSR import UDITSR
import torch


import numpy as np

import random,os


def run_time(config):
    
    print(now_time()+" start loading data")
    train_dataset,valid_dataset,test_dataset,global_graph_triplets= load_data(config)
    print(now_time()+" start building graph")
    global_graph = build_graph(config.user_num,config.item_num, global_graph_triplets,config.sample_graph,config.device)
    model = UDITSR(config).to(config.device)

    if not os.path.exists(config.save_model_dir):
        os.makedirs(config.save_model_dir)

    
    model_path = config.save_model_dir+"/main"+now_time()+".pt"

    print("-------------save----------",model_path)

    
    valid_data = Dataset(config,valid_dataset,test_batchsize=config.test_batchsize)
    test_data = Dataset(config,test_dataset,test_batchsize=config.test_batchsize)



    early_stopping = EarlyStopping(model_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_data = Dataset(config,train_dataset)
    print("train_total_step",train_data.total_step)
    print(now_time()+" start training")
    
    for epoch in range(1,config.epochs+1):
        print("----------------")
        train_sample=0
        '''train_loss_sum=0
        query_r_loss_sum =0
        cst_loss_sum = 0'''
        train_loss_sum_dict={}
        while True:
            data_dict = train_data.next_batch(data_type="train")

            sample_num = len(data_dict['USER'])
            train_sample+=sample_num
            if config.sample_graph:
                user_node = list(set(data_dict["USER"].tolist()))
                item_node = list(set(data_dict["ITEM"].tolist())|set(data_dict['NEG_ITEM'].tolist()))
                
                #first-order sample
                s1_g = global_graph.in_subgraph({'user': user_node, 'item': item_node}) 
                uqr_edges = s1_g.edges(form='uv', etype='uqi')
                rqu_edges = s1_g.edges(form='uv', etype='iqu')
                s1_user_node = uqr_edges[0].tolist()
                s1_item_node = rqu_edges[0].tolist()
                
                del s1_g
            
                #second-order sample
                s2_user_node = list(set(s1_user_node)|set(user_node))
                s2_item_node = list(set(s1_item_node)|set(item_node))
                s2_g = global_graph.in_subgraph({'user':s2_user_node,'item':s2_item_node})
                losses_dict = model.get_loss(data_dict,s2_g)
            else:
                losses_dict = model.get_loss(data_dict,global_graph)  
            loss = sum(losses_dict.values())
            for k,v in losses_dict.items():
                train_loss_sum_dict[k]= train_loss_sum_dict.get(k,0)+v.item()*sample_num
            #exit()
            optimizer.zero_grad()
            loss.backward()
            
            
            optimizer.step()
            torch.cuda.empty_cache()
            if train_data.step>=train_data.total_step:
                break
        print("{} epoch:{}, the number of samples:{}".format(now_time(),epoch,train_sample),end=" ")
        for k,v in train_loss_sum_dict.items():
            print("{}:{:.4f}".format(k,v/train_sample),end=",")
        print()

        print("valid",end=" ")
        flag=test(epoch,model,valid_data,"valid",global_graph,early_stopping=early_stopping)
        test(epoch,model,test_data,"test",global_graph)
        if flag:
            break
        
            
            
 
    pre_state_dict=torch.load(model_path)
    model.load_state_dict(pre_state_dict)
    print("---------------------best!!!")
    
    test(1,model,test_data,"test",global_graph)
    os.remove(model_path)
if __name__=='__main__':
    config= Config()
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("------------当前的模型参数设置-------------")
    print(config)
    run_time(config)
         


        