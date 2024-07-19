
import os
import torch
import pandas as pd
from collections import defaultdict
import math
import random
import numpy as np
import dgl
import pickle as pkl
import datetime

import copy
import math
from conf import Config
from sklearn.metrics import roc_auc_score,average_precision_score

import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=10, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        
        

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #path = os.path.join(self.save_path, 'best_network.pth')
        path = self.save_path
        torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
        
        self.val_loss_min = val_loss





    

def print_metric(test_labels,predict_labels,imp_indexs,pair_metrics):
    imp_indexs = np.array(imp_indexs)
    
    sample_indexs = imp_indexs[:,0].tolist()
    user_indexs = imp_indexs[:,1].tolist()
    
    g_l, g_p = group_labels(np.array(test_labels),np.array(predict_labels), sample_indexs)
    metrics = cal_metric(
        g_l, g_p, pair_metrics
        )
    
    if len(pair_metrics)>1:

        g_l, g_p = group_labels(np.array(test_labels),np.array(predict_labels), user_indexs)
        tmp_metric = cal_metric(
            g_l, g_p, ['gauc']
            )
        metrics['gauc'] = tmp_metric['gauc']
    for metric,value in metrics.items():
        print(metric,":",value,end=" ")
    print()
    return metrics


     
def test(epoch,model,test_data,data_type,global_graph=None, early_stopping =None,show_test=True):
    if data_type=="test" and epoch%10!=1: 
        return
    config =Config()
    model.eval()



    with torch.no_grad ():
        search_test_labels,search_predict_labels,recommend_test_labels,recommend_predict_labels =[],[],[],[]
        search_imply_indexs,recommend_imply_indexs =[],[]
        test_sample=0
        test_loss_sum=0
        model.graph_aggr(global_graph)#get aggr embedding
        while True:
            

            data_dict = test_data.next_batch(data_type=data_type)
            
            sample_num = len(data_dict['USER'])
            test_sample+=sample_num
            test_loss,predict=model.get_loss(data_dict,train=False)
            
           
            test_loss_sum+=test_loss.item()*sample_num
            
            tmp_labels = data_dict['LABEL'].reshape(-1).tolist()
            tmp_pred = predict.reshape(-1).tolist()
            
            time_lst = data_dict['SAMPLE_NUM']
            type_lst = data_dict['TYPE'].reshape(-1).tolist()
            user_lst = data_dict['USER'].reshape(-1).tolist()
            for i in range(sample_num):
                if type_lst[i]==1: 

                    search_test_labels.append(tmp_labels[i])
                    search_predict_labels.append(tmp_pred[i])
                    search_imply_indexs.append([time_lst[i],user_lst[i]])     
                else:
                    recommend_test_labels.append(tmp_labels[i])
                    recommend_predict_labels.append(tmp_pred[i])
                    recommend_imply_indexs.append([time_lst[i],user_lst[i]])
            
                
            if test_data.step>=test_data.total_step:
                break
        
            
        print("{},样本数量:{},loss:{:.4f},".format(now_time(),test_sample,test_loss_sum/test_sample),end=" ")
       
        print(" ")

        if config.pair_metrics is not None:
            if data_type == "valid": #validate data 
                metrics = print_metric(search_test_labels+recommend_test_labels,search_predict_labels+recommend_predict_labels,search_imply_indexs+recommend_imply_indexs,["hit@5"])
            else: 
                print("search:   ",len(search_test_labels),end=" ")
                print_metric(search_test_labels,search_predict_labels,search_imply_indexs,config.pair_metrics)
                print("recommend:",len(recommend_test_labels),end=" ")
                    
                print_metric(recommend_test_labels,recommend_predict_labels,recommend_imply_indexs,config.pair_metrics) #print the results for test data 
        
        if  early_stopping:
            
            early_stopping(-metrics['hit@5'], model) #choose hit@5 for the early_stop
            #达到早停止条件时，early_stop会被置为True
            if early_stopping.early_stop:
                print("Early stopping")
                return True #说明停止
        return False



        
def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '

class Dataset:
    def __init__(self,config,data,test_batchsize=None) -> None:
        self.device = config.device
        if test_batchsize:
            self.batch_size = test_batchsize #bigger than batch_size to accelerate the inference speed
        else:
            self.batch_size = config.batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0
        self.data = data
        
        
        self.per_query_num = config.per_query_num
        self.user_per_query_num = config.user_per_query_num
        self.item_per_query_num = config.item_per_query_num

        self.item_list= list(range(config.item_num))

        self.user2item = pkl.load(open(config.user2item_path,'rb'))
        self.item2query = pkl.load(open(config.item2query_path,'rb'))


    def neg_item(self,userid):  
        while True: 
            item_ = random.choice(self.item_list)
            if item_ not in self.user2item[userid]: # 
                item_query =list(self.item2query[item_]) if item_ in self.item2query.keys() else []
                
                return item_,item_query
  
    def next_batch(self,data_type="train"):
        
        if self.step == self.total_step:
            self.step = 0

            if data_type=="train":  #shuffle train dataset
                np.random.seed(0)
                np.random.shuffle(self.index_list)
        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        user_batch =[]
        query_batch =[]
        item_batch =[]
        neg_item_batch=[]
        label_batch =[]
        sample_num_batch =[]
        type_batch = []
        usertopkquery_batch = []
        itemtopkquery_batch = []
        neg_item_query_batch = []

        for idx in self.index_list[start:offset]:
            
            user,query,item,sample_num,label,type,usertopkquery,itemtopkquery = self.data[idx]
            
            if data_type=="train": #train
                neg_item,neg_item_query=self.neg_item(user) #negative sample
                neg_item_batch.append(neg_item)
                neg_item_query =neg_item_query[:self.item_per_query_num]+[0]*(self.item_per_query_num-len(neg_item_query))
                neg_item_query_batch.append(neg_item_query)
                
            type_batch.append(type) #0: recommendation, 1: search
            query = query[:self.per_query_num]+[0]*(self.per_query_num-len(query))
            usertopkquery = usertopkquery[:self.user_per_query_num]+[0]*(self.user_per_query_num-len(usertopkquery))
            itemtopkquery = itemtopkquery[:self.item_per_query_num]+[0]*(self.item_per_query_num-len(itemtopkquery))
            usertopkquery_batch.append(usertopkquery)
            query_batch.append(query)
            user_batch.append(user)
            item_batch.append(item)
            label_batch.append(label)
            sample_num_batch.append(sample_num)
            itemtopkquery_batch.append(itemtopkquery)

        data_dict={}
        if data_type=="train":
            data_dict['NEG_ITEM']=torch.tensor(neg_item_batch,dtype=torch.int64).to(self.device)
            data_dict['NEG_ITEMTOPKQUERY']=torch.tensor(neg_item_query_batch,dtype=torch.int64).to(self.device)
        else:
            data_dict['SAMPLE_NUM'] = sample_num_batch
        data_dict['TYPE'] = torch.tensor(type_batch,dtype=torch.int64).to(self.device)
        data_dict["LABEL"]=torch.tensor(label_batch,dtype=torch.float32).unsqueeze(dim=1).to(self.device)
        data_dict["QUERY"]=torch.tensor(query_batch,dtype=torch.int64).to(self.device)
        data_dict['USER']=torch.tensor(user_batch,dtype=torch.int64).to(self.device)
        data_dict["ITEM"]=torch.tensor(item_batch,dtype=torch.int64).to(self.device)
        data_dict['USERTOPKQUERY']=torch.tensor(usertopkquery_batch,dtype=torch.int64).to(self.device)
        data_dict['ITEMTOPKQUERY']=torch.tensor(itemtopkquery_batch,dtype=torch.int64).to(self.device)
        return data_dict

 
           

def group_labels(labels, preds, group_keys):
    """Devide labels and preds into several group according to values in group keys.
    Args:
        labels (list): ground truth label list.
        preds (list): prediction score list.
        group_keys (list): group key list.
    Returns:
        all_labels: labels after group.
        all_preds: preds after group.
    """

    all_keys = list(set(group_keys))
    g_l = {k: [] for k in all_keys}
    g_p = {k: [] for k in all_keys}
    for l, p, k in zip(labels, preds, group_keys):
        g_l[k].append(l)
        g_p[k].append(p)

    all_labels = []
    all_preds = []
    for k in all_keys:
        all_labels.append(np.array(g_l[k]))
        all_preds.append(np.array(g_p[k]))
    

    return all_labels, all_preds







def cal_metric(true_labels, pred_labels,metrics):

    res = {}

    

    for true_label, pred_label in zip(true_labels, pred_labels):
        

        order=np.argsort(pred_label)[::-1]
        #print("order",order)
        for metric in metrics:
            #print("me",metric)
            y_true, y_score =copy.deepcopy(true_label),copy.deepcopy(pred_label)
            
            if metric == "mrr":
                
                #print(y_true.shape,y_score.shape)
                y_true = np.take(y_true, order)  #Label arranged in order
                #print("y_true",y_true)
                rr_score = y_true / (np.arange(len(y_true)) + 1)  #The reciprocal of the position, when y_true[0]=1, the result is 1; otherwise, it will decrease as the position gets bigger.
                res['mrr']= res.get('mrr',0)+np.sum(rr_score) / np.sum(y_true)
            elif metric =='map':
        
                res['map'] = res.get('map',0)+average_precision_score(true_label,pred_label)
                
            elif metric =='auc':

                #print(np.argwhere(order == 0))
                
                res['auc']=res.get('auc',0)+roc_auc_score(y_true, y_score) 
            elif metric =='gauc':  
                res['gauc']=res.get('gauc',0)+roc_auc_score(y_true, y_score)
            elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
                
                hit_list = [1, 2]
                ks = metric.split("@")
                if len(ks) > 1:
                    hit_list = [int(token) for token in ks[1].split(";")]
                for k in hit_list:
                    
                    ground_truth = np.where(y_true == 1.0)[0]
                    
                    argsort = order[:k]
                    #print("hit",ground_truth,argsort,y_score)
                    for idx in argsort:
                        if idx in ground_truth:
                            res["hit@{0}".format(k)]=res.get("hit@{0}".format(k),0)+1
                            break
                    res["hit@{0}".format(k)]=res.get("hit@{0}".format(k),0)+0
            elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            
                ndcg_list = [1, 2]
                ks = metric.split("@")
                if len(ks) > 1:
                    ndcg_list = [int(token) for token in ks[1].split(";")]
                for k in ndcg_list:
                    #ndcg_temp = ndcg_score(labels, preds, k)
                    
    
                    tmp_y_true = np.take(y_true, order[:k])
                
                    gains = 2 ** tmp_y_true - 1  #When y_true[i] =1, the maximum correlation is 1, otherwise it is 0.
                    discounts = np.log2(np.arange(len(tmp_y_true)) + 2)
                    actual = np.sum(gains / discounts)

                    order2 = np.argsort(y_true)[::-1]
    
                    tmp_y_true = np.take(y_true, order2[:k])
                
                    gains = 2 ** tmp_y_true - 1  #When y_true[i] =1, the maximum correlation is 1, otherwise it is 0.
                    discounts = np.log2(np.arange(len(tmp_y_true)) + 2)
                    best = np.sum(gains / discounts)
                    res["ndcg@{0}".format(k)] = res.get("ndcg@{0}".format(k),0)+actual/best
            elif metric == 'avg_pos':
                tmp_y_true = np.take(y_true, order)#Sort by score
                pos = np.where(tmp_y_true == 1.0)[0][0] #Find the location with label=1.0.
                res['avg_pos']= res.get('avg_pos',0)+ pos
                #order = np.argsort(y_score)[::-1] 
                #print("order",order)
   
    count = len(true_labels)
    for key in res.keys():
        
        res[key]=round(res[key]/count,4)
    return res
                










def pre_process(data,config):
    
    if os.path.exists(config.user2query_path):
        user2query = pkl.load(open(config.user2query_path,'rb'))
        item2query = pkl.load(open(config.item2query_path,'rb'))

        return user2query,item2query 
        
    user2query=defaultdict(dict)
    item2query = defaultdict(dict)
    user2item= defaultdict(set)
    
   
    total_num=len(data)
    for i in range(total_num):
        if data['label'][i]==0: #not the true interaction
            continue
        
        
        user = data['user'][i]
        item=data['item'][i]
        user2item[user].add(item)
        raw_query =  data['query'][i]
        

        query=eval(raw_query)
        for q in query:
            user2query[user][q]=user2query[user].get(q,0)+1
            item2query[item][q] = item2query[item].get(q,0)+1
        
    with open(config.user2item_path, 'wb') as f:
        pkl.dump(user2item, f)
    
         
    
    user2top_query={}
    user2query_count=0
    query_repeat_count=0  
    
    raw_query_count=0
    for user in user2query.keys():
        
        
        raw_query_count+=len(user2query[user]) 
        std_query = dict(sorted(user2query[user].items(),key=lambda x :x[1],reverse=True)[:config.per_query_num])
        user2query_count+= len(std_query)
        query_repeat_count+=sum(std_query.values())
        sorted_userq= list(std_query.keys()) #获得std_query_num 的用户对应的query
        user2top_query[user]=sorted_userq
        
    print("The average number of different query for users:{:.4f},average number of different query for users after truncation:{:.4f},average number of repetitions of query after truncation:{:.4f}".format(raw_query_count/len(user2query),user2query_count/len(user2query),query_repeat_count/user2query_count))
    #print("用户std_query_num的query的词",user2top_query)
    item2top_query={}
    item2query_count=0
    query_repeat_count=0  #average number of repetitions of query words
    raw_query_count=0
    with open(config.user2query_path, 'wb') as f:
        pkl.dump(user2top_query, f) 
    
    
    for item in item2query.keys():
        raw_query_count+=len(item2query[item]) 
        std_query = dict(sorted(item2query[item].items(),key=lambda x :x[1],reverse=True)[:config.item_per_query_num])
        
        item2query_count+= len(std_query)
        query_repeat_count+=sum(std_query.values())
        sorted_itemq= list(std_query.keys()) #获得std_query_num 的用户对应的query
        item2top_query[item]=sorted_itemq
    with open(config.item2query_path, 'wb') as f:
        pkl.dump(item2top_query, f) 
    print("The average number of different query for items:{:.4f},average number of different query for items after truncation:{:.4f},average number of repetitions of query after truncation:{:.4f}".format(raw_query_count/len(item2query),item2query_count/len(item2query),query_repeat_count/item2query_count))

    
    return user2top_query,item2top_query


def load_train_graph(dataset,config):
    # get the graph data from the train dataset
    '''
    Output: [[user,item],query,edge_type,item_query,user_query]
    '''
    per_query_num = config.per_query_num 
    user_query_num =  config.user_per_query_num
    item_query_num = config.item_per_query_num
    train_graph_elements =[[],[],[],[],[]]
    
    
    search_count=0
    total_num = len(dataset)
    
    for i in range(total_num):
        user,query,item,_,_,edge_type,usertopkquery,item_query =dataset[i]

        #search: current queries; recommendation: user's topk queries for generating the current demand
        
        query = query[:per_query_num]+[0]*(per_query_num-len(query)) #train_graph_elements[3].append(item_query)
        
        search_count+=edge_type #1:search 0:recommendation
        item_query = item_query[:item_query_num]+[0]*(item_query_num-len(item_query))
        user_query = usertopkquery[:user_query_num]+[0]*(user_query_num-len(usertopkquery))
        #user_query = usertopkquery[:user_query_num]+
        train_graph_elements[0].append([user,item])
        train_graph_elements[1].append(query)
        train_graph_elements[2].append(edge_type)
        
        train_graph_elements[3].append(item_query)
        train_graph_elements[4].append(user_query)
        
    print("graph!search_count:",search_count,"recommendation_count:",total_num-search_count)
    return train_graph_elements







def load_single_data(data,user2query,item2query,train=False): 
    #load train/valid/test data
    total_num = len(data)
    single_dataset =[]
    recommendation_count=0
    search_count=0
   
    for i in range(total_num):
        query = data['query'][i]
        query = eval(query)
        user = data['user'][i]
        item = data['item'][i]
        #type: #if it's search, type =1 else type =0
        if len(query)>0: #search
            search_count +=1
            type=1 
        else:
            recommendation_count+=1
            type=0
            query = []
        usertopkquery = user2query[user] if user in user2query.keys() else []
        itemtopkquery = item2query[item] if item in item2query.keys() else []
        label = data['label'][i]
        sample_num =0  #for padding the train data
        if not train:
            sample_num = data['sample_num'][i]
        single_dataset.append([user,query,item,sample_num,label,type,usertopkquery,itemtopkquery])
        
    print("search_count:",search_count,"recommendation_count:",recommendation_count)
    return single_dataset



def load_data(config):
    '''
    Input:
    train_data,test_data
    {
            'user':[],
            'query':[],
            'item':[],
            'label':[],
            'sample_num':[] for test
        }
    Output: 
    train_data/test_data: [user,query,item,sample_num,label,type,usertopkquery,itemtopkquery]
    train_global_graph:
        [[user,item],query,edge_type,item_query,user_query]
    '''
    nrow =300 if config.test else None
    train_data = pd.read_csv(config.train_data_path,nrows=nrow)
    valid_data = pd.read_csv(config.valid_data_path,nrows=nrow)
    test_data = pd.read_csv(config.test_data_path,nrows=nrow)
  
    user2query,item2query= pre_process(train_data,config)  #get user's interacting item,user's queries and item's queries in train data
    train_dataset= load_single_data(train_data,user2query,item2query,train=True)
    valid_dataset = load_single_data(valid_data,user2query,item2query,train=False)  
    test_dataset = load_single_data(test_data,user2query,item2query,train=False)

    train_global_graph= load_train_graph(train_dataset,config) #get the graph information


    return train_dataset, valid_dataset,test_dataset,train_global_graph



def build_graph(num_user,num_item,train_graph_elements,sample_graph,device=None):

    
    uqi_u,uqi_i = np.array(train_graph_elements[0]).transpose() 
    
    
    num_nodes_dict={
        'user':num_user,
        'item':num_item,
      
    }
    # interaction data
    graph_data = {
    ('user', 'uqi', 'item'): (uqi_u, uqi_i),
    ('item', 'iqu', 'user'): (uqi_i, uqi_u),
    
    }
    g = dgl.heterograph(graph_data,num_nodes_dict=num_nodes_dict)
    
    #the attributes
    uqi_query = np.array(train_graph_elements[1]) 
    uqi_type= np.array(train_graph_elements[2])
    item_querys = np.array(train_graph_elements[3])
    user_querys = np.array(train_graph_elements[4])
    

    g.edges['uqi'].data['query']=torch.IntTensor(uqi_query)
    g.edges['uqi'].data['type']=torch.IntTensor(uqi_type)
    g.edges['uqi'].data['itemquery']=torch.IntTensor(item_querys)
    g.edges['uqi'].data['userquery']=torch.IntTensor(user_querys)

    if sample_graph:
        g.edges['iqu'].data['query']=torch.IntTensor(uqi_query)
        g.edges['iqu'].data['type']=torch.IntTensor(uqi_type)
        g.edges['iqu'].data['itemquery']=torch.IntTensor(item_querys)
        g.edges['iqu'].data['userquery']=torch.IntTensor(user_querys)
    

    if device:
        g=g.to(device)
    return g

