import argparse
import inspect
import torch



class Config:
    test = False
    random_seed=0
    base_data = "UDITSR_data"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    epochs = 100
    
    
    sample_graph = False
        
    if device == torch.device('cpu'):
        print("you use cpu now!!!")
    batch_size = 256
    test_batchsize= 20000 #speed up inference
    dropout =0
    #---------hyparameter---------
    cst_lambda =0.2  #intention translation loss
    sg_lambda = 1.5  #seach-supervised loss
    emb_dim=100
    layer_num=2
    lr=1e-4

    per_query_num = 3
    user_per_query_num =3
    item_per_query_num=10
    weight_decay=1e-5
    
    
    
    pair_metrics =["hit@5","ndcg@5","mrr","auc","avg_pos"]
    
    #-------- the dataset information
    user_num=56887
    item_num=4059
    word_num=5000
    per_item_num = 10
    dir = "../"
    save_model_dir = dir+"save_model/"+base_data+"/"
    
    data_dir = dir+base_data+"/"
    
    train_data_path =data_dir+'train_data.csv'
    valid_data_path = data_dir+'valid_data.csv'
    test_data_path = data_dir+'test_data.csv'

    item2query_path = data_dir+'item2query.pkl'
    user2query_path = data_dir+'user2query.pkl'
    user2item_path = data_dir+'user2item.pkl'
    

    def __init__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))

        parser = argparse.ArgumentParser()
        for key, val in attributes:
            parser.add_argument('--' + key, dest=key, type=type(val), default=val)
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)

    def __str__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))
        to_str = ''
        for key, val in attributes:
            to_str += '{} = {}\n'.format(key, val)
        return to_str
