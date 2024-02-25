# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import torch
import pickle
import argparse
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset, Data_CHLS
from trainers import FinetuneTrainer, DistSAModelTrainer
from models import S3RecModel
from seqmodels import SASRecModel, DistSAModel, DistMeanSAModel
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed, get_data_from_pkl
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def cold_hot_long_short(data_raw, dataset_name):
    item_list = []
    len_list = []
    target_item = []

    for id_temp in data_raw['train']:
        temp_list = data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp]
        len_list.append(len(temp_list))
        target_item.append(data_raw['test'][id_temp][0])
        item_list += temp_list
    item_num_count = Counter(item_list)
    split_num = np.percentile(list(item_num_count.values()), 80)
    cold_item, hot_item = [], []
    for item_num_temp in item_num_count.items():
        if item_num_temp[1] < split_num:
            cold_item.append(item_num_temp[0])
        else:
            hot_item.append(item_num_temp[0])
    cold_ids, hot_ids = [], []
    cold_list, hot_list = [], []
    for id_temp, item_temp in enumerate(data_raw['test'].values()):
        if item_temp[0] in hot_item:
            hot_ids.append(id_temp)
            if dataset_name == 'ml-1m':
                hot_list.append(data_raw['train'][id_temp+1] + data_raw['val'][id_temp+1] + data_raw['test'][id_temp+1])
            else:
                hot_list.append(data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp])
        else:
            cold_ids.append(id_temp)
            if dataset_name == 'ml-1m':
                cold_list.append(data_raw['train'][id_temp+1] + data_raw['val'][id_temp+1] + data_raw['test'][id_temp+1])
            else:
                cold_list.append(data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp])
    cold_hot_dict = {'hot': hot_list, 'cold': cold_list}

    len_short = np.percentile(len_list, 20)
    len_midshort = np.percentile(len_list, 40)
    len_midlong = np.percentile(len_list, 60)
    len_long = np.percentile(len_list, 80)
    
    len_seq_dict = {'short': [], 'mid_short': [], 'mid': [], 'mid_long': [], 'long': []}
    for id_temp, len_temp in enumerate(len_list):
        if dataset_name == 'ml-1m':
            temp_seq = data_raw['train'][id_temp+1] + data_raw['val'][id_temp+1] + data_raw['test'][id_temp+1]
        else:
            temp_seq = data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp]
        if len_temp <= len_short:
            len_seq_dict['short'].append(temp_seq)
        elif len_short < len_temp <= len_midshort:
            len_seq_dict['mid_short'].append(temp_seq)
        elif len_midshort < len_temp <= len_midlong:
            len_seq_dict['mid'].append(temp_seq)
        elif len_midlong < len_temp <= len_long:
            len_seq_dict['mid_long'].append(temp_seq)
        else:
            len_seq_dict['long'].append(temp_seq)
    return cold_hot_dict, len_seq_dict, split_num, [len_short, len_midshort, len_midlong, len_long], len_list, list(item_num_count.values())



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='../../datasets/data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='amazon_beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=10, type=int, help="pretrain epochs 10, 20, 30...")

    # model args
    parser.add_argument("--model_name", default='Finetune_full', type=str)  ## DistSAModel, Finetune_full
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")  ## 64
    parser.add_argument("--num_hidden_layers", type=int, default=1, help="number of layers")  ## 1
    parser.add_argument('--num_attention_heads', default=4, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.0, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument('--distance_metric', default='wasserstein', type=str)
    parser.add_argument('--pvn_weight', default=0.005, type=float)
    parser.add_argument('--kernel_param', default=1.0, type=float)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=1024, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=101, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=1997, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="1", help="gpu_id")
    parser.add_argument('--long_head', default=False, help='Long and short sequence, head and long-tail items')
    parser.add_argument('--diversity_measure', default=False, help='Measure the diversity of recommendation results')
    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    
    # args.data_file = args.data_dir + args.data_name + '.txt'

    args.data_file = args.data_dir + args.data_name + '/dataset.pkl'
    # args.data_file = '../../datasets/data/category/' + args.data_name +'/dataset.pkl'  ## category 

    #item2attribute_file = args.data_dir + args.data_name + '_item2attributes.json'
    with open(args.data_file, 'rb') as f:
        data_raw = pickle.load(f)

    # user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = get_user_seqs(args.data_file)
    user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = get_data_from_pkl(args.data_file)

    #item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.num_users = num_users
    args.mask_id = max_item + 1
    #args.attribute_size = attribute_size + 1

    # save model args
    args_str = f'{args.model_name}-{args.data_name}-{args.hidden_size}-{args.num_hidden_layers}-{args.num_attention_heads}-{args.hidden_act}-{args.attention_probs_dropout_prob}-{args.hidden_dropout_prob}-{args.max_seq_length}-{args.lr}-{args.weight_decay}-{args.ckp}-{args.kernel_param}-{args.pvn_weight}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    #args.item2attribute = item2attribute
    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = SASRecDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
    eval_sampler = SequentialSampler(eval_dataset)
    #eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=200)

    test_dataset = SASRecDataset(args, user_seq, data_type='test')
    test_sampler = SequentialSampler(test_dataset)
    #test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=200)


    if args.model_name == 'DistSAModel':
        model = DistSAModel(args=args)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=100)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)
        trainer = DistSAModelTrainer(model, train_dataloader, eval_dataloader, test_dataloader, args)
    elif args.model_name == 'DistMeanSAModel':
        model = DistMeanSAModel(args=args)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=100)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)
        trainer = DistSAModelTrainer(model, train_dataloader, eval_dataloader,
                                    test_dataloader, args)
    else:
        cold_hot_dict, len_seq_dict, split_hotcold, split_length, list_len, list_num = cold_hot_long_short(data_raw, args.data_name)
        cold_data = Data_CHLS(cold_hot_dict['cold'], args)
        cold_data_loader = cold_data.get_pytorch_dataloaders()

        hot_data = Data_CHLS(cold_hot_dict['hot'], args)
        hot_data_loader = hot_data.get_pytorch_dataloaders()
        

        short_data = Data_CHLS(len_seq_dict['short'], args)
        short_data_loader = short_data.get_pytorch_dataloaders()
        

        mid_short_data = Data_CHLS(len_seq_dict['mid_short'], args)
        mid_short_data_loader = mid_short_data.get_pytorch_dataloaders()
        

        mid_data = Data_CHLS(len_seq_dict['mid'], args)
        mid_data_loader = mid_data.get_pytorch_dataloaders()
        

        mid_long_data = Data_CHLS(len_seq_dict['mid_long'], args)
        mid_long_data_loader = mid_long_data.get_pytorch_dataloaders()
        

        long_data = Data_CHLS(len_seq_dict['long'], args)
        long_data_loader = long_data.get_pytorch_dataloaders()
        
    
        model = SASRecModel(args=args)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)
        
        trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader, test_dataloader, hot_data_loader, cold_data_loader, short_data_loader, mid_short_data_loader, mid_data_loader, mid_long_data_loader, long_data_loader, args)
    
    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info, _ = trainer.test(0, full_sort=True)

    else:
        #pretrained_path = os.path.join(args.output_dir, f'{args.data_name}-epochs-{args.ckp}.pt')
        #try:
        #    trainer.load(pretrained_path)
        #    print(f'Load Checkpoint From {pretrained_path}!')

        #except FileNotFoundError:
        #    print(f'{pretrained_path} Not Found! The Model is same as SASRec')
        
        if args.model_name == 'DistSAModel':
            early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
        else:
            early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
        
        time_avg_epoch = []
        for epoch in range(args.epochs):
            time_start = time.time()
            trainer.train(epoch)
            time_end = time.time()
            time_avg_epoch.append(time_end - time_start)
            if epoch % 10 == 0:
                print('Averager training time (one epoch): ', np.mean(time_avg_epoch))

            # evaluate on MRR
            scores, _, _ = trainer.valid(epoch, full_sort=True)
            # early_stopping(np.array(scores[-1:]), trainer.model)
            early_stopping(np.array(list(scores.values())[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        # print('---------------Valid----------------------------')
        # trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        # valid_scores, _, _ = trainer.valid('best', full_sort=True)
        print('-------------------Test----------------------------')
        trainer.args.train_matrix = test_rating_matrix
        scores, result_info, _ = trainer.test('best', full_sort=True)
    
    print('Test:--------------------------------------')
    print(scores)
    
    print(args_str)
    #print(result_info)
    if args.long_head:
        
        print('--------------Cold item-----------------------')
        scores, result_info, _ = trainer.cold('best', full_sort=True)
        print('--------------hot item-----------------------')
        scores, result_info, _ = trainer.hot('best', full_sort=True)
        print('--------------Short-----------------------')
        scores, result_info, _ = trainer.short('best', full_sort=True)
        print('--------------Mid_short-----------------------')
        scores, result_info, _ = trainer.mid_short('best', full_sort=True)
        print('--------------Mid-----------------------')
        scores, result_info, _ = trainer.mid('best', full_sort=True)
        print('--------------Mid_long-----------------------')
        scores, result_info, _ = trainer.mid_long('best', full_sort=True)
        print('--------------Long-----------------------')
        scores, result_info, _ = trainer.longg('best', full_sort=True)
        
 
main()
