import os
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import logging
import time
import pickle
from utils import Data_Train, Data_Val, Data_Test, Data_CHLS
from model import create_model_diffu, Att_Diffuse_model
from trainer import model_train, LSHT_inference
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='amazon_beauty', help='Dataset name: toys, amazon_beauty, steam, ml-1m')
parser.add_argument('--log_file', default='log/', help='log dir path')
parser.add_argument('--random_seed', type=int, default=1997, help='Random seed')  
parser.add_argument('--max_len', type=int, default=50, help='The max length of sequence')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--batch_size', type=int, default=512, help='Batch Size')  
parser.add_argument("--hidden_size", default=128, type=int, help="hidden size of model")
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout of representation')
parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout of item embedding')
parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
parser.add_argument('--num_blocks', type=int, default=4, help='Number of Transformer blocks')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs for training')  ## 500
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--loss_lambda', type=float, default=0.001, help='loss weight for diffusion')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--schedule_sampler_name', type=str, default='lossaware', help='Diffusion for t generation')
parser.add_argument('--diffusion_steps', type=int, default=32, help='Diffusion step')
parser.add_argument('--lambda_uncertainty', type=float, default=0.001, help='uncertainty weight')
parser.add_argument('--noise_schedule', default='trunc_lin', help='Beta generation')  ## cosine, linear, trunc_cos, trunc_lin, pw_lin, sqrt
parser.add_argument('--rescale_timesteps', default=True, help='rescal timesteps')
parser.add_argument('--eval_interval', type=int, default=20, help='the number of epoch to eval')
parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop')
parser.add_argument('--description', type=str, default='Diffu_norm_score', help='Model brief introduction')
parser.add_argument('--long_head', default=False, help='Long and short sequence, head and long-tail items')
parser.add_argument('--diversity_measure', default=False, help='Measure the diversity of recommendation results')
parser.add_argument('--epoch_time_avg', default=False, help='Calculate the average time of one epoch training')
args = parser.parse_args()

print(args)

if not os.path.exists(args.log_file):
    os.makedirs(args.log_file)
if not os.path.exists(args.log_file + args.dataset):
    os.makedirs(args.log_file + args.dataset )


logging.basicConfig(level=logging.INFO, filename=args.log_file + args.dataset + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)
logger.info(args)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def item_num_create(args, item_num):
    args.item_num = item_num
    return args


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


def main(args):    
    fix_random_seed_as(args.random_seed)
    path_data = '../datasets/data/' + args.dataset + '/dataset.pkl'
    with open(path_data, 'rb') as f:
        data_raw = pickle.load(f)
    
    # cold_hot_long_short(data_raw, args.dataset)
    
    args = item_num_create(args, len(data_raw['smap']))
    tra_data = Data_Train(data_raw['train'], args)
    val_data = Data_Val(data_raw['train'], data_raw['val'], args)
    test_data = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()
    diffu_rec = create_model_diffu(args)
    rec_diffu_joint_model = Att_Diffuse_model(diffu_rec, args)
    
    best_model, test_results = model_train(tra_data_loader, val_data_loader, test_data_loader, rec_diffu_joint_model, args, logger)


    if args.long_head:
        cold_hot_dict, len_seq_dict, split_hotcold, split_length, list_len, list_num = cold_hot_long_short(data_raw, args.dataset)
        cold_data = Data_CHLS(cold_hot_dict['cold'], args)
        cold_data_loader = cold_data.get_pytorch_dataloaders()
        print('--------------Cold item-----------------------')
        LSHT_inference(best_model, args, cold_data_loader)

        hot_data = Data_CHLS(cold_hot_dict['hot'], args)
        hot_data_loader = hot_data.get_pytorch_dataloaders()
        print('--------------hot item-----------------------')
        LSHT_inference(best_model, args, hot_data_loader)

        short_data = Data_CHLS(len_seq_dict['short'], args)
        short_data_loader = short_data.get_pytorch_dataloaders()
        print('--------------Short-----------------------')
        LSHT_inference(best_model, args, short_data_loader)

        mid_short_data = Data_CHLS(len_seq_dict['mid_short'], args)
        mid_short_data_loader = mid_short_data.get_pytorch_dataloaders()
        print('--------------Mid_short-----------------------')
        LSHT_inference(best_model, args, mid_short_data_loader)

        mid_data = Data_CHLS(len_seq_dict['mid'], args)
        mid_data_loader = mid_data.get_pytorch_dataloaders()
        print('--------------Mid-----------------------')
        LSHT_inference(best_model, args, mid_data_loader)

        mid_long_data = Data_CHLS(len_seq_dict['mid_long'], args)
        mid_long_data_loader = mid_long_data.get_pytorch_dataloaders()
        print('--------------Mid_long-----------------------')
        LSHT_inference(best_model, args, mid_long_data_loader)

        long_data = Data_CHLS(len_seq_dict['long'], args)
        long_data_loader = long_data.get_pytorch_dataloaders()
        print('--------------Long-----------------------')
        LSHT_inference(best_model, args, long_data_loader)
    

if __name__ == '__main__':
    main(args)
