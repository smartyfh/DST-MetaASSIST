"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
from model_meta import SomDST
import transformers
from transformers import BertPreTrainedModel, BertModel
from transformers import BertTokenizer, BertConfig
from transformers import get_linear_schedule_with_warmup as get_linear_schedule_with_warmup_T
from utils.data_utils import prepare_dataset, MultiWozDataset
from utils.data_utils import OP_SET, make_turn_label, postprocessing
from utils.eval_utils import compute_prf, compute_acc, per_domain_join_accuracy
from utils.model_eval import model_evaluation

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import argparse
import random
from tqdm import tqdm, trange

#====================================
import higher
import itertools
from WeightNet import SlotWeight
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
#====================================

import json
import time
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fnc = nn.CrossEntropyLoss(reduction='none')
loss_fnc2 = nn.CrossEntropyLoss()


def set_seed(args):
    n_gpu = 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    print(n_gpu)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    rng = random.Random(args.random_seed)
    torch.manual_seed(args.random_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return rng, n_gpu

 
def main(args):
    def worker_init_fn(worker_id):
        np.random.seed(args.random_seed + worker_id)
        
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
     # logger
    logger_file_name = "logging"
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, "%s.txt"%(logger_file_name)))
    logger.addHandler(fileHandler)
    logger.info(args)

    # set seed
    rng, n_gpu = set_seed(args)

    ontology = json.load(open(args.ontology_data))
    slot_meta = list(ontology.keys())
    logger.info(slot_meta)
    
    op2id = OP_SET[args.op_code]
    print(op2id)
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=True)
    
    # data
    test_data_raw = prepare_dataset(data_path=args.test_data_path,
                                    tokenizer=tokenizer,
                                    slot_meta=slot_meta,
                                    n_history=args.n_history,
                                    max_seq_length=args.max_seq_length,
                                    op_code=args.op_code)
    print("# test examples %d" % len(test_data_raw))

    model_config = BertConfig.from_json_file(args.bert_config_path)
    model_config.dropout = args.dropout
    model_config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    model_config.hidden_dropout_prob = args.hidden_dropout_prob
    
    print("Test using best model...")
    best_epoch = 0
    ckpt_path = os.path.join(args.save_dir, 'model_best.bin')
    model = SomDST(model_config, len(op2id), op2id['update'])
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)

    
    res = model_evaluation(model, test_data_raw, tokenizer, slot_meta, best_epoch, device, args.op_code,
                           is_gt_op=False, is_gt_p_state=False, is_gt_gen=False)
    logger.info(res)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_root", default='data/mwz2.0', type=str)
    parser.add_argument("--train_data", default='train_dials_v2.json', type=str)
    parser.add_argument("--dev_data", default='dev_dials_v2.json', type=str)
    parser.add_argument("--test_data", default='test_dials_v2.json', type=str)
    parser.add_argument("--ontology_data", default='ontology-modified.json', type=str)
    parser.add_argument("--vocab_path", default='assets/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='assets/bert_config_base_uncased.json', type=str)
    parser.add_argument("--save_dir", default='output-meta20-v4/exp', type=str)

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--meta_batch_size", default=8, type=int)
    parser.add_argument("--enc_warmup", default=0.1, type=float)
    parser.add_argument("--dec_warmup", default=0.1, type=float)
    parser.add_argument("--wnet_warmup", default=0.1, type=float)
    parser.add_argument("--init_weight", default=0.4, type=float)
    parser.add_argument("--enc_lr", default=4e-5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--wnet_lr", default=8e-6, type=float)
    parser.add_argument("--n_epochs", default=25, type=int) #####
    parser.add_argument("--eval_epoch", default=1, type=int)

    parser.add_argument("--op_code", default="4", type=str)
    parser.add_argument("--slot_token", default="[SLOT]", type=str)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float)
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float)
    parser.add_argument("--decoder_teacher_forcing", default=0.5, type=float)
    parser.add_argument("--word_dropout", default=0.1, type=float)
    parser.add_argument("--not_shuffle_state", default=True, action='store_false')
    parser.add_argument("--shuffle_p", default=0.0, type=float)

    parser.add_argument("--n_history", default=1, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--msg", default=None, type=str)
    
    parser.add_argument("--do_train", action='store_true')
    
    args = parser.parse_args()
    args.train_data_path = os.path.join(args.data_root, args.train_data)
    args.dev_data_path = os.path.join(args.data_root, args.dev_data)
    args.test_data_path = os.path.join(args.data_root, args.test_data)
    args.ontology_data = os.path.join(args.data_root, args.ontology_data)
    args.shuffle_state = False if args.not_shuffle_state else True
    assert args.shuffle_state == False # must deactivate shuffle_state, we need to keep slots ordered
    
    print('pytorch version: ', torch.__version__)
    args.torch_version = torch.__version__
    args.transformers_version = transformers.__version__
    args.save_dir = args.save_dir + \
    f'-sd{args.random_seed}-bz{args.batch_size}-{args.meta_batch_size}-lr{args.enc_lr}-{args.dec_lr}-{args.wnet_lr}-ep{args.n_epochs}'
    
    main(args)
