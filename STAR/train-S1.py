# import faulthandler
# faulthandler.enable()
# learn slot-wise weight
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import json
import time
import logging
from tqdm import tqdm, trange

from torch.utils.data import DataLoader, RandomSampler
from utils.data_utils import Processor, MultiWozDataset
from utils.eval_utils import model_evaluation
from utils.loss_utils import *
from utils.label_lookup import get_label_lookup_from_first_token
from models.DST import UtteranceEncoding, BeliefTracker

#====================================
import higher
import itertools
from models.WeightNet import SlotWeight
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
#====================================

import transformers
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup as get_linear_schedule_with_warmup_T

os.environ['CUDA_VISIBLE_DEVICES']='1'
# torch.cuda.set_device(0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_linear_schedule_with_warmup(optimizer, enc_num_warmup_steps, dec_num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    see https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/optimization.py#L75
    """
    def enc_lr_lambda(current_step: int):
        if current_step < enc_num_warmup_steps:
            return float(current_step) / float(max(1, enc_num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - enc_num_warmup_steps))
        )
    
    def dec_lr_lambda(current_step: int):
        if current_step < dec_num_warmup_steps:
            return float(current_step) / float(max(1, dec_num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - dec_num_warmup_steps))
        )

    return LambdaLR(optimizer, [enc_lr_lambda, enc_lr_lambda, dec_lr_lambda], last_epoch)

def set_seed(args, device):
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if device == "cuda":
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
def get_sv_lookup(slot_meta, ontology, tokenizer, sv_encoder, device):
    slot_lookup = get_label_lookup_from_first_token(slot_meta, tokenizer, sv_encoder, device)
    value_lookup = []
    for slot in ontology.keys():
        value_lookup.append(get_label_lookup_from_first_token(ontology[slot], tokenizer, sv_encoder, device))
    return slot_lookup, value_lookup

def prepare_optimizer(model, enc_learning_rate, dec_learning_rate, num_train_steps, enc_warmup_ratio, dec_warmup_ratio):
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    enc_param_optimizer = list(model.encoder.named_parameters())
    dec_param_optimizer = list(model.decoder.parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in enc_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in enc_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': dec_param_optimizer, 'lr': dec_learning_rate}
        ]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=enc_learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_steps * enc_warmup_ratio), 
                                                int(num_train_steps * dec_warmup_ratio), num_train_steps)
    print(f'Number of parameter groups: {len(optimizer.param_groups)}')
    return optimizer, scheduler

def get_unreduced_loss(slot_output, value_lookup, label_ids, pseudo_label_ids):
    _, pred_all_distance = slot_value_matching(slot_output, value_lookup)
                
    loss_slot_gt = unreduced_cross_entropy_loss(pred_all_distance, label_ids)
    loss_slot_pseudo = unreduced_cross_entropy_loss(pred_all_distance, pseudo_label_ids)
    
    return loss_slot_gt, loss_slot_pseudo

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # logger
    logger_file_name = args.save_dir.split('/')[1]
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, "%s.txt"%(logger_file_name)))
    logger.addHandler(fileHandler)
    logger.info(args)
    
    # cuda setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("device: {}".format(device))
    
    # set random seed
    set_seed(args, device)

    #******************************************************
    # load data
    #******************************************************
    processor = Processor(args)
    slot_meta = processor.slot_meta
    ontology = processor.ontology
    logger.info(slot_meta)
   
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)
    
    if args.do_train:
        train_data_raw = processor.get_instances(args.data_dir, args.train_data, tokenizer, True)
        print("# train examples %d" % len(train_data_raw))
        
        meta_data_raw = processor.get_instances(args.data_dir, args.dev_data, tokenizer)
        print("# meta examples %d" % len(meta_data_raw))
        
        dev_data_raw = processor.get_instances(args.data_dir, args.dev_data, tokenizer)
        print("# dev examples %d" % len(dev_data_raw))
    
    test_data_raw = processor.get_instances(args.data_dir, args.test_data, tokenizer)
    print("# test examples %d" % len(test_data_raw))
    logger.info("Data loaded!")
    
    ## Initialize slot and value embeddings
    sv_encoder = UtteranceEncoding.from_pretrained(args.pretrained_model)
    for p in sv_encoder.bert.parameters():
        p.requires_grad = False  
    slot_lookup, value_lookup = get_sv_lookup(slot_meta, ontology, tokenizer, sv_encoder, device)
    
    if args.do_train:
        train_data = MultiWozDataset(train_data_raw,
                                     tokenizer,
                                     word_dropout=args.word_dropout,
                                     max_seq_length=args.max_seq_length,
                                     use_pseudo_label=True)
        meta_data = MultiWozDataset(meta_data_raw,
                                    tokenizer,
                                    word_dropout=0.0, # do word dropout here???
                                    max_seq_length=args.max_seq_length) 

        num_train_steps = int(len(train_data_raw) / args.train_batch_size * args.n_epochs)
        logger.info("***** Run training *****")
        logger.info(" Num examples = %d", len(train_data_raw))
        logger.info(" Batch size = %d", args.train_batch_size)
        logger.info(" Num steps = %d", num_train_steps)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size,
                                      collate_fn=train_data.collate_fn)
        
        meta_sampler = RandomSampler(meta_data)
        meta_dataloader = DataLoader(meta_data,
                                     sampler=meta_sampler,
                                     batch_size=args.meta_batch_size,
                                     collate_fn=meta_data.collate_fn)
        meta_dataloader = itertools.cycle(meta_dataloader)
        
        #******************************************************
        # build model
        #******************************************************
        ## model initialization
        base_model = BeliefTracker(args.pretrained_model, args.attn_head, dropout_prob=args.dropout_prob,
                                   num_self_attention_layer=args.num_self_attention_layer)
        base_model.to(device)
        
        meta_model = BeliefTracker(args.pretrained_model, args.attn_head, dropout_prob=args.dropout_prob,
                                   num_self_attention_layer=args.num_self_attention_layer)
        meta_model.to(device)
        
        # Number of slots
        SW = SlotWeight(len(slot_meta), init_val=np.log(args.init_weight/(1.0 - args.init_weight)))
        SW.to(device)

        ## prepare optimizer
        base_optimizer, base_scheduler = \
        prepare_optimizer(base_model, args.enc_lr, args.dec_lr, num_train_steps, args.enc_warmup, args.dec_warmup)
        logger.info(base_optimizer)
        # meta model is a copy of the base model, thus shares the optimizer and scheduler
        meta_optimizer, meta_scheduler = \
        prepare_optimizer(meta_model, args.enc_lr, args.dec_lr, num_train_steps, args.enc_warmup, args.dec_warmup)

        sw_param_optimizer = list(SW.parameters())
        sw_optimizer = optim.AdamW(sw_param_optimizer, lr=args.sw_lr)
        sw_scheduler = get_linear_schedule_with_warmup_T(sw_optimizer, 
                                                         int(num_train_steps * args.sw_warmup), 
                                                         num_train_steps)
        
        #******************************************************
        # training
        #******************************************************
        logger.info("Training...")

        best_loss = None
        best_acc = None
        last_update = None

        for epoch in trange(int(args.n_epochs), desc="Epoch"):       
            batch_loss, meta_batch_loss = [], []
            for step, batch in enumerate(tqdm(train_dataloader)):
                base_model.train()

                batch = [b.to(device) for b in batch]
                input_ids, segment_ids, input_mask, label_ids, pseudo_label_ids = batch
                
                # forward (meta model)
                meta_model.load_state_dict(base_model.state_dict())
                meta_optimizer.load_state_dict(base_optimizer.state_dict())
                meta_optimizer.zero_grad()
                with higher.innerloop_ctx(meta_model, meta_optimizer) as (meta_m, meta_opt):
                    meta_m.train()
                    slot_output = meta_m(input_ids=input_ids, 
                                         attention_mask=input_mask, 
                                         token_type_ids=segment_ids, 
                                         slot_emb=slot_lookup) # [batch_size, num_slots, dim]
                    
                    loss_slot_gt, loss_slot_pseudo = \
                    get_unreduced_loss(slot_output, value_lookup, label_ids, pseudo_label_ids)
                    
                    s_weight = SW()
                
                    meta_loss = torch.sum((1.0-s_weight)*loss_slot_gt + s_weight*loss_slot_pseudo) / loss_slot_gt.size(0)
                    # first backward
                    meta_opt.step(meta_loss)
                    
                    # compute on the meta validation set
                    batch_v = next(meta_dataloader)
                    batch_v = [b.to(device) for b in batch_v]
                    input_ids_v, segment_ids_v, input_mask_v, label_ids_v = batch_v
                    # second forward
                    meta_m.eval() # disable dropout
                    slot_output_v = meta_m(input_ids=input_ids_v, 
                                           attention_mask=input_mask_v, 
                                           token_type_ids=segment_ids_v, 
                                           slot_emb=slot_lookup) # [batch_size, num_slots, dim]
                    _, pred_all_distance = slot_value_matching(slot_output_v, value_lookup)
                    loss_v, _, _ = hard_cross_entropy_loss(pred_all_distance, label_ids_v)
                    # backward over backward
                    sw_optimizer.zero_grad()
                    loss_v.backward()
                    sw_optimizer.step()
                    sw_scheduler.step()
                    meta_batch_loss.append(loss_v.item())
                
                # Now we have the updated weight net  
                # forward (base model)
                slot_output = base_model(input_ids=input_ids, 
                                         attention_mask=input_mask, 
                                         token_type_ids=segment_ids, 
                                         slot_emb=slot_lookup) # [batch_size, num_slots, dim]

                loss_slot_gt, loss_slot_pseudo = \
                get_unreduced_loss(slot_output, value_lookup, label_ids, pseudo_label_ids)
                with torch.no_grad():    
                    s_weight = SW()

                loss = torch.sum((1.0-s_weight)*loss_slot_gt + s_weight*loss_slot_pseudo) / loss_slot_gt.size(0)
                # backward (base model)
                base_optimizer.zero_grad()
                loss.backward()
                base_optimizer.step()
                base_scheduler.step()

                batch_loss.append(loss.item())
                if step % 300 == 0:
                    logger.info("[%d/%d] [%d/%d] mean_loss: %.6f mean_meta_loss: %.6f" % \
                               (epoch+1, args.n_epochs, step, len(train_dataloader), 
                                np.mean(batch_loss), np.mean(meta_batch_loss)))
                    batch_loss, meta_batch_loss = [], []
                    logger.info(f'Slot weights: {s_weight.cpu().numpy()}')

            if (epoch+1) % args.eval_epoch == 0:
                eval_res = model_evaluation(base_model, dev_data_raw, tokenizer, 
                                            slot_lookup, value_lookup, ontology, epoch+1)
                if last_update is None or best_loss > eval_res['loss']:
                    best_loss = eval_res['loss']
#                     save_path = os.path.join(args.save_dir, 'model_best_loss.bin')
#                     torch.save(base_model.state_dict(), save_path)
                    print("Best Loss : ", best_loss)
                    print("\n")
                if last_update is None or best_acc < eval_res['joint_acc']:
                    best_acc = eval_res['joint_acc']
                    save_path = os.path.join(args.save_dir, 'model_best_acc.bin')
                    save_path_w = os.path.join(args.save_dir, 'sw.bin')
                    torch.save(base_model.state_dict(), save_path)
                    torch.save(SW.state_dict(), save_path_w)
                    last_update = epoch
                    print("Best Acc : ", best_acc)
                    print("\n")

                logger.info("*** Epoch=%d, Last Update=%d, Dev Loss=%.6f, Dev Acc=%.6f, Dev Turn Acc=%.6f, Best Loss=%.6f, Best Acc=%.6f ***" % (epoch, last_update, eval_res['loss'], eval_res['joint_acc'], eval_res['joint_turn_acc'], best_loss, best_acc))

            if (epoch+1) % args.eval_epoch == 0:
                eval_res = model_evaluation(base_model, test_data_raw, tokenizer, 
                                            slot_lookup, value_lookup, ontology, epoch+1)

                logger.info("*** Epoch=%d, Last Update=%d, Tes Loss=%.6f, Tes Acc=%.6f, Tes Turn Acc=%.6f, Best Loss=%.6f, Best Acc=%.6f ***" % (epoch, last_update, eval_res['loss'], eval_res['joint_acc'], eval_res['joint_turn_acc'], best_loss, best_acc))

            if last_update + args.patience <= epoch:
                    break

#         print("Test using best loss model...")
#         best_epoch = 0
#         ckpt_path = os.path.join(args.save_dir, 'model_best_loss.bin')
#         model = BeliefTracker(args.pretrained_model, args.attn_head, dropout_prob=args.dropout_prob,
#                               num_self_attention_layer=args.num_self_attention_layer)
#         ckpt = torch.load(ckpt_path, map_location='cpu')
#         model.load_state_dict(ckpt)
#         model.to(device)

#         test_res = model_evaluation(model, test_data_raw, tokenizer, slot_lookup, value_lookup,
#                                     ontology, best_epoch, is_gt_p_state=False)
#         logger.info("Results based on best loss: ")
#         logger.info(test_res)
    #----------------------------------------------------------------------
    print("Test using best acc model...")
    best_epoch = 1
    ckpt_path = os.path.join(args.save_dir, 'model_best_acc.bin')
    model = BeliefTracker(args.pretrained_model, args.attn_head, dropout_prob=args.dropout_prob,
                          num_self_attention_layer=args.num_self_attention_layer)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.to(device)

    test_res = model_evaluation(model, test_data_raw, tokenizer, slot_lookup, value_lookup, 
                                ontology, best_epoch, is_gt_p_state=False)
    logger.info("Results based on best acc: ")
    logger.info(test_res)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default='data/mwz2.4', type=str)
    parser.add_argument("--train_data", default='train_dials_v2.json', type=str)
    parser.add_argument("--dev_data", default='dev_dials_v2.json', type=str)
    parser.add_argument("--test_data", default='test_dials_v2.json', type=str)
    parser.add_argument("--pretrained_model", default='bert-base-uncased', type=str)
    parser.add_argument("--save_dir", default='output-meta24-S1/exp', type=str)

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--meta_batch_size", default=8, type=int)
    parser.add_argument("--enc_warmup", default=0.1, type=float)
    parser.add_argument("--dec_warmup", default=0.1, type=float)
    parser.add_argument("--sw_warmup", default=0.1, type=float)
    parser.add_argument("--enc_lr", default=4e-5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--sw_lr", default=5e-5, type=float)
    parser.add_argument("--init_weight", default=0.5, type=float)
    parser.add_argument("--n_epochs", default=15, type=int)
    parser.add_argument("--eval_epoch", default=1, type=int)
    parser.add_argument("--eval_step", default=100000, type=int)

    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--word_dropout", default=0.1, type=float)
    
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--patience", default=6, type=int)
    parser.add_argument("--attn_head", default=4, type=int)
    parser.add_argument("--num_history", default=20, type=int)
    parser.add_argument("--num_self_attention_layer", default=6, type=int)
    
    parser.add_argument("--do_train", action='store_true')
       
    args = parser.parse_args()
    
    print('pytorch version: ', torch.__version__)
    args.torch_version = torch.__version__
    args.transformers_version = transformers.__version__
    args.save_dir = args.save_dir + \
    f'-sd{args.random_seed}-bz{args.train_batch_size}-{args.meta_batch_size}-lr{args.enc_lr}-{args.dec_lr}-{args.sw_lr}-ep{args.n_epochs}'

    main(args)
