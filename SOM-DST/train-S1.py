"""
SOM-DST
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
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


def masked_cross_entropy_for_value_full(logits, target, op_ids, update_id, num_slots, pad_idx=0):
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    deno = mask.sum(-1).float() # B, n
    deno[deno==0.0] = 1e-9
    loss_unreduced = torch.div(losses.sum(-1), deno)
    src = torch.masked_select(loss_unreduced, mask.sum(-1).gt(0))
    mask_op = op_ids.eq(update_id)
    loss_full = torch.zeros(loss_unreduced.size(0), num_slots).to(logits.device)
    loss_full.masked_scatter_(mask_op, src)
    deno2 = mask_op.float().sum(-1).view(-1, 1)
    deno2[deno2==0.0] = 1e-9
    loss_full = torch.div(loss_full, deno2)
    if mask_op.sum() != len(src):
        print(mask_op.sum(), len(src), mask.sum(-1).gt(0).sum(), mask.sum())
    assert mask_op.sum() == len(src)
    return loss_full # B, J


def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / max(mask.sum().float(), 1e-9)
    return loss


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


def get_unreduced_loss(state_scores, op2id, slot_meta, tokenizer, op_ids, gen_scores, gen_ids):
    loss_s_full = loss_fnc(state_scores.view(-1, len(op2id)), op_ids.view(-1))
#     print(loss_s_full)
#     print(len(slot_meta))
    loss_s_full = loss_s_full.view(*op_ids.size()) / len(slot_meta) # B, J
    if gen_ids is not None:
        loss_g_full = masked_cross_entropy_for_value_full(gen_scores.contiguous(),
                                                          gen_ids.contiguous(),
                                                          op_ids,
                                                          op2id['update'], 
                                                          len(slot_meta), 
                                                          tokenizer.vocab['[PAD]'])
        loss_slot = loss_s_full + loss_g_full # B, J
    else:
        loss_slot = loss_s_full
#     print(loss_s_full)
#     print(loss_g_full)
    return loss_slot


def get_unreduced_losses(state_scores, op2id, slot_meta, tokenizer, op_ids, gen_scores, gen_ids, 
                         op_ids_pseudo, gen_scores_pseudo, gen_ids_pseudo):
    loss_slot_gt = get_unreduced_loss(state_scores, op2id, slot_meta, tokenizer, op_ids, gen_scores, gen_ids) # B, J
    
    loss_slot_pseudo = get_unreduced_loss(state_scores, op2id, slot_meta, tokenizer, 
                                          op_ids_pseudo, gen_scores_pseudo, gen_ids_pseudo)

    return loss_slot_gt, loss_slot_pseudo

    
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
    if args.do_train:
        train_data_raw = prepare_dataset(data_path=args.train_data_path,
                                         tokenizer=tokenizer,
                                         slot_meta=slot_meta,
                                         n_history=args.n_history,
                                         max_seq_length=args.max_seq_length,
                                         op_code=args.op_code,
                                         use_pseudo=True)
        train_data = MultiWozDataset(train_data_raw,
                                     tokenizer,
                                     slot_meta,
                                     args.max_seq_length,
                                     rng,
                                     ontology,
                                     args.word_dropout,
                                     args.shuffle_state,
                                     args.shuffle_p,
                                     use_pseudo_label=True)
        print("# train examples %d" % len(train_data_raw))
        
        meta_data_raw = prepare_dataset(data_path=args.dev_data_path,
                                        tokenizer=tokenizer,
                                        slot_meta=slot_meta,
                                        n_history=args.n_history,
                                        max_seq_length=args.max_seq_length,
                                        op_code=args.op_code)
        meta_data = MultiWozDataset(meta_data_raw,
                                    tokenizer,
                                    slot_meta,
                                    args.max_seq_length,
                                    rng,
                                    ontology,
                                    0.0, # no word dropout
                                    args.shuffle_state,
                                    args.shuffle_p)
        print("# meta examples %d" % len(meta_data_raw))

        dev_data_raw = prepare_dataset(data_path=args.dev_data_path,
                                       tokenizer=tokenizer,
                                       slot_meta=slot_meta,
                                       n_history=args.n_history,
                                       max_seq_length=args.max_seq_length,
                                       op_code=args.op_code)
        print("# dev examples %d" % len(dev_data_raw))

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
    
    if args.do_train:
        num_train_steps = int(len(train_data_raw) / args.batch_size * args.n_epochs)
        logger.info("***** Running training *****")
        logger.info(" Num examples = %d", len(train_data_raw))
        logger.info(" Batch size = %d", args.batch_size)
        logger.info(" Num steps = %d", num_train_steps)
    
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=args.batch_size,
                                      collate_fn=train_data.collate_fn,
                                      num_workers=args.num_workers,
                                      worker_init_fn=worker_init_fn)
        
        meta_sampler = RandomSampler(meta_data)
        meta_dataloader = DataLoader(meta_data,
                                     sampler=meta_sampler,
                                     batch_size=args.meta_batch_size,
                                     collate_fn=meta_data.collate_fn,
                                     num_workers=args.num_workers,
                                     worker_init_fn=worker_init_fn)
        meta_dataloader = itertools.cycle(meta_dataloader)
        
        # build model
        base_model = SomDST(model_config, len(op2id), op2id['update'])

        # re-initialize added special tokens ([SLOT], [NULL], [EOS])
        base_model.encoder.bert.embeddings.word_embeddings.weight.data[1].normal_(mean=0.0, std=0.02)
        base_model.encoder.bert.embeddings.word_embeddings.weight.data[2].normal_(mean=0.0, std=0.02)
        base_model.encoder.bert.embeddings.word_embeddings.weight.data[3].normal_(mean=0.0, std=0.02)
        base_model.to(device)
        
        meta_model = SomDST(model_config, len(op2id), op2id['update'])
        meta_model.to(device)
        
        # Number of slots
        SW = SlotWeight(len(slot_meta), init_val=np.log(args.init_weight/(1.0 - args.init_weight)))
        SW.to(device)

        # optimizer
        base_optimizer, base_scheduler = \
        prepare_optimizer(base_model, args.enc_lr, args.dec_lr, num_train_steps, args.enc_warmup, args.dec_warmup)
        logger.info(base_optimizer)
        # meta model is a copy of the base model, thus shares the optimizer and scheduler
        meta_optimizer, meta_scheduler = \
        prepare_optimizer(meta_model, args.enc_lr, args.dec_lr, num_train_steps, args.enc_warmup, args.dec_warmup)
        
        wnet_param_optimizer = list(SW.parameters())
        wnet_optimizer = optim.AdamW(wnet_param_optimizer, lr=args.wnet_lr)
        wnet_scheduler = get_linear_schedule_with_warmup_T(wnet_optimizer, 
                                                           int(num_train_steps * args.wnet_warmup), 
                                                           num_train_steps)
#         if n_gpu > 1:
#             model = torch.nn.DataParallel(model)
        # training
        best_score = {'epoch': 0, 'joint_acc': 0, 'op_acc': 0, 'final_slot_f1': 0}
        logger.info("Training...")
        for epoch in trange(int(args.n_epochs), desc="Epoch"):
            batch_loss, meta_batch_loss = [], []
            for step, batch in enumerate(tqdm(train_dataloader)):
                base_model.train()
                
                batch = [b.to(device) if not isinstance(b, int) and b is not None else b for b in batch]
                input_ids, input_mask, segment_ids, state_position_ids, op_ids, gen_ids, max_value, max_update, \
                op_ids_pseudo, gen_ids_pseudo, max_value_pseudo, max_update_pseudo = batch

                if rng.random() < args.decoder_teacher_forcing:  # teacher forcing
                    teacher = gen_ids
                    teacher_pseudo = gen_ids_pseudo
                else:
                    teacher = None
                    teacher_pseudo = None
                
                # forward (meta model)
                meta_model.load_state_dict(base_model.state_dict())
                meta_optimizer.load_state_dict(base_optimizer.state_dict())
                meta_optimizer.zero_grad()
                with higher.innerloop_ctx(meta_model, meta_optimizer) as (meta_m, meta_opt):
                    meta_m.train()
                    with torch.backends.cudnn.flags(enabled=False):
                        model_output_all = meta_m(input_ids=input_ids,
                                                  token_type_ids=segment_ids,
                                                  state_positions=state_position_ids,
                                                  attention_mask=input_mask,
                                                  max_value=max_value,
                                                  op_ids=op_ids,
                                                  max_update=max_update,
                                                  teacher=teacher,
                                                  use_pseudo=True,
                                                  max_value_pseudo=max_value_pseudo,
                                                  op_ids_pseudo=op_ids_pseudo,
                                                  max_update_pseudo=max_update_pseudo,
                                                  teacher_pseudo=teacher_pseudo)

                    state_scores, gen_scores, gen_scores_pseudo = model_output_all

                    loss_slot_gt, loss_slot_pseudo = get_unreduced_losses(state_scores, op2id, slot_meta, 
                    tokenizer, op_ids, gen_scores, gen_ids, op_ids_pseudo, gen_scores_pseudo, gen_ids_pseudo)
                    
                    s_weight = SW()

                    meta_loss = torch.sum((1.0-s_weight)*loss_slot_gt + s_weight*loss_slot_pseudo) / loss_slot_gt.size(0)
                    # first backward
                    meta_opt.step(meta_loss)
                    
                    # compute on the meta validation set
                    batch_v = next(meta_dataloader)
                    batch_v = [b.to(device) if not isinstance(b, int) and b is not None else b for b in batch_v]
                    input_ids_v, input_mask_v, segment_ids_v, state_position_ids_v, \
                    op_ids_v, gen_ids_v, max_value_v, max_update_v = batch_v
                    # second forward
                    with torch.backends.cudnn.flags(enabled=False):
                        meta_m.eval() # disable dropout, cudnn RNN backward can only be called in training mode
                        state_scores_v, gen_scores_v = meta_m(input_ids=input_ids_v,
                                                              token_type_ids=segment_ids_v,
                                                              state_positions=state_position_ids_v,
                                                              attention_mask=input_mask_v,
                                                              max_value=max_value_v,
                                                              op_ids=op_ids_v,
                                                              max_update=max_update_v,
                                                              teacher=None,
                                                              use_pseudo=False)
                    loss_s = loss_fnc2(state_scores_v.view(-1, len(op2id)), op_ids_v.view(-1))
                    if max_update_v > 0:
                        loss_g = masked_cross_entropy_for_value(gen_scores_v.contiguous(),
                                                                gen_ids_v.contiguous(),
                                                                tokenizer.vocab['[PAD]'])
                    else:
                        loss_g = 0
                    loss_v = loss_s + loss_g
                    # backward over backward
                    wnet_optimizer.zero_grad()
                    loss_v.backward()
                    wnet_optimizer.step()
                    wnet_scheduler.step()
                    meta_batch_loss.append(loss_v.item())
                
                # Now we have the updated weight net    
                # forward (base model)
                model_output_all = base_model(input_ids=input_ids,
                                              token_type_ids=segment_ids,
                                              state_positions=state_position_ids,
                                              attention_mask=input_mask,
                                              max_value=max_value,
                                              op_ids=op_ids,
                                              max_update=max_update,
                                              teacher=teacher,
                                              use_pseudo=True,
                                              max_value_pseudo=max_value_pseudo,
                                              op_ids_pseudo=op_ids_pseudo,
                                              max_update_pseudo=max_update_pseudo,
                                              teacher_pseudo=teacher_pseudo)
                
                state_scores, gen_scores, gen_scores_pseudo = model_output_all

                loss_slot_gt, loss_slot_pseudo = get_unreduced_losses(state_scores, op2id, slot_meta, 
                tokenizer, op_ids, gen_scores, gen_ids, op_ids_pseudo, gen_scores_pseudo, gen_ids_pseudo)
                with torch.no_grad():
                    s_weight = SW()

                loss = torch.sum((1.0-s_weight)*loss_slot_gt + s_weight*loss_slot_pseudo) / loss_slot_gt.size(0)
                # backward (base model)
                base_optimizer.zero_grad()
                loss.backward()
                base_optimizer.step()
                base_scheduler.step()

                batch_loss.append(loss.item())

                if step % 100 == 0:
                    logger.info("[%d/%d] [%d/%d] mean_loss: %.6f mean_meta_loss: %.6f" % \
                               (epoch+1, args.n_epochs, step, len(train_dataloader), 
                                np.mean(batch_loss), np.mean(meta_batch_loss)))
                    batch_loss, meta_batch_loss = [], []
                    logger.info(f'Slot weights: {s_weight.cpu().numpy()}')

            if (epoch+1) % args.eval_epoch == 0:
                eval_res = model_evaluation(base_model, dev_data_raw, tokenizer, slot_meta, epoch+1, device, args.op_code)
                eval_res_test = model_evaluation(base_model, test_data_raw, tokenizer, slot_meta, epoch+1, device, args.op_code)
                logger.info(eval_res_test)
                logger.info(eval_res)
                if eval_res['joint_acc'] > best_score['joint_acc']:
                    best_score = eval_res
                    model_to_save = base_model.module if hasattr(base_model, 'module') else base_model
                    save_path = os.path.join(args.save_dir, 'model_best.bin')
                    save_path_w = os.path.join(args.save_dir, 'wnet.bin')
                    torch.save(model_to_save.state_dict(), save_path)
                    torch.save(SW.state_dict(), save_path_w)
#                 print("Epoch: %d, Test acc: %.6f, Val acc: %.6f, Best Score : %.6f"% (epoch+1, eval_res_test['joint_acc'], eval_res['joint_acc'], best_score['joint_acc']))
                print("\n")
                print(best_score)
                logger.info("Epoch: %d Test acc: %.6f Val acc: %.6f Best Score : %.6f"% (epoch+1, eval_res_test['joint_acc'], eval_res['joint_acc'], best_score['joint_acc']))
                logger.info(" epoch end ")
            

    print("Test using best model...")
    best_epoch = best_score['epoch']
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
    parser.add_argument("--data_root", default='data/mwz2.4', type=str)
    parser.add_argument("--train_data", default='train_dials_v2.json', type=str)
    parser.add_argument("--dev_data", default='dev_dials_v2.json', type=str)
    parser.add_argument("--test_data", default='test_dials_v2.json', type=str)
    parser.add_argument("--ontology_data", default='ontology-modified.json', type=str)
    parser.add_argument("--vocab_path", default='assets/vocab.txt', type=str)
    parser.add_argument("--bert_config_path", default='assets/bert_config_base_uncased.json', type=str)
    parser.add_argument("--save_dir", default='output-meta24-S1/exp', type=str)

    parser.add_argument("--random_seed", default=42, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--meta_batch_size", default=8, type=int)
    parser.add_argument("--enc_warmup", default=0.1, type=float)
    parser.add_argument("--dec_warmup", default=0.1, type=float)
    parser.add_argument("--wnet_warmup", default=0.1, type=float)
    parser.add_argument("--init_weight", default=0.5, type=float)
    parser.add_argument("--enc_lr", default=4e-5, type=float)
    parser.add_argument("--dec_lr", default=1e-4, type=float)
    parser.add_argument("--wnet_lr", default=4e-5, type=float)
    parser.add_argument("--n_epochs", default=30, type=int) #####
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
