"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import logging
import argparse
import math
from tqdm import tqdm, trange
import numpy as np
import torch
import random
import pickle

from multimodalKB.models.MultimodalKB_Local import MultimodalKBLocal
from transformers import BertTokenizer, RobertaTokenizer
from s2s_ft.modeling_decoding import BertForSeq2SeqDecoder, BertConfig
from transformers.tokenization_bert import whitespace_tokenize
import s2s_ft.s2s_loader as seq2seq_loader
from s2s_ft.utils_combined import load_and_cache_examples
from transformers import \
    BertTokenizer, RobertaTokenizer
from s2s_ft.tokenization_unilm import UnilmTokenizer
from s2s_ft.tokenization_minilm import MinilmTokenizer
from multimodalKB.utils.config import *

TOKENIZER_CLASSES = {
    'bert': BertTokenizer,
    'minilm': MinilmTokenizer,
    'roberta': RobertaTokenizer,
    'unilm': UnilmTokenizer,
}

class WhitespaceTokenizer(object):
    def tokenize(self, text):
        return whitespace_tokenize(text)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(TOKENIZER_CLASSES.keys()))
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="Path to the model checkpoint.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Path to config.json for the model.")

    # tokenizer_name
    parser.add_argument("--tokenizer_name", default=None, type=str, required=True, 
                        help="tokenizer name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    # decoding parameters
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument('--subset', type=int, default=0,
                        help="Decode a subset of the input dataset.")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--split", type=str, default="",
                        help="Data split (train/val/test).")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=1, type=int)
    parser.add_argument('--need_score_traces', action='store_true')
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--mode', default="s2s",
                        choices=["s2s", "l2r", "both"])
    parser.add_argument('--max_tgt_length', type=int, default=128,
                        help="maximum length of target sequence")
    parser.add_argument('--s2s_special_token', action='store_true',
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', action='store_true',
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', action='store_true',
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('-ds', '--dataset', help='dataset, babi or kvr', required=False)
    parser.add_argument('-t', '--task', help='Task Number', required=False, default="")
    parser.add_argument('-dec', '--decoder', help='decoder model', required=False)
    parser.add_argument('-hdd', '--hidden', help='Hidden size', required=False, default=128)
    parser.add_argument('-bsz', '--batch', help='Batch_size', required=False, default=8)
    parser.add_argument('-lr', '--learn', help='Learning Rate', required=False, default=0.001)
    parser.add_argument('-dr', '--drop', help='Drop Out', required=False, default=0.2)
    parser.add_argument('-um', '--unk_mask', help='mask out input token to UNK', type=int, required=False, default=1)
    parser.add_argument('-l', '--layer', help='Layer Number', required=False, default=1)
    parser.add_argument('-lm', '--limit', help='Word Limit', required=False, default=-10000)
    parser.add_argument('-path', '--path', help='path of the file to load', required=False)
    parser.add_argument('-clip', '--clip', help='gradient clipping', required=False, default=10)
    parser.add_argument('-tfr', '--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False,
                        default=0.5)

    parser.add_argument('-sample', '--sample', help='Number of Samples', required=False, default=None)
    parser.add_argument('-evalp', '--evalp', help='evaluation period', required=False, default=1)
    parser.add_argument('-an', '--addName', help='An add name for the save folder', required=False, default='')
    parser.add_argument('-gs', '--genSample', help='Generate Sample', required=False, default=0)
    parser.add_argument('-es', '--earlyStop', help='Early Stop Criteria, BLEU or ENTF1', required=False, default='BLEU')
    parser.add_argument('-abg', '--ablationG', help='ablation global memory pointer', type=int, required=False,
                        default=0)
    parser.add_argument('-abh', '--ablationH', help='ablation context embedding', type=int, required=False, default=0)
    parser.add_argument('-rec', '--record', help='use record function during inference', type=int, required=False,
                        default=0)
    parser.add_argument('-inchannels', '--inchannels', help='input channels', type=int, required=False, default=1024)
    parser.add_argument('-outchannels', '--outchannels', help='output channels', type=int, required=False, default=256)
    parser.add_argument('-convkernelsize', '--convkernelsize', help='convolutional kernel size', type=int,
                        required=False, default=3)
    parser.add_argument('-poolkernelsize', '--poolkernelsize', help='pool kernel size', type=int, required=False,
                        default=3)
    # parser.add_argument('-beam','--beam_search', help='use beam_search during inference, default is greedy search', type=int, required=False, default=0)
    # parser.add_argument('-viz','--vizualization', help='vizualization', type=int, required=False, default=0)

    args = parser.parse_args()

    if args.need_score_traces and args.beam_size <= 1:
        raise ValueError(
            "Score trace is only available for beam search with beam size > 1.")
    if args.max_tgt_length >= args.max_seq_length - 2:
        raise ValueError("Maximum tgt length exceeds max seq length - 2.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # device = "cpu"
    n_gpu = torch.cuda.device_count()

    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
    else:
        random_seed = random.randint(0, 10000)
        logger.info("Set random seed as: {}".format(random_seed))
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
    
    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.tokenizer_name, do_lower_case=args.do_lower_case, 
        cache_dir=args.cache_dir if args.cache_dir else None)

    if args.model_type == "roberta":
        vocab = tokenizer.encoder
    else:
        vocab = tokenizer.vocab

    # tokenizer.max_len = args.max_seq_length

    config_file = args.config_path if args.config_path else os.path.join(args.model_path, "config.json")
    logger.info("Read decoding config from: %s" % config_file)
    config = BertConfig.from_json_file(config_file)

    bi_uni_pipeline = []
    bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(
        list(vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
        max_tgt_length=args.max_tgt_length, pos_shift=args.pos_shift,
        source_type_id=config.source_type_id, target_type_id=config.target_type_id, 
        cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token, pad_token=tokenizer.pad_token))

    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        [tokenizer.mask_token, tokenizer.sep_token, tokenizer.sep_token])
    forbid_ignore_set = None
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))
    print(args.model_path)
    found_checkpoint_flag = False
    for model_recover_path in [args.model_path.strip()]:
        logger.info("***** Recover model: %s *****", model_recover_path)
        found_checkpoint_flag = True

        to_pred, test_data_info, lang = load_and_cache_examples(
            args.input_file, tokenizer, local_rank=-1,
            cached_features_file=None, shuffle=False)

        multimodalKB_model = MultimodalKBLocal(
            int(args.hidden),
            lang,
            40,
            args.path,
            "",
            lr=float(args.learn),
            n_layers=int(args.layer),
            dropout=float(args.drop),
            input_channels=int(args.inchannels),
            output_channels=int(args.outchannels),
            conv_kernel_size=int(args.convkernelsize),
            pool_kernel_size=int(args.poolkernelsize),
            config=config)

        model = BertForSeq2SeqDecoder.from_pretrained(
            model_recover_path, config=config, mask_word_id=mask_word_id, search_beam_size=args.beam_size,
            length_penalty=args.length_penalty, eos_id=eos_word_ids, sos_id=sos_word_id,
            forbid_duplicate_ngrams=args.forbid_duplicate_ngrams, forbid_ignore_set=forbid_ignore_set,
            ngram_size=args.ngram_size, min_len=args.min_len, mode=args.mode,
            max_position_embeddings=args.max_seq_length, pos_shift=args.pos_shift, 
        )

        if args.fp16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        torch.cuda.empty_cache()
        model.eval()
        next_i = 0
        max_src_length = args.max_seq_length - 2 - args.max_tgt_length

        input_lines = []
        for line in to_pred:
            # convert to ids and truncate by max_src_length
            input_lines.append(line["source_ids"][:max_src_length])
        if args.subset > 0:
            logger.info("Decoding subset: %d", args.subset)
            input_lines = input_lines[:args.subset]

        input_lines = sorted(list(enumerate(input_lines)),
                             key=lambda x: -len(x[1]))
        # sort test_data_info by the same orders.
        indexes = [ele[0] for ele in input_lines]
        conv_arr_re_ordered = [test_data_info['conv_arr'][ele] for ele in indexes]
        kb_arr_re_ordered = [test_data_info['kb_arr'][ele] for ele in indexes]
        img_arr_re_ordered = [test_data_info['img_arr'][ele] for ele in indexes]
        turns_re_ordered = [test_data_info['turns'][ele] for ele in indexes]
        cls_ids_re_ordered = [test_data_info['cls_ids'][ele] for ele in indexes]
        test_data_info['conv_arr'] = conv_arr_re_ordered
        test_data_info['kb_arr'] = kb_arr_re_ordered
        test_data_info['img_arr'] = img_arr_re_ordered
        test_data_info['turns'] = turns_re_ordered
        test_data_info['cls_ids'] = cls_ids_re_ordered

        output_lines = [""] * len(input_lines)
        score_trace_list = [None] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / args.batch_size)

        with tqdm(total=total_batch) as pbar:
            batch_count = 0
            first_batch = True
            while next_i < len(input_lines):
                # sample a batch of instances from instance pool
                # - sample source_tokens
                _chunk = input_lines[next_i:next_i + args.batch_size]
                # - sample data_info for multimodalKB model inputs and convert to ids
                data_info = {}
                conv_arr = test_data_info['conv_arr'][next_i:next_i + args.batch_size]
                conv_arr = preprocess_conv_arr(conv_arr)
                kb_arr = test_data_info['kb_arr'][next_i:next_i + args.batch_size]
                kb_arr = preprocess(kb_arr, lang.word2index, trg=False)
                img_arr = torch.Tensor(test_data_info['img_arr'][next_i:next_i + args.batch_size])
                turns = torch.Tensor(test_data_info['turns'][next_i:next_i + args.batch_size])
                cls_ids = test_data_info['cls_ids'][next_i:next_i + args.batch_size]
                # cls_ids = torch.Tensor(test_data_info['cls_ids'][next_i:next_i + args.batch_size])
                # - pad sampled data_info
                conv_arr, conv_arr_lengths = merge(conv_arr, False)
                kb_arr, kb_arr_lengths = merge(kb_arr, True)
                img_arr, _ = merge_image(img_arr)
                conv_arr = _cuda(conv_arr.transpose(0, 1).contiguous())
                img_arr = _cuda(img_arr.contiguous())
                if (len(list(kb_arr.size())) > 1): kb_arr = _cuda(kb_arr.transpose(0, 1).contiguous())

                data_info["conv_arr"] = conv_arr
                data_info["kb_arr"] = kb_arr
                data_info["img_arr"] = img_arr
                data_info["turns"] = turns
                data_info["cls_ids"] = cls_ids
                data_info['conv_arr_lengths'] = conv_arr_lengths

                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]
                next_i += args.batch_size
                batch_count += 1

                with torch.no_grad():
                    multimodalKB_model.local_semantics_extractor.train(False)
                    local_semantic_vectors, lengths = multimodalKB_model.train_batch(data_info, int(args.clip), reset=((batch_count-1)==0))
                    multimodalKB_model.local_semantics_extractor.train(True)

                    instances = []
                    # max_a_len = max([len(x) for x in buf])
                    max_a_len = int(max(lengths))
                    # make pseudo input_ids according to lengths information
                    pseudo_buf = [[0] * int(len) for len in lengths]
                    # for instance in [(x, max_a_len) for x in buf]:
                    for instance in [(x, max_a_len) for x in pseudo_buf]:
                        for proc in bi_uni_pipeline:
                            instances.append(proc(instance))
                    batch = seq2seq_loader.batch_list_to_batch_tensors(
                        instances)
                    batch = [
                        t.to(device) if t is not None else None for t in batch]

                    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                    traces = model(input_ids, token_type_ids,
                                   position_ids, input_mask, local_semantic_vectors, lengths, task_idx=task_idx, mask_qkv=mask_qkv)
                    if args.beam_size > 1:
                        traces = {k: v.tolist() for k, v in traces.items()}
                        output_ids = traces['pred_seq']
                    else:
                        output_ids = traces.tolist()
                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []
                        for t in output_buf:
                            if t in (tokenizer.sep_token, tokenizer.pad_token):
                                break
                            output_tokens.append(t)
                        if args.model_type == "roberta":
                            output_sequence = tokenizer.convert_tokens_to_string(output_tokens)
                        else:
                            output_sequence = ' '.join(detokenize(output_tokens))
                        if '\n' in output_sequence:
                            output_sequence = " [X_SEP] ".join(output_sequence.split('\n'))
                        output_lines[buf_id[i]] = output_sequence
                        if first_batch or batch_count % 50 == 0:
                            logger.info("{} = {}".format(buf_id[i], output_sequence))
                        if args.need_score_traces:
                            score_trace_list[buf_id[i]] = {
                                'scores': traces['scores'][i], 'wids': traces['wids'][i], 'ptrs': traces['ptrs'][i]}
                pbar.update(1)
                first_batch = False
        if args.output_file:
            fn_out = args.output_file
        else:
            fn_out = model_recover_path+'.'+args.split
        with open(fn_out, "w", encoding="utf-8") as fout:
            for l in output_lines:
                fout.write(l)
                fout.write("\n")

        if args.need_score_traces:
            with open(fn_out + ".trace.pickle", "wb") as fout_trace:
                pickle.dump(
                    {"version": 0.0, "num_samples": len(input_lines)}, fout_trace)
                for x in score_trace_list:
                    pickle.dump(x, fout_trace)

    if not found_checkpoint_flag:
        logger.info("Not found the model checkpoint file!")


def preprocess_conv_arr(sequence):
    ret = []
    for seq in sequence:
        story = []
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        for i, word in enumerate(seq):
            temp = tokenizer._convert_token_to_id(word)
            story.append(temp)
        story = torch.Tensor(story)
        ret.append(story)
    return ret


def preprocess(sequence, word2id, trg=True):
    """Converts words to ids."""
    ret = []
    for seq in sequence:
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in seq.split(' ')] + [EOS_token]
        else:
            story = []
            for i, word_triple in enumerate(seq):
                story.append([])
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
        try:
            story = torch.Tensor(story)
        except:
            print(story)
        ret.append(story)
    return ret


def merge(sequences, story_dim):
    lengths = [len(seq) for seq in sequences]
    max_len = 1 if max(lengths) == 0 else max(lengths)
    if (story_dim):
        padded_seqs = torch.ones(len(sequences), max_len, MEM_TOKEN_SIZE).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if len(seq) != 0:
                padded_seqs[i, :end, :] = seq[:end]
    else:
        padded_seqs = torch.ones(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
    return padded_seqs, lengths

def merge_index(sequences):
    lengths = [len(seq) for seq in sequences]
    padded_seqs = torch.zeros(len(sequences), max(lengths)).float()
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = seq[:end]
    return padded_seqs, lengths

def merge_image(sequences):
    lengths = [len(seq) for seq in sequences]
    padded_seqs = torch.zeros(len(sequences), max(lengths), DEFAULT_INPUT_CHANNELS, DEFAULT_INPUT_KERNEL_SIZE,
                              DEFAULT_INPUT_KERNEL_SIZE).float()
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end, :, :, :] = seq[:end]
    return padded_seqs, lengths


def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x


if __name__ == "__main__":
    main()
