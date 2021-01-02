from __future__ import absolute_import, division, print_function

import logging
import os
import json
import random
import glob
import torch
import tqdm
import torch.utils.data
from multimodalKB.utils.config import *
from multimodalKB.utils.utils_temp import entityList, get_type_dict, get_img_fea, load_img_fea
from multimodalKB.utils.utils_general import *
import ast
from transformers import BertTokenizer


logger = logging.getLogger(__name__)


def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    def index_words(self, story, trg=False):
        if trg:
            for word in story.split(' '):
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


class Seq2seqDatasetForBert(torch.utils.data.Dataset):
    def __init__(
            self, features, max_source_len, max_target_len,
            vocab_size, cls_id, sep_id, pad_id, mask_id,
            random_prob, keep_prob, offset, num_training_instances, data_info, lang,
            span_len=1, span_prob=1.0):
        self.features = features
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.offset = offset
        if offset > 0:
            logger.info("  ****  Set offset %d in Seq2seqDatasetForBert ****  ", offset)
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.pad_id = pad_id
        self.random_prob = random_prob
        self.keep_prob = keep_prob
        self.mask_id = mask_id
        self.vocab_size = vocab_size
        self.num_training_instances = num_training_instances
        self.span_len = span_len
        self.span_prob = span_prob

        # Combine multimodal dataset init properties
        self.data_info = {}
        for key in data_info.keys():
            self.data_info[key] = data_info[key]
        self.num_total_seqs = len(data_info['context_arr'])
        self.src_word2id = lang.word2index
        self.trg_word2id = lang.word2index
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return int(self.num_training_instances)

    def __trunk(self, ids, max_len):
        if len(ids) > max_len - 1:
            ids = ids[:max_len - 1]
        ids = ids + [self.sep_id]
        return ids

    def __pad(self, ids, max_len):
        if len(ids) < max_len:
            return ids + [self.pad_id] * (max_len - len(ids))
        else:
            assert len(ids) == max_len
            return ids

    def __getitem__(self, idx):
        idx = (self.offset + idx) % len(self.features)

        # Combine multimodal dataset __getitem__ instructions
        context_arr = self.data_info['context_arr'][idx]
        context_arr = self.preprocess(context_arr, self.src_word2id, trg=False)
        response = self.data_info['response'][idx]
        response = self.preprocess(response, self.trg_word2id)
        ptr_index = torch.Tensor(self.data_info['ptr_index'][idx])
        conv_arr = self.data_info['conv_arr'][idx]
        conv_arr = self.preprocess_conv_arr(conv_arr)
        # conv_arr = self.preprocess(conv_arr, self.src_word2id, trg=False)
        kb_arr = self.data_info['kb_arr'][idx]
        kb_arr = self.preprocess(kb_arr, self.src_word2id, trg=False)
        img_arr = torch.Tensor(self.data_info['img_arr'][idx])
        calibration_vocab = torch.Tensor(self.data_info['calibration_vocab'][idx])
        turns = torch.Tensor(self.data_info['turns'][idx])
        # cls_ids = torch.Tensor(self.data_info['cls_ids'][idx])

        # processed information
        data_info = {}
        for k in self.data_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = self.data_info[k][idx]

        # additional plain information
        data_info['context_arr_plain'] = self.data_info['context_arr'][idx]
        data_info['response_plain'] = self.data_info['response'][idx]
        data_info['kb_arr_plain'] = self.data_info['kb_arr'][idx]

        # idx = (self.offset + idx) % len(self.features)
        feature = self.features[idx]
        source_ids = self.__trunk([self.cls_id] + feature["source_ids"], self.max_source_len)
        target_ids = self.__trunk(feature["target_ids"], self.max_target_len)
        pseudo_ids = []
        for tk_id in target_ids:
            p = random.random()
            if p < self.keep_prob:
                pseudo_ids.append(tk_id)
            elif p < self.keep_prob + self.random_prob:
                pseudo_ids.append(random.randint(0, self.vocab_size - 1))
            else:
                pseudo_ids.append(self.mask_id)

        num_source_tokens = len(source_ids)
        num_target_tokens = len(target_ids)

        source_ids = self.__pad(source_ids, self.max_source_len)
        target_ids = self.__pad(target_ids, self.max_target_len)
        pseudo_ids = self.__pad(pseudo_ids, self.max_target_len)

        if self.span_len > 1:
            span_ids = []
            span_id = 1
            while len(span_ids) < num_target_tokens:
                p = random.random()
                if p < self.span_prob:
                    span_len = random.randint(2, self.span_len)
                    span_len = min(span_len, num_target_tokens - len(span_ids))
                else:
                    span_len = 1
                span_ids.extend([span_id] * span_len)
                span_id += 1
            span_ids = self.__pad(span_ids, self.max_target_len)
            data_info['unilm_info'] = (source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens, span_ids)
            # return source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens, span_ids
        else:
            data_info['unilm_info'] = (source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens)
            # return source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens
        return data_info

    def preprocess_conv_arr(self, sequence):
        story = []
        for i, word in enumerate(sequence):
            temp = self.tokenizer._convert_token_to_id(word)
            story.append(temp)
        story = torch.Tensor(story)
        return story

    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
        try:
            story = torch.Tensor(story)
        except:
            print(story)
        return story


def batch_list_to_batch_tensors(data):
    # Combine multimodal dataset collate_fn
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

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['conv_arr']), reverse=True)
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    context_arr, context_arr_lengths = merge(item_info['context_arr'], True)
    response, response_lengths = merge(item_info['response'], False)
    ptr_index, _ = merge(item_info['ptr_index'], False)
    conv_arr, conv_arr_lengths = merge(item_info['conv_arr'], False)
    # conv_arr, conv_arr_lengths = merge(item_info['conv_arr'], True)
    kb_arr, kb_arr_lengths = merge(item_info['kb_arr'], True)
    img_arr, _ = merge_image(item_info['img_arr'])
    calibration_vocab, _ = merge_index(item_info['calibration_vocab'])

    # convert to contiguous and cuda
    context_arr = _cuda(context_arr.contiguous())
    response = _cuda(response.contiguous())
    ptr_index = _cuda(ptr_index.contiguous())
    conv_arr = _cuda(conv_arr.transpose(0, 1).contiguous())
    img_arr = _cuda(img_arr.contiguous())
    calibration_vocab = _cuda(calibration_vocab.contiguous())
    if (len(list(kb_arr.size())) > 1): kb_arr = _cuda(kb_arr.transpose(0, 1).contiguous())

    # processed information
    data_info = {}
    for k in item_info.keys():
        try:
            data_info[k] = locals()[k]
        except:
            data_info[k] = item_info[k]

    # additional plain information
    data_info['context_arr_lengths'] = context_arr_lengths
    data_info['response_lengths'] = response_lengths
    data_info['conv_arr_lengths'] = conv_arr_lengths
    data_info['kb_arr_lengths'] = kb_arr_lengths

    batch = item_info['unilm_info']
    batch_tensors = []
    for x in zip(*batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(_cuda(torch.tensor(x, dtype=torch.long)))
    data_info['unilm_info'] = batch_tensors
    return data_info


def get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    os.path.basename(output_dir)
    both_set = set([int(os.path.basename(fn).split('.')[1]) for fn in fn_model_list]
                   ) & set([int(os.path.basename(fn).split('.')[1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None


def read_langs(file_name, global_entity, type_dict, img_path, max_line=None):
    print("Reading lines from {}".format(file_name))
    data, context_arr, conv_arr, kb_arr, img_arr, cls_ids = [], [], [], [], [], []
    max_res_len, sample_counter, turn, kb_rec_cnt = 0, 0, 0, 0
    src_tokens = ''
    image_feas = load_img_fea(img_path)
    with open(file_name) as fin:
        cnt_lin = 1
        for line in fin:
            line = line.strip()
            if line:
                nid, line = line.split(' ', 1)
                if '\t' in line:
                    if kb_rec_cnt < 30:
                        continue
                    try:
                        u, r, gold_ent = line.split('\t')
                    except:
                        print(line)
                        continue
                    gen_u = generate_memory(u, "$u", str(turn), image_feas)
                    if len(gen_u[0]) > 4:
                        print(gen_u)
                        print(u, r)
                    context_arr += gen_u
                    # conv_arr += gen_u
                    u_token = u.split(' ')
                    u_token.insert(0, "[CLS]")
                    u_token.append("[SEP]")
                    len_conv_arr = len(conv_arr)
                    cls_ids.append(len_conv_arr)
                    conv_arr += u_token
                    if src_tokens == '':
                        src_tokens = src_tokens + u
                    else:
                        src_tokens = src_tokens + ' ' + u
                    ptr_index, ent_words = [], []

                    ent_words = ast.literal_eval(gold_ent)

                    # Get local pointer position for each word in system response
                    for key in r.split():
                        if key in global_entity and key not in ent_words:
                            ent_words.append(key)
                        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in global_entity)]
                        index = max(index) if (index) else len(context_arr)
                        ptr_index.append(index)

                    data_detail = {
                        'context_arr':list(context_arr+[['$$$$']*MEM_TOKEN_SIZE]),  # dialogue history + kb
                        'response':r,  # response
                        'ptr_index':ptr_index+[len(context_arr)],
                        'ent_index':ent_words,
                        'conv_arr':list(conv_arr),  # dialogue history ---> bert encode
                        'kb_arr':list(kb_arr),  # kb ---> memory encode
                        'img_arr':list(img_arr),  # image ---> attention encode
                        'id':int(sample_counter),
                        'ID':int(cnt_lin),
                        'domain':"",
                        'turns':[turn],
                        'src_tokens': src_tokens,
                        'cls_ids': list(cls_ids)}
                    data.append(data_detail)

                    gen_r = generate_memory(r, "$s", str(turn), image_feas)
                    if len(gen_r[0]) > 4:
                        print(gen_r)
                        print(u, r)
                    context_arr += gen_r
                    # conv_arr += gen_r
                    r_token = r.split(' ')
                    r_token.append("[SEP]")
                    conv_arr = conv_arr[:-1]
                    conv_arr += r_token
                    src_tokens = src_tokens + ' ' + r
                    if max_res_len < len(r.split()):
                        max_res_len = len(r.split())
                    sample_counter += 1
                    turn += 1
                else:
                    kb_rec_cnt += 1
                    r = line
                    if "image" not in r:
                        kb_info = generate_memory(r, "", str(nid), image_feas)
                        if len(kb_info[0]) > 4:
                            print(kb_info)
                            print(r)
                        context_arr = kb_info + context_arr
                        kb_arr += kb_info
                    else:
                        image_info = generate_memory(r, "", str(nid), image_feas)
                        img_arr += image_info
            else:
                cnt_lin += 1
                turn, kb_rec_cnt = 0, 0
                context_arr, conv_arr, kb_arr, img_arr, cls_ids = [], [], [], [], []
                src_tokens = ''
                if(max_line and cnt_lin>max_line):
                    break
    return data, max_res_len


def generate_memory(sent, speaker, time, image_feas):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker == "$u" or speaker == "$s":
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn'+str(time), 'word'+str(idx)] + ["PAD"]*(MEM_TOKEN_SIZE-4)
            sent_new.append(temp)
    else:
        try:
            if sent_token[1] == "R_rating":
                sent_token = sent_token + ["PAD"]*(MEM_TOKEN_SIZE-len(sent_token))
            # add logic to cope with image info
            elif sent_token[1].startswith("image"):
                # add image feature retrieve logic
                image_key = sent_token[-1]
                image_fea = get_img_fea(image_key, image_feas)
                sent_token = image_fea
            else:
                sent_token = sent_token[::-1] + ["PAD"]*(MEM_TOKEN_SIZE-len(sent_token))
            sent_new.append(sent_token)
        except:
            print(sent)
            print(sent_token)
            exit()
    return sent_new


def get_seq(pairs, lang, type):
    data_info = {}
    for k in pairs[0].keys():
        data_info[k] = []

    # add calibration label info
    data_info['calibration_vocab'] = []

    for pair in pairs:
        for k in pair.keys():
            data_info[k].append(pair[k])
        if (type):
            lang.index_words(pair['context_arr'])
            lang.index_words(pair['response'], trg=True)

    # parse calibration label info
    for pair in pairs:
        zeros = torch.zeros(lang.n_words)
        for k in pair.keys():
            if k == 'response':
                tokens = pair[k].split(' ')
                for token in tokens:
                    if token in lang.word2index:
                        id = lang.word2index[token]
                        zeros[id] = 1
                    else:
                        zeros[0] = 1
        data_info['calibration_vocab'].append(zeros)
    return data_info


def load_and_cache_examples(
        example_file, tokenizer, local_rank, cached_features_file, shuffle=True):
    # Combine multimodal dataset loader
    data_path_babi = '/home/shiquan/Projects/unilm_joint_learning/unilm/s2s-ft/multimodalKB/data/0_synthetic/dialog-babi'
    # data_path_babi = '/Users/shiquan/PycharmProjects/MultiModalKB/s2s-ft/multimodalKB/data/0_synthetic/dialog-babi'
    data_path = '/home/shiquan/Projects/unilm_joint_learning/unilm/s2s-ft/multimodalKB/data/2_20K_multimodal_dataset/train_split.txt'
    # data_path = '/home/shiquan/Projects/unilm_joint_learning/unilm/s2s-ft/multimodalKB/data/2_20K_multimodal_dataset/test_split.txt'
    # data_path = '/Users/shiquan/PycharmProjects/MultiModalKB/s2s-ft/multimodalKB/data/2_20K_multimodal_dataset/for_debug.txt'
    # img_path = '/Users/shiquan/PycharmProjects/MultiModalKB/s2s-ft/multimodalKB/images/restaurant'
    img_path = '/home/shiquan/Projects/unilm_joint_learning/unilm/s2s-ft/multimodalKB/images/restaurant'
    file_train = '{}'.format(data_path)
    # file_train = '{}-trn-multimodal-phase1-version-lowercase.txt'.format(data_path)
    # file_train = '{}-task{}trn.txt'.format(data_path, task)
    # file_dev = '{}-dev-multimodal-phase1-version-lowercase.txt'.format(data_path)
    # file_dev = '{}-task{}dev.txt'.format(data_path, task)
    # file_test = '{}-tst-multimodal-phase1-version-lowercase.txt'.format(data_path)
    # file_test = '{}-task{}tst.txt'.format(data_path, task)
    kb_path = data_path_babi + '-kb-all.txt'
    file_test_OOV = '{}-tst-multimodal-OOV.txt'.format(data_path)  # no-OOV dataset available!
    type_dict = get_type_dict(kb_path, dstc2=False)
    global_ent = entityList(kb_path, 4)

    pair_train, train_max_len = read_langs(file_train, global_ent, type_dict, img_path)
    # pair_dev, dev_max_len = read_langs(file_dev, global_ent, type_dict, img_path)
    # pair_test, test_max_len = read_langs(file_test, global_ent, type_dict, img_path)
    # max_resp_len = max(train_max_len, dev_max_len, test_max_len)
    max_resp_len = train_max_len

    lang = Lang()

    train_data_info = get_seq(pair_train, lang, True)

    print("Read %s sentence pairs train" % len(pair_train))
    # print("Read %s sentence pairs dev" % len(pair_dev))
    # print("Read %s sentence pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if cached_features_file is not None and os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", example_file)

        # examples = []
        # with open(example_file, mode="r", encoding="utf-8") as reader:
        #     for line in reader:
        #         examples.append(json.loads(line))
        features = []

        for example in pair_train:
            t1 = example["src_tokens"]
            t2 = example["response"]
            source_tokens = tokenizer.tokenize(example["src_tokens"])
            target_tokens = tokenizer.tokenize(example["response"])
            features.append({
                    "source_ids": tokenizer.convert_tokens_to_ids(source_tokens),
                    "target_ids": tokenizer.convert_tokens_to_ids(target_tokens),
                })

        if shuffle:
            random.shuffle(features)

        if local_rank in [-1, 0] and cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank == 0:
        torch.distributed.barrier()

    return features, train_data_info, lang
