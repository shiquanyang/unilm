from utils.config import *
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
import pdb


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

class Dataset(Dataset):
    def __init__(self, data_info, src_word2id, trg_word2id):
        self.data_info = {}
        for key in data_info.keys():
            self.data_info[key] = data_info[key]
        self.num_total_seqs = len(data_info['context_arr'])
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __getitem__(self, index):
        context_arr = self.data_info['context_arr'][index]
        context_arr = self.preprocess(context_arr, self.src_word2id, trg=False)
        response = self.data_info['response'][index]
        response = self.preprocess(response, self.trg_word2id)
        ptr_index = torch.Tensor(self.data_info['ptr_index'][index])
        conv_arr = self.data_info['conv_arr'][index]
        # conv_arr = self.preprocess_conv_arr(conv_arr)
        conv_arr = self.preprocess(conv_arr, self.src_word2id, trg=False)
        kb_arr = self.data_info['kb_arr'][index]
        kb_arr = self.preprocess(kb_arr, self.src_word2id, trg=False)
        img_arr = torch.Tensor(self.data_info['img_arr'][index])
        calibration_vocab = torch.Tensor(self.data_info['calibration_vocab'][index])
        turns = torch.Tensor(self.data_info['turns'][index])

        # processed information
        data_info = {}
        for k in self.data_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = self.data_info[k][index]

        # additional plain information
        data_info['context_arr_plain'] = self.data_info['context_arr'][index]
        data_info['response_plain'] = self.data_info['response'][index]
        data_info['kb_arr_plain'] = self.data_info['kb_arr'][index]

        return data_info

    def __len__(self):
        return self.num_total_seqs

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

    def collate_fn(self, data):
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
            padded_seqs = torch.zeros(len(sequences), max(lengths), DEFAULT_INPUT_CHANNELS, DEFAULT_INPUT_KERNEL_SIZE, DEFAULT_INPUT_KERNEL_SIZE).float()
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
        # conv_arr, conv_arr_lengths = merge(item_info['conv_arr'], False)
        conv_arr, conv_arr_lengths = merge(item_info['conv_arr'], True)
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

        return data_info

def get_seq(pairs, lang, batch_size, type):
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

    dataset = Dataset(data_info, lang.word2index, lang.word2index)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=type,
                                              # shuffle = False,
                                              collate_fn=dataset.collate_fn)
    return data_loader
