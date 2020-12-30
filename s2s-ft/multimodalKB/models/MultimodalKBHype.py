import torch
import torch.nn as nn
from utils.config import *
from models.BertEncoder import BertEncoder
from models.GraphMemory import GraphMemory
from models.VisualMemory import VisualMemory
from models.Decoder import Decoder
from models.ContextRNN import ContextRNN
from models.ImageEncoder import ImageEncoder
from torch import optim
from torch.optim import lr_scheduler
import random
from utils.masked_cross_entropy import *
import numpy as np
import json
from utils.measures import wer, moses_multi_bleu
from utils.rsgd import RiemannianSGD
from utils.rsgd_utils import rgrad
from utils.rsgd_utils import expm
from utils.rsgd_utils import logm
from utils.rsgd_utils import ptransp
import pdb


class MultimodalKBHype(nn.Module):
    def __init__(self, hidden_size, lang, max_response_len, path, task, lr, n_layers, dropout, input_channels, output_channels, conv_kernel_size, pool_kernel_size):
        super(MultimodalKBHype, self).__init__()
        self.name = "MultimodalKBHype"
        self.task = task
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.max_resp_len = max_response_len
        self.decoder_hop = n_layers

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th')
                self.graphmemory = torch.load(str(path)+'/graphmemory.th')
                self.visualmemory = torch.load(str(path)+'/visualmemory.th')
                self.decoder = torch.load(str(path)+'/dec.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th',lambda storage, loc: storage)
                self.graphmemory = torch.load(str(path)+'/graphmemory.th',lambda storage, loc: storage)
                self.visualmemory = torch.load(str(path)+'/visualmemory.th',lambda storage, loc: storage)
                self.decoder = torch.load(str(path)+'/dec.th',lambda storage, loc: storage)
        else:
            # self.encoder = BertEncoder(lang.n_words, hidden_size, dropout)
            self.encoder = ContextRNN(lang.n_words, hidden_size, dropout)
            self.graphmemory = GraphMemory(lang.n_words, hidden_size, n_layers, dropout)
            # self.visualmemory = VisualMemory(input_channels, output_channels, conv_kernel_size, pool_kernel_size, hidden_size, n_layers, dropout)
            self.visualmemory = ImageEncoder(input_channels, output_channels, conv_kernel_size, pool_kernel_size, hidden_size, n_layers, dropout)
            self.decoder = Decoder(self.encoder.embedding, lang, hidden_size, dropout)

        # Initialize optimizers and criterion
        # self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        # self.graphmemory_optimizer = optim.Adam(self.graphmemory.parameters(), lr=lr)
        # self.visualmemory_optimizer = optim.Adam(self.visualmemory.parameters(), lr=lr)
        # self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.encoder_optimizer = RiemannianSGD([{
            'params': self.encoder.parameters(),
            'rgrad': rgrad,
            'expm': expm,
            'logm': logm,
            'ptransp': ptransp,
        }], lr=lr)
        self.graphmemory_optimizer = RiemannianSGD([{
            'params': self.graphmemory.parameters(),
            'rgrad': rgrad,
            'expm': expm,
            'logm': logm,
            'ptransp': ptransp,
        }], lr=lr)
        self.visualmemory_optimizer = RiemannianSGD([{
            'params': self.visualmemory.parameters(),
            'rgrad': rgrad,
            'expm': expm,
            'logm': logm,
            'ptransp': ptransp,
        }], lr=lr)
        self.decoder_optimizer = RiemannianSGD([{
            'params': self.decoder.parameters(),
            'rgrad': rgrad,
            'expm': expm,
            'logm': logm,
            'ptransp': ptransp,
        }], lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)
        self.criterion_bce = nn.BCELoss()
        self.reset()

        if USE_CUDA:
            self.encoder.cuda()
            self.graphmemory.cuda()
            self.visualmemory.cuda()
            self.decoder.cuda()

    def print_loss(self):
        print_loss_v = self.loss_v / self.print_every
        self.print_every += 1
        return 'L:{:.2f}'.format(print_loss_v)

    def save_model(self, dec_type):
        # name_data = "KVR/" if self.task=='' else "BABI/"
        name_data = "BABI/"
        layer_info = str(self.n_layers)
        directory = 'save/MultimodalKB-'+args["addName"]+name_data+str(self.task)+'HDD'+str(self.hidden_size)+'BSZ'+str(args['batch'])+'DR'+str(self.dropout)+'L'+layer_info+'lr'+str(self.lr)+'IC'+str(self.input_channels)+'OC'+str(self.output_channels)+'CK'+str(self.conv_kernel_size)+'PK'+str(self.pool_kernel_size)+str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.graphmemory, directory + '/graphmemory.th')
        torch.save(self.visualmemory, directory + '/visualmemory.th')
        torch.save(self.decoder, directory + '/dec.th')

    def reset(self):
        self.print_every, self.loss_v = 1, 0

    def _cuda(self, x):
        if USE_CUDA:
            return torch.Tensor(x).cuda()
        else:
            return torch.Tensor(x)

    def train_batch(self, data, clip, reset=0):
        if reset:
            self.reset()
        self.encoder_optimizer.zero_grad()
        self.graphmemory_optimizer.zero_grad()
        self.visualmemory_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Encode and Decode
        use_teacher_forcing = random.random() < args['teacher_forcing_ratio']
        max_target_length = max(data['response_lengths'])
        all_decoder_outputs_vocab, _, _ = self.encode_and_decode(data, max_target_length, use_teacher_forcing, False)

        # Loss calculation and backpropagation
        loss_v = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(),
            data['response'].contiguous(),
            data['response_lengths'])
        loss_v.backward()

        # Clip gradient norms
        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        gc = torch.nn.utils.clip_grad_norm_(self.graphmemory.parameters(), clip)
        vc = torch.nn.utils.clip_grad_norm_(self.visualmemory.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        # Update parameters with optimizers
        self.encoder_optimizer.step()
        self.graphmemory_optimizer.step()
        self.visualmemory_optimizer.step()
        self.decoder_optimizer.step()
        self.loss_v += loss_v.item()

    def encode_and_decode(self, data, max_target_length, use_teacher_forcing, get_decoded_words):
        # Build unknown mask for memory
        # if args['unk_mask'] and self.decoder.training:
        #     story_size = data['context_arr'].size()
        #     rand_mask = np.ones(story_size)
        #     bi_mask = np.random.binomial([np.ones((story_size[0], story_size[1]))], 1 - self.dropout)[0]
        #     rand_mask[:, :, 0] = rand_mask[:, :, 0] * bi_mask
        #     conv_rand_mask = np.ones(data['conv_arr'].size())
        #     for bi in range(story_size[0]):
        #         start, end = data['kb_arr_lengths'][bi], data['kb_arr_lengths'][bi] + data['conv_arr_lengths'][bi]
        #         conv_rand_mask[:end - start, bi, :] = rand_mask[bi, start:end, :]
        #     rand_mask = self._cuda(rand_mask)
        #     conv_rand_mask = self._cuda(conv_rand_mask)
        #     conv_story = data['conv_arr'] * conv_rand_mask.long()
        #     story = data['context_arr'] * rand_mask.long()
        # else:
        #     story, conv_story = data['context_arr'], data['conv_arr']

        conv_story = data['conv_arr']
        # Encode dialog history and KB to vectors
        dh_outputs, dh_hidden = self.encoder(conv_story, data['conv_arr_lengths'])
        kb_readout = self.graphmemory.load_graph(data['kb_arr'], dh_hidden)
        # vis_readout = self.visualmemory.load_images(data['img_arr'], dh_hidden)
        vis_readout = self.visualmemory(data['img_arr'], dh_hidden)
        # encoded_hidden = torch.cat((dh_hidden.squeeze(0), kb_readout, vis_readout), dim=1)
        encoded_hidden = torch.cat((dh_hidden.squeeze(0), kb_readout), dim=1)
        # encoded_hidden = dh_hidden.squeeze(0)
        # encoded_hidden = torch.cat((dh_hidden.squeeze(0), kb_readout, vis_readout), dim=1)

        batch_size = len(data['context_arr_lengths'])

        outputs_vocab, decoded_fine, decoded_coarse = self.decoder(
            encoded_hidden,
            data['response'],
            max_target_length,
            batch_size,
            use_teacher_forcing,
            get_decoded_words)

        return outputs_vocab, decoded_fine, decoded_coarse

    def evaluate(self, dev, matric_best, early_stop=None):
        print("STARTING EVALUATION")
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.graphmemory.train(False)
        self.visualmemory.train(False)
        self.decoder.train(False)

        ref, hyp = [], []
        acc, total = 0, 0
        dialog_acc_dict = {}
        F1_pred, F1_cal_pred, F1_nav_pred, F1_wet_pred = 0, 0, 0, 0
        F1_count, F1_cal_count, F1_nav_count, F1_wet_count = 0, 0, 0, 0
        pbar = tqdm(enumerate(dev), total=len(dev))
        new_precision, new_recall, new_f1_score = 0, 0, 0

        if args['dataset'] == 'kvr':
            with open('data/KVR/kvret_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                    else:
                        for item in global_entity['poi']:
                            global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
                global_entity_list = list(set(global_entity_list))

        for j, data_dev in pbar:
            # Encode and Decode
            _, decoded_fine, decoded_coarse = self.encode_and_decode(data_dev, self.max_resp_len, False, True)
            decoded_coarse = np.transpose(decoded_coarse)
            decoded_fine = np.transpose(decoded_fine)
            for bi, row in enumerate(decoded_fine):
                st = ''
                for e in row:
                    if e == 'EOS':
                        break
                    else:
                        st += e + ' '
                st_c = ''
                for e in decoded_coarse[bi]:
                    if e == 'EOS':
                        break
                    else:
                        st_c += e + ' '
                pred_sent = st.lstrip().rstrip()
                pred_sent_coarse = st_c.lstrip().rstrip()
                gold_sent = data_dev['response_plain'][bi].lstrip().rstrip()
                ref.append(gold_sent)
                hyp.append(pred_sent)

                if args['dataset'] == 'kvr':
                    # compute F1 SCORE
                    single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_cal'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_cal_pred += single_f1
                    F1_cal_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_nav'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_nav_pred += single_f1
                    F1_nav_count += count
                    single_f1, count = self.compute_prf(data_dev['ent_idx_wet'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_wet_pred += single_f1
                    F1_wet_count += count
                else:
                    # compute Dialogue Accuracy Score
                    current_id = data_dev['ID'][bi]
                    if current_id not in dialog_acc_dict.keys():
                        dialog_acc_dict[current_id] = []
                    if gold_sent == pred_sent:
                        dialog_acc_dict[current_id].append(1)
                    else:
                        dialog_acc_dict[current_id].append(0)

                # compute Per-response Accuracy Score
                total += 1
                if (gold_sent == pred_sent):
                    acc += 1

                if args['genSample']:
                    self.print_examples(bi, data_dev, pred_sent, pred_sent_coarse, gold_sent)

        # Set back to training mode
        self.encoder.train(True)
        self.graphmemory.train(True)
        self.visualmemory.train(True)
        self.decoder.train(True)

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        acc_score = acc / float(total)
        print("ACC SCORE:\t" + str(acc_score))

        if args['dataset'] == 'kvr':
            F1_score = F1_pred / float(F1_count)
            print("F1 SCORE:\t{}".format(F1_pred / float(F1_count)))
            print("\tCAL F1:\t{}".format(F1_cal_pred / float(F1_cal_count)))
            print("\tWET F1:\t{}".format(F1_wet_pred / float(F1_wet_count)))
            print("\tNAV F1:\t{}".format(F1_nav_pred / float(F1_nav_count)))
            print("BLEU SCORE:\t" + str(bleu_score))
        else:
            dia_acc = 0
            for k in dialog_acc_dict.keys():
                if len(dialog_acc_dict[k]) == sum(dialog_acc_dict[k]):
                    dia_acc += 1
            # print("Dialog Accuracy:\t" + str(dia_acc * 1.0 / len(dialog_acc_dict.keys())))
            print("BLEU SCORE:\t" + str(bleu_score))

        if (early_stop == 'BLEU'):
            if (bleu_score >= matric_best):
                self.save_model('BLEU-' + str(bleu_score))
                print("MODEL SAVED")
            return bleu_score
        elif (early_stop == 'ENTF1'):
            if (F1_score >= matric_best):
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                print("MODEL SAVED")
            return F1_score
        else:
            if (acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(acc_score))
                print("MODEL SAVED")
            return acc_score

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count
