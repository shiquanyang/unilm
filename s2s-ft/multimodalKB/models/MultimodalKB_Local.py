import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.config import *
from models.BertEncoder import BertEncoder
from models.GraphMemory import GraphMemory
from models.VisualMemory import VisualMemory
from models.Decoder import Decoder
from models.ContextRNN import ContextRNN
from models.ImageEncoder import ImageEncoder
from models.Calibration import Calibration
from models.GlobalSemanticsAggregator import GlobalSemanticsAggregator
from models.LocalSemanticsExtractor import LocalSemanticsExtractor
from torch import optim
from torch.optim import lr_scheduler
import random
from utils.masked_cross_entropy_with_ppl import *
import numpy as np
import json
from utils.measures import wer, moses_multi_bleu
import pdb


class MultimodalKBLocal(nn.Module):
    def __init__(self, hidden_size, lang, max_response_len, path, task, lr, n_layers, dropout, input_channels, output_channels, conv_kernel_size, pool_kernel_size):
        super(MultimodalKBLocal, self).__init__()
        self.name = "MultimodalKBLocal"
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
                self.local_semantics_extractor = torch.load(str(path)+'/local_semantics_extractor.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                self.local_semantics_extractor = torch.load(str(path)+'/local_semantics_extractor.th',lambda storage, loc:storage)
        else:
            self.local_semantics_extractor = LocalSemanticsExtractor(lang.n_words, hidden_size, dropout, lang, lang.n_words, hidden_size, n_layers, input_channels, output_channels, conv_kernel_size, pool_kernel_size, n_layers)

        # Initialize optimizers and criterion
        self.local_semantics_extractor_optimizer = optim.Adam(self.local_semantics_extractor.parameters(), lr=lr)
        self.criterion_bce = nn.BCELoss()
        self.reset()

        if USE_CUDA:
            self.local_semantics_extractor.cuda()

    def print_loss(self):
        print_loss = self.loss / self.print_every
        print_loss_v = self.loss_v / self.print_every
        print_loss_c = self.loss_c / self.print_every
        self.print_every += 1
        return 'L:{:.2f},V:{:.2f},C:{:.2f},PPL:{:.2f}'.format(print_loss, print_loss_v, print_loss_c, self.ppl)

    def save_model(self, dec_type):
        name_data = "MULTIWOZ/" if self.task=='multiwoz' else "BABI/"
        layer_info = str(self.n_layers)
        directory = 'save/MultimodalKBLocal-'+args["addName"]+name_data+str(self.task)+'HDD'+str(self.hidden_size)+'BSZ'+str(args['batch'])+'DR'+str(self.dropout)+'L'+layer_info+'lr'+str(self.lr)+'IC'+str(self.input_channels)+'OC'+str(self.output_channels)+'CK'+str(self.conv_kernel_size)+'PK'+str(self.pool_kernel_size)+str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.local_semantics_extractor, directory + '/local_semantics_extractor.th')

    def reset(self):
        self.print_every, self.loss, self.loss_v, self.loss_c, self.ppl = 1, 0, 0, 0, 0

    def _cuda(self, x):
        if USE_CUDA:
            return torch.Tensor(x).cuda()
        else:
            return torch.Tensor(x)

    def train_batch(self, data, clip, reset=0):
        if reset:
            self.reset()
        self.local_semantics_extractor_optimizer.zero_grad()

        conv_story = data['conv_arr']
        input_turns = data['turns']

        # Encode dialog history and KB to vectors
        local_semantic_vectors, lengths = self.local_semantics_extractor(conv_story, input_turns, data['kb_arr'], data['img_arr'])

        return local_semantic_vectors, lengths

    def add_constant_value_dimension(self, tensor):
        size_t = tensor.size()
        extended_tensor = torch.ones(size_t[0], (size_t[1] + 1))
        for i, t in enumerate(tensor):
            extended_tensor[i, :size_t[1]] = t[:size_t[1]]
        return extended_tensor.contiguous().cuda()
        # return extended_tensor

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
        input_turns = data['turns']
        # Encode dialog history and KB to vectors
        local_semantic_vectors, lengths = self.local_semantics_extractor(conv_story, input_turns, data['kb_arr'], data['img_arr'])
        # comment global_semantics_aggregator for debugging
        ngram_vectors = self.global_semantics_aggregator(local_semantic_vectors, input_turns)
        encoded_hidden = ngram_vectors
        # encoded_hidden = local_semantic_vectors


        # dh_outputs, dh_hidden = self.encoder(conv_story, data['conv_arr_lengths'])
        # kb_readout = self.graphmemory.load_graph(data['kb_arr'], dh_hidden)
        # vis_readout = self.visualmemory.load_images(data['img_arr'], dh_hidden)
        # vis_readout = self.visualmemory(data['img_arr'], dh_hidden)
        # encoded_hidden = torch.cat((dh_hidden.squeeze(0), kb_readout, vis_readout), dim=1)
        # encoded_hidden = torch.cat((dh_hidden.squeeze(0), kb_readout), dim=1)
        # encoded_hidden = dh_hidden.squeeze(0)
        # encoded_hidden = torch.cat((dh_hidden.squeeze(0), kb_readout, vis_readout), dim=1)

        # dh_hidden = dh_hidden.squeeze(0)

        # dh_hidden_t = dh_hidden
        # dh_hidden = self.add_constant_value_dimension(dh_hidden)
        # kb_readout = self.add_constant_value_dimension(kb_readout)
        # vis_readout = self.add_constant_value_dimension(vis_readout)
        # b = dh_hidden.size(0)
        # m = dh_hidden.size(1)
        # n = kb_readout.size(1)
        # n = vis_readout.size(1)
        # encoded_hidden = dh_hidden.unsqueeze(2).expand(b, m, n) @ vis_readout.unsqueeze(1).expand(b, m, n)

        # encoded_hidden = torch.einsum('bp, bqr->bpqr', kb_readout, encoded_hidden)

        # calculate calibration distribution
        calibration_vocab = self.calibration(encoded_hidden)

        batch_size = len(data['context_arr_lengths'])

        outputs_vocab, decoded_fine, decoded_coarse = self.decoder(
            encoded_hidden,
            data['response'],
            max_target_length,
            batch_size,
            use_teacher_forcing,
            get_decoded_words,
            calibration_vocab)

        return outputs_vocab, decoded_fine, decoded_coarse, calibration_vocab

    def parse_context_arr_plain(self, context_arr_plain, fout):
        utterances = []
        turn = 0
        list = []
        for element in context_arr_plain:
            if 'PAD' in element:
                subject = element[2]
                relation = element[1]
                object = element[0]
                kb_str = subject + ' ' + relation + ' ' + object + '\n'
                fout.write(kb_str)
            elif '$$$$' in element:
                utterances.append(list)
                continue
            else:
                turn_num = int(element[2].split('turn')[1])
                if turn_num != turn:
                    utterances.append(list)
                    list = []
                    turn = turn_num
                list.append(element[0])
        for i, element in enumerate(utterances):
            for idx, word in enumerate(element):
                if idx == 0:
                    fout.write(word)
                else:
                    fout.write(' ' + word)
            fout.write('\n')

    def output_predictions(self, pred_sent, gold_sent, fout):
        fout.write("Predicted Response: " + pred_sent + '\n')
        fout.write("Gold Response: " + gold_sent + '\n')
        fout.write('\n')

    def evaluate(self, dev, matric_best, early_stop=None):
        print("STARTING EVALUATION")
        # Set to not-training mode to disable dropout
        # self.encoder.train(False)
        # self.graphmemory.train(False)
        # self.visualmemory.train(False)
        self.local_semantics_extractor.train(False)
        self.global_semantics_aggregator.train(False)
        self.decoder.train(False)

        ref, hyp = [], []
        acc, total = 0, 0
        dialog_acc_dict = {}
        F1_pred, F1_cal_pred, F1_nav_pred, F1_wet_pred, F1_pred_multiwoz, F1_pred_babi = 0, 0, 0, 0, 0, 0
        F1_count, F1_cal_count, F1_nav_count, F1_wet_count, F1_count_multiwoz, F1_count_babi = 0, 0, 0, 0, 0, 0
        pbar = tqdm(enumerate(dev), total=len(dev))
        new_precision, new_recall, new_f1_score = 0, 0, 0

        fout = open('generated_responses.txt', 'w')

        global_entity_list = []
        if args['dataset'] == 'kvr':
            with open('data/KVR/kvret_entities.json') as f:
                global_entity = json.load(f)
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                    else:
                        for item in global_entity['poi']:
                            global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
                global_entity_list = list(set(global_entity_list))

        sample_cnt, accumulated_ppl = 0, 0.0
        for j, data_dev in pbar:
            max_target_length = max(data_dev['response_lengths'])
            # Encode and Decode
            all_decoder_outputs_vocab, decoded_fine, decoded_coarse, calibration_vocab = self.encode_and_decode(data_dev, max_target_length, False, True)
            # all_decoder_outputs_vocab, decoded_fine, decoded_coarse, calibration_vocab = self.encode_and_decode(data_dev, self.max_resp_len, False, True)

            # print("all_decoder_outputs_vocab shape:"+str(all_decoder_outputs_vocab.shape))
            # print("data_dev[response] shape:"+str(data_dev['response'].shape))
            # print("data_dev[response_lengths] shape:"+str(len(data_dev['response_lengths'])))
            samples = len(data_dev['context_arr_lengths'])
            sample_cnt += samples
            # Loss calculation and backpropagation
            loss_v, ppl = masked_cross_entropy(
                all_decoder_outputs_vocab.transpose(0, 1).contiguous(),
                data_dev['response'].contiguous(),
                data_dev['response_lengths'])
            accumulated_ppl += ppl.sum(0).item()

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
                self.parse_context_arr_plain(data_dev['context_arr_plain'][bi], fout)
                self.output_predictions(pred_sent, gold_sent, fout)

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
                elif args['dataset'] == 'multiwoz':
                    single_f1_multiwoz, count_multiwoz = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_pred_multiwoz += single_f1_multiwoz
                    F1_count_multiwoz += count_multiwoz
                elif args['dataset'] == 'babi':
                    single_f1_babi, count_babi = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(),
                                                        global_entity_list, data_dev['kb_arr_plain'][bi])
                    F1_pred_babi += single_f1_babi
                    F1_count_babi += count_babi
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

        fout.close()
        # Set back to training mode
        # self.encoder.train(True)
        # self.graphmemory.train(True)
        # self.visualmemory.train(True)
        self.local_semantics_extractor.train(True)
        self.global_semantics_aggregator.train(True)
        self.decoder.train(True)

        # compute ppl here.
        # pdb.set_trace()
        ppl_avg = accumulated_ppl / sample_cnt

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        acc_score = acc / float(total)
        # print("ACC SCORE:\t" + str(acc_score))

        if args['dataset'] == 'kvr':
            F1_score = F1_pred / float(F1_count)
            print("F1 SCORE:\t{:.4f}".format(F1_pred / float(F1_count)))
            print("\tCAL F1:\t{:.4f}".format(F1_cal_pred / float(F1_cal_count)))
            print("\tWET F1:\t{:.4f}".format(F1_wet_pred / float(F1_wet_count)))
            print("\tNAV F1:\t{:.4f}".format(F1_nav_pred / float(F1_nav_count)))
            print("BLEU SCORE:\t" + str(bleu_score))
        elif args['dataset'] == 'multiwoz':
            F1_score_multiwoz = F1_pred_multiwoz / float(F1_count_multiwoz)
            print("F1 SCORE:\t{:.4f}".format(F1_pred_multiwoz / float(F1_count_multiwoz)))
            print("BLEU SCORE:\t" + str(bleu_score))
            print("PPL SCORE:\t{:.2f}".format(ppl_avg))
        elif args['dataset'] == 'babi':
            F1_score_multiwoz = F1_pred_babi / float(F1_count_babi)
            print("ACC SCORE:\t" + str(acc_score))
            print("F1 SCORE:\t{:.4f}".format(F1_pred_babi / float(F1_count_babi)))
            print("BLEU SCORE:\t" + str(bleu_score))
            print("PPL SCORE:\t{:.2f}".format(ppl_avg))
        else:
            dia_acc = 0
            for k in dialog_acc_dict.keys():
                if len(dialog_acc_dict[k]) == sum(dialog_acc_dict[k]):
                    dia_acc += 1
            # print("Dialog Accuracy:\t" + str(dia_acc * 1.0 / len(dialog_acc_dict.keys())))
            print("BLEU SCORE:\t" + str(bleu_score))
            print("PPL SCORE:\t{:.2f}".format(ppl_avg))

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
