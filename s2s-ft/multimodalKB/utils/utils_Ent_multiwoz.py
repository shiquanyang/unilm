import json
import numpy as np
from utils.config import *
from utils.utils_temp import entityList, get_type_dict, get_img_fea, load_img_fea
import logging
from utils.utils_general import *
import ast


def read_langs(file_name, global_entity, type_dict, img_path, max_line=None):
    print("Reading lines from {}".format(file_name))
    data, context_arr, conv_arr, kb_arr, img_arr = [], [], [], [], []
    max_res_len, sample_counter, turn = 0, 0, 0
    src_tokens = ''
    image_feas = load_img_fea(img_path)
    with open(file_name) as fin:
        cnt_lin = 1
        for line in fin:
            line = line.strip()
            if line:
                nid, line = line.split(' ', 1)
                if '\t' in line:
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
                        'src_tokens': src_tokens}
                    data.append(data_detail)

                    gen_r = generate_memory(r, "$s", str(turn), image_feas)
                    if len(gen_r[0]) > 4:
                        print(gen_r)
                        print(u, r)
                    context_arr += gen_r
                    # conv_arr += gen_r
                    r_token = r.split(' ')
                    conv_arr += r_token
                    src_tokens = src_tokens + ' ' + r
                    if max_res_len < len(r.split()):
                        max_res_len = len(r.split())
                    sample_counter += 1
                    turn += 1
                else:
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
                turn = 0
                context_arr, conv_arr, kb_arr, img_arr = [], [], [], []
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

def prepare_data_seq(task, batch_size=100):
    # data_path = '/Users/shiquan/PycharmProjects/Multimodal-Knowledge-Base/data/0_synthetic/dialog-babi'
    # img_path = '/Users/shiquan/PycharmProjects/Multimodal-Knowledge-Base/images/restaurant'
    data_path_babi = '/home/yimeng/shiquan/Multimodal-Knowledge-Base/data/0_synthetic/dialog-babi'
    data_path = '/home/yimeng/shiquan/Multimodal-Knowledge-Base/data/1_multiwoz/multiwoz'
    img_path = '/home/yimeng/shiquan/Multimodal-Knowledge-Base/images/restaurant'
    file_train = '{}-trn-multimodal-phase1-version-lowercase.txt'.format(data_path)
    # file_train = '{}-task{}trn.txt'.format(data_path, task)
    file_dev = '{}-dev-multimodal-phase1-version-lowercase.txt'.format(data_path)
    # file_dev = '{}-task{}dev.txt'.format(data_path, task)
    file_test = '{}-tst-multimodal-phase1-version-lowercase.txt'.format(data_path)
    # file_test = '{}-task{}tst.txt'.format(data_path, task)
    kb_path = data_path_babi + '-kb-all.txt'
    file_test_OOV = '{}-tst-multimodal-OOV.txt'.format(data_path)  # no-OOV dataset available!
    type_dict = get_type_dict(kb_path, dstc2=False)
    global_ent = entityList(kb_path, 4)

    pair_train, train_max_len = read_langs(file_train, global_ent, type_dict, img_path)
    pair_dev, dev_max_len = read_langs(file_dev, global_ent, type_dict, img_path)
    pair_test, test_max_len = read_langs(file_test, global_ent, type_dict, img_path)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len)

    lang = Lang()

    train = get_seq(pair_train, lang, batch_size, True)
    dev = get_seq(pair_dev, lang, batch_size, False)
    test = get_seq(pair_test, lang, batch_size, False)

    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, lang, max_resp_len

if __name__ == "__main__":
    prepare_data_seq(4)
#     # load_img_fea('/Users/shiquan/PycharmProjects/Multimodal-Knowledge-Base/images/restaurant')