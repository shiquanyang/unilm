import os
import json
import pdb


def find_all_files(path):
    for root, ds, fs in os.walk(path):
        for f in fs:
            full_name = os.path.join(root, f)
            yield full_name

def parse_dialogue_data(in_file):
    data = []
    with open(in_file, 'r') as f:
        file = json.load(f)
        for message in file['messages']:
            id = message['id']
            text = message['text']
            # if id == 'Dialogue Collector':
            if id == 'You':
                data.append(text)
    return data

def check_short_conversation(data):
    cnt = 0
    is_short_turn = 0
    turn_cnt = 0
    for element in data:
        turn_cnt += 1
        element_list = element.strip().split(' ')
        if len(element_list) <= 3:
            is_short_turn += 1
        if turn_cnt == 2 and is_short_turn == 2:
            cnt += 1
        if turn_cnt == 2:
            turn_cnt = 0
            is_short_turn = 0
    if cnt > 1:
        return True
    else:
        return False

def check_reptitive_conversation(data):
    questions = {}
    turn_cnt = 0
    for element in data:
        turn_cnt += 1
        if turn_cnt % 2 == 1:
            if element not in questions:
                questions[element] = 1
            else:
                questions[element] += 1
    for key in questions:
        if questions[key] > 1:
            return True
    return False

def main():
    # path = '/Users/shiquan/PycharmProjects/Multimodal-Knowledge-Base/data'
    path = '/home/yimeng/shiquan/ParlAI/parlai/mturk/run_data/live'
    output_file = '/home/yimeng/shiquan/ParlAI/parlai/mturk/run_data/live/samples_filtered.txt'
    badcase_file = '/home/yimeng/shiquan/ParlAI/parlai/mturk/run_data/live/badcases.txt'
    short_file = '/home/yimeng/shiquan/ParlAI/parlai/mturk/run_data/live/short_conversations.txt'
    reptitive_file = '/home/yimeng/shiquan/ParlAI/parlai/mturk/run_data/live/reptitive_conversations.txt'
    f_out = open(output_file, 'w')
    f_out_badcase = open(badcase_file, 'w')
    f_out_short = open(short_file, 'w')
    f_out_reptitive = open(reptitive_file, 'w')
    total_samples, valid_samples, badcases, short_samples, reptitive_samples = 0, 0, 0, 0, 0
    for i in find_all_files(path):
        if 'workers' in i:
            total_samples += 1
            ret = parse_dialogue_data(i)
            if len(ret) <= 6 and len(ret) > 0:
                cnt = 0
                for element in ret:
                    cnt += 1
                    if cnt % 2 == 1:
                        f_out_badcase.write(element + '# ')
                    if cnt % 2 == 0:
                        f_out_badcase.write(element + '\n')
                badcases += 1
                f_out_badcase.write('\n')
            # pdb.set_trace()
            elif len(ret) > 6:
                is_short = check_short_conversation(ret)
                is_reptitive = check_reptitive_conversation(ret)
                if is_short:
                    cnt = 0
                    for element in ret:
                        cnt += 1
                        if cnt % 2 == 1:
                            f_out_short.write(element + '# ')
                        if cnt % 2 == 0:
                            f_out_short.write(element + '\n')
                    short_samples += 1
                    f_out_short.write('\n')
                if is_reptitive:
                    cnt = 0
                    for element in ret:
                        cnt += 1
                        if cnt % 2 == 1:
                            f_out_reptitive.write(element + '# ')
                        if cnt % 2 == 0:
                            f_out_reptitive.write(element + '\n')
                    reptitive_samples += 1
                    f_out_reptitive.write('\n')
                if not is_short and not is_reptitive:
                    cnt = 0
                    for element in ret:
                        cnt += 1
                        if cnt % 2 == 1:
                            f_out.write(element + '# ')
                        if cnt % 2 == 0:
                            f_out.write(element + '\n')
                    valid_samples += 1
                    f_out.write('\n')
            elif len(ret) == 0:
                continue
            # print(i)
    print("Total samples: " + str(total_samples))
    print("Valid samples: " + str(valid_samples))
    print("Badcases: " + str(badcases))
    print("Short samples: " + str(short_samples))
    print("Reptitive samples: " + str(reptitive_samples))
    f_out.close()
    f_out_badcase.close()
    f_out_short.close()
    f_out_reptitive.close()

if __name__ == '__main__':
    main()
    pass