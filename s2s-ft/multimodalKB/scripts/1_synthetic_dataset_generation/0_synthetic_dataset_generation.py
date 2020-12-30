import numpy
import random
from random import choice
import os
import json

hotel_domain = ['hotel kids', 'hotel room', 'hotel room beach', 'hotel room with queen size bed', 'hotel room with single size bed', 'hotel with balcony', 'hotel with garden', 'hotel with in room pool', 'hotel with lake view', 'hotel with lazy river', 'hotel with mountain view', 'hotel with outdoor pool']
rest_domain = ['restaurant-dining-room', 'restaurant-kid-friendly', 'restaurant-riverside', 'restaurant-rooftop', 'restaurant-rooftop-bar', 'restaurant-with-balcony', 'restaurant-with-bar', 'restaurant-with-couches', 'restaurant-with-dance-floor', 'restaurant-with-garden', 'restaurant-with-private-room', 'restaurant-with-sea-view']
rest_dinning_subtypes = ['window', 'wall', 'decorationluxury', 'decorationmodern', 'decorationinferior', 'tablefortwo', 'tableforfour', 'tableforsix']
rest_domain_length = {'restaurant-dining-room':353,
                      'restaurant-kid-friendly':119,
                      'restaurant-riverside':211,
                      'restaurant-rooftop':299,
                      'restaurant-rooftop-bar':203,
                      'restaurant-with-balcony':67,
                      'restaurant-with-bar':124,
                      'restaurant-with-couches':140,
                      'restaurant-with-dance-floor':168,
                      'restaurant-with-garden':174,
                      'restaurant-with-private-room':308,
                      'restaurant-with-sea-view':139}
rest_domain_consume_length = {'restaurant-dining-room':0,
                              'restaurant-kid-friendly':0,
                              'restaurant-riverside':0,
                              'restaurant-rooftop':0,
                              'restaurant-rooftop-bar':0,
                              'restaurant-with-balcony':0,
                              'restaurant-with-bar':0,
                              'restaurant-with-couches':0,
                              'restaurant-with-dance-floor':0,
                              'restaurant-with-garden':0,
                              'restaurant-with-private-room':0,
                              'restaurant-with-sea-view':0}

tasks = ['trn', 'dev', 'tst']
rest_images = {}
images_consume_history = []

for task in tasks:
    input_file = '/Users/shiquan/PycharmProjects/Multimodal-Knowledge-Base/data/0_synthetic/dialog-babi-task4{}.txt'.format(task)
    output_file = '/Users/shiquan/PycharmProjects/Multimodal-Knowledge-Base/data/0_synthetic/dialog-babi-{}-multimodal.txt'.format(task)
    images_path = '/Users/shiquan/PycharmProjects/Multimodal-Knowledge-Base/images/restaurant/'
    qa_template_path = '/Users/shiquan/PycharmProjects/Multimodal-Knowledge-Base/scripts/1_synthetic_dataset_generation/qa_template/QA_template.json'

    f_out = open(output_file, 'w')

    # rest_images = {}
    # images_consume_history = []

    with open(qa_template_path, 'r') as f:
        qa_template = json.load(f)

    with open(input_file, 'r') as fin:
        cnt_line, sample_counter, has_assigned_image, has_output_image = 1, 1, 0, 0
        for line in fin:
            line = line.strip()
            if line:
                line_list = line.split(' ')
                if len(line_list) == 4:
                    # output one line
                    new_line = str(cnt_line) + " " + ' '.join(line_list[1:])
                    f_out.write(new_line + '\n')
                    cnt_line += 1
                    if not has_assigned_image:
                        rest_name = line_list[1]
                        # if restaurant has not been assigned image before, assign one.
                        if rest_name not in rest_images:
                            rest_images[rest_name] = {}
                            # check valid rest_domain so far.
                            rest_domain_valid = [key for key in rest_domain_length if (rest_domain_length[key] > rest_domain_consume_length[key])]
                            sampled_image_types = random.sample(rest_domain_valid, 2)
                            for type in sampled_image_types:
                                file_names = []
                                path = images_path + type
                                for parent, dirnames, filenames in os.walk(path):
                                    file_names = filenames
                                while True:
                                    x = random.randint(0, len(filenames)-1)
                                    file_name = file_names[x]
                                    image_path = path + '/'+ file_name
                                    if image_path not in images_consume_history:
                                        rest_domain_consume_length[type] += 1
                                        break
                                rest_images[rest_name][type] = image_path
                                images_consume_history.append(image_path)
                        has_assigned_image = 1
                else:
                    # output one line
                    if not has_output_image:
                        image_info = rest_images[rest_name]
                        num = 0
                        for key in image_info.keys():
                            f_out.write(str(cnt_line) + ' ' + rest_name + ' R_image{} '.format(num) + image_info[key] + '\n')
                            num += 1
                            cnt_line += 1
                        has_output_image = 1
                    new_line = str(cnt_line) + " " + ' '.join(line_list[1:])
                    f_out.write(new_line + "\n")
                    cnt_line += 1
            else:
                # use rest_name to get image types
                types = rest_images[rest_name]
                types_list = list(types.keys())
                diff = list(set(rest_domain) - set(types))
                diff_list = random.sample(diff, 1)
                # types_list = types_list + random.sample(list(diff), 1)
                for key in types_list:
                    # if random.randint(0, 9) / 10 > 0.8:
                    #     new_key = random.sample(list(diff), 1)
                    #     new_key = ''.join(new_key)
                    # else:
                    #     new_key = key
                    new_key = key
                    if new_key == 'restaurant-dining-room':
                        subtype = random.sample(rest_dinning_subtypes, 1)
                        subtype = ''.join(subtype)
                        question_candidates = qa_template['restaurant'][new_key][subtype]['questions']
                        pos_answer_candidates = qa_template['restaurant'][new_key][subtype]['positive_answers']
                        neg_answer_candidates = qa_template['restaurant'][new_key][subtype]['negative_answers']
                    else:
                        question_candidates = qa_template['restaurant'][new_key]['questions']
                        pos_answer_candidates = qa_template['restaurant'][new_key]['positive_answers']
                        neg_answer_candidates = qa_template['restaurant'][new_key]['negative_answers']
                    question = random.sample(question_candidates, 1)
                    # if random.randint(0, 9) / 10 > 0.8:
                        # answer = 'No'
                        # answer = random.sample(neg_answer_candidates, 1)
                    # else:
                        # answer = 'Yes'
                    answer = random.sample(pos_answer_candidates, 1)
                    output = str(cnt_line) + ' ' + ' '.join(question) + '\t' + ''.join(answer) + '\n'
                    f_out.write(output)
                    cnt_line += 1
                for key in diff_list:
                    new_key = key
                    if new_key == 'restaurant-dining-room':
                        subtype = random.sample(rest_dinning_subtypes, 1)
                        subtype = ''.join(subtype)
                        question_candidates = qa_template['restaurant'][new_key][subtype]['questions']
                        pos_answer_candidates = qa_template['restaurant'][new_key][subtype]['positive_answers']
                        neg_answer_candidates = qa_template['restaurant'][new_key][subtype]['negative_answers']
                    else:
                        question_candidates = qa_template['restaurant'][new_key]['questions']
                        pos_answer_candidates = qa_template['restaurant'][new_key]['positive_answers']
                        neg_answer_candidates = qa_template['restaurant'][new_key]['negative_answers']
                    question = random.sample(question_candidates, 1)
                    # if random.randint(0, 9) / 10 > 0.8:
                        # answer = 'No'
                        # answer = random.sample(neg_answer_candidates, 1)
                    # else:
                        # answer = 'Yes'
                    answer = random.sample(neg_answer_candidates, 1)
                    output = str(cnt_line) + ' ' + ' '.join(question) + '\t' + ''.join(answer) + '\n'
                    f_out.write(output)
                    cnt_line += 1
                f_out.write('\n')
                cnt_line = 1
                has_assigned_image = 0
                has_output_image = 0
