import numpy as np
import pdb


path = "/home/yimeng/shiquan/ParlAI/parlai/mturk/run_data/live/samples.txt"
vocabulary_size, max_turns_per_dialog, min_turns_per_dialog, average_turns_per_dialog, average_tokens_per_turn, total_number_of_turns = 0, 0, 0, 0, 0, 0
dict = {}
turns_per_dialog = []
tokens_per_turn = []
with open(path, 'r') as f:
    turns_tmp = 0
    for line in f:
        line = line.strip()
        if line:
            line_list = line.split('#')
            if len(line_list) == 2:
                total_number_of_turns += 1
                turns_tmp += 1
                user = line_list[0]
                agent = line_list[1]
                user_tokens = user.split(' ')
                agent_tokens = agent.split(' ')
                tokens_per_turn.append((int(len(user_tokens))+int(len(agent_tokens))))
                for token in user_tokens:
                    if token not in dict:
                        dict[token] = 1
                    else:
                        dict[token] += 1
                for token in agent_tokens:
                    if token not in dict:
                        dict[token] = 1
                    else:
                        dict[token] += 1
        else:
            # pdb.set_trace()
            turns_per_dialog.append(int(turns_tmp))
            turns_tmp = 0

print("Vocabulary size:" + str(len(dict)))
print("Max turns per dialog:" + str(max(turns_per_dialog)))
print("Min turns per dialog:" + str(min(turns_per_dialog)))
print("Avg turns per dialog:" + str(sum(turns_per_dialog) / float(len(turns_per_dialog))))
print("Avg tokens per turn:" + str(sum(tokens_per_turn) / float(len(tokens_per_turn))))
print("Total number of turns:" + str(total_number_of_turns))

