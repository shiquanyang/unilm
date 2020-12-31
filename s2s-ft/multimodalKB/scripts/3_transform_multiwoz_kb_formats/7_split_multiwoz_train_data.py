import numpy
import random
import pdb


file_name = '/Users/shiquan/PycharmProjects/MultiModalKB/s2s-ft/multimodalKB/data/2_20K_multimodal_dataset/all_instances_extended_kb_lowercase_gold.txt'
file_filtered = '/Users/shiquan/PycharmProjects/MultiModalKB/s2s-ft/multimodalKB/data/2_20K_multimodal_dataset/all_instances_extended_kb_lowercase_gold_filtered.txt'
input_file = '/Users/shiquan/PycharmProjects/MultiModalKB/s2s-ft/multimodalKB/data/2_20K_multimodal_dataset/all_instances_extended_kb_lowercase_gold_filtered.txt'
output_train_file = '/Users/shiquan/PycharmProjects/MultiModalKB/s2s-ft/multimodalKB/data/2_20K_multimodal_dataset/train_split.txt'
output_dev_file = '/Users/shiquan/PycharmProjects/MultiModalKB/s2s-ft/multimodalKB/data/2_20K_multimodal_dataset/dev_split.txt'
output_test_file = '/Users/shiquan/PycharmProjects/MultiModalKB/s2s-ft/multimodalKB/data/2_20K_multimodal_dataset/test_split.txt'
data_list = []
fout_train = open(output_train_file, 'w')
fout_dev = open(output_dev_file, 'w')
fout_test = open(output_test_file, 'w')
fout_filter = open(file_filtered, 'w')


####################
# Filter Noisy Data
####################
print("Reading lines from {}".format(file_name))
kb_rec_cnt = 0
with open(file_name) as fin:
    for line in fin:
        line = line.strip()
        if line:
            if '\t' in line:
                if kb_rec_cnt < 30:
                    continue
                fout_filter.write(line + "\n")
            else:
                kb_rec_cnt += 1
                fout_filter.write(line + "\n")
        else:
            kb_rec_cnt = 0
            fout_filter.write("\n")
fout_filter.close()


with open(input_file, 'r') as fin:
    sample_cnt = 0
    data = []
    for line in fin:
        line = line.strip()
        if line:
            data.append(line)
        else:
            data_list.append(data)
            sample_cnt += 1
            data = []

train_cnt = int(0.7 * sample_cnt)
dev_cnt = int(0.1 * sample_cnt)
test_cnt = int(sample_cnt - train_cnt - dev_cnt)
train_data = random.sample(data_list, train_cnt)

diff_list = []
for element in data_list:
    if element in train_data:
        continue
    diff_list.append(element)

dev_data = random.sample(diff_list, dev_cnt)

diff_list_new = []
for element in diff_list:
    if element in dev_data:
        continue
    diff_list_new.append(element)

# test_data = random.sample(diff_list_new, test_cnt)
test_data = diff_list_new

print("Train data size:"+str(len(train_data)))
print("Dev data size:"+str(len(dev_data)))
print("Test data size:"+str(len(test_data)))

for data in train_data:
    for element in data:
        fout_train.write(element+'\n')
    fout_train.write('\n')
fout_train.close()

for data in dev_data:
    for element in data:
        fout_dev.write(element+'\n')
    fout_dev.write('\n')
fout_dev.close()

for data in test_data:
    for element in data:
        fout_test.write(element+'\n')
    fout_test.write('\n')
fout_test.close()

