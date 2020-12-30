from utils.config import *
from models.MultimodalKB import *
from models.MultimodalKB_GlobalLocal import *
import pdb


'''
Command:

python test.py -ds= -path= 

'''
directory = args['path'].split("/")
task = directory[2].split('HDD')[0]
HDD = directory[2].split('HDD')[1].split('BSZ')[0]
L = directory[2].split('L')[1].split('lr')[0].split("-")[0]
IC = directory[2].split('IC')[1].split('OC')[0]
OC = directory[2].split('OC')[1].split('CK')[0]
CK = directory[2].split('CK')[1].split('PK')[0]
PK = directory[2].split('PK')[1].split('BLEU')[0]
decoder = directory[1].split('-')[0]
BSZ = int(directory[2].split('BSZ')[1].split('DR')[0])
DS = 'multiwoz' if 'multiwoz' in directory[1].split('-')[1].lower() else 'babi'

if DS=='multiwoz':
    from utils.utils_Ent_multiwoz import *
elif DS=='babi':
    from utils.utils_Ent_babi import *
else:
    print("You need to provide the --dataset information")

train, dev, test, lang, max_resp_len = prepare_data_seq(task, batch_size=BSZ)
# train, dev, test, testOOV, lang, max_resp_len = prepare_data_seq(task, batch_size=BSZ)

model = globals()[decoder](
	int(HDD),
	lang,
	max_resp_len,
	args['path'],
	"",
	lr=0.0,
	n_layers=int(L),
	dropout=0.0,
    input_channels=int(IC),
    output_channels=int(OC),
    conv_kernel_size=int(CK),
    pool_kernel_size=int(PK))

acc_test = model.evaluate(test, 1e7)
# if testOOV!=[]:
# 	acc_oov_test = model.evaluate(testOOV, 1e7)