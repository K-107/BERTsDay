# -*- coding: utf-8 -*-

############################### TODO ##########################################
# 필요한 모듈 불러오기
###############################################################################

import argparse
import os
import pickle
import tensorflow as tf

from to_array.bert_to_array import BERTToArray
from models.bert_slot_model import BertSlotModel
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


# read command-line parameters
parser = argparse.ArgumentParser('Evaluating the BERT / ALBERT NLU model')
parser.add_argument('--model', '-m', help = 'Path to BERT / ALBERT NLU model', type = str, required = True)
parser.add_argument('--type', '-tp', help = 'bert or albert', type = str, default = 'bert', required = False)
parser.add_argument('--bertpath', '-bp', help = 'bert model hub path (=modularized pretrained bert path)', type = str, default = "/content/drive/MyDrive/bert-module")



VALID_TYPES = ['bert', 'albert']

args = parser.parse_args()
load_folder_path = args.model
type_ = args.type

# this line is to disable gpu
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1,
                        allow_soft_placement=True,
                        device_count = {'CPU': 1})
sess = tf.compat.v1.Session(config=config)

if type_ == 'bert':
############################### TODO 경로 고치기 ##########################################
    bert_model_hub_path = args.bertpath
###########################################################################################
    is_bert = True
elif type_ == 'albert':
    bert_model_hub_path = 'https://tfhub.dev/google/albert_base/1'
    is_bert = False
else:
    raise ValueError('type must be one of these values: %s' % str(VALID_TYPES))


############################### TODO ##########################################
# 모델과 벡터라이저 불러오기
###############################################################################
vocab_file = os.path.join(bert_model_hub_path, "assets/vocab.korean.rawtext.list")
bert_to_array = BERTToArray(is_bert, vocab_file)

#모델
print('Loading models ...')
if not os.path.exists(load_folder_path):
    raise FileNotFoundError('Folder `%s` not exist' % load_folder_path)

with open(os.path.join(load_folder_path, 'tags_to_array.pkl'), 'rb') as handle:
    tags_to_array = pickle.load(handle)
    slots_num = len(tags_to_array.label_encoder.classes_)

model = BertSlotModel.load(load_folder_path, sess)

while True:
    print('\nEnter your sentence: ')
    try:
        input_text = input().strip()

        #벡터화
        data_text_arr = list(input_text)
        data_tags_arr = list(input_text)
        data_input_ids, data_input_mask, data_segment_ids = bert_to_array.transform(data_text_arr)

        #예측 결과 출력
        inferred_tags, slots_score = model.predict_slots([data_input_ids, data_input_mask, data_segment_ids], tags_to_array)
        print("Inferred tags")
        print(inferred_tags)
        print("Slots score")
        print(slots_score)

    except:
        continue
        
    if input_text == 'quit':
        break



############################### TODO ##########################################
# 사용자가 입력한 한 문장을 슬롯태깅 모델에 넣어서 결과 뽑아내기
###############################################################################

"""
def get_results(input_ids, input_mask, segment_ids, tags_arr, tags_to_array):
    inferred_tags, slots_score = model.predict_slots([data_input_ids, data_input_mask, data_segment_ids],
                                                    tags_to_array)

    gold_tags = [x.split() for x in tags_arr]

    f1_score = metrics.f1_score(flatten(gold_tags), flatten(inferred_tags), average='micro')

    tag_incorrect = ''
    for i, sent in enumerate(input_ids):
        if inferred_tags[i] != gold_tags[i]:
            tokens = bert_to_array.tokenizer.convert_ids_to_tokens(input_ids[i])
            tag_incorrect += 'sent {}\n'.format(tokens)
            tag_incorrect += ('pred: {}\n'.format(inferred_tags[i]))
            tag_incorrect += ('score: {}\n'.format(slots_score[i]))
            tag_incorrect += ('ansr: {}\n\n'.format(gold_tags[i]))


    return f1_score, tag_incorrect


f1_score, tag_incorrect = get_results(data_input_ids, data_input_mask, data_segment_ids,
                                                            data_tags_arr, tags_to_array)
"""

i

tf.compat.v1.reset_default_graph()

