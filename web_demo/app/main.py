# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import os
import sys

#sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "Bert_fine_tuning"))
from to_array.bert_to_array import BERTToArray
from models.bert_slot_model import BertSlotModel
from to_array.tokenizationK import FullTokenizer
import pickle
import tensorflow as tf

bert_model_hub_path = os.path.join(os.getcwd(), "Bert_pretrained") # 프리트레인 경로
load_folder_path = os.path.join(os.getcwd(), "Fine_tuned") # 파인튜닝 경로
is_bert = True

# this line is to enable gpu
os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = tf.ConfigProto(intra_op_parallelism_threads=0,
                        inter_op_parallelism_threads=0,
                        allow_soft_placement=True,
                        device_count = {'GPU': 1})
sess = tf.compat.v1.Session(config=config)

# 모델과 벡터라이저 불러오기
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
tokenizer = FullTokenizer(vocab_file=vocab_file)
print("모델 로드 성공")


app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():

############################### TODO ##########################################
# 슬롯 사전 만들기
    app.slot_dict = {'a_slot': None, 'b_slot':None}
###############################################################################


    return render_template("index.html")
    
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg').strip() # 사용자가 입력한 문장

############################### TODO ##########################################
# 1. 사용자가 입력한 한 문장을 슬롯태깅 모델에 넣어서 결과 뽑아내기
# 2. 추출된 슬롯 정보를 가지고 더 필요한 정보 물어보는 규칙 만들기 (if문)
    app.slot_dict['a_slot'] = ''
    print(app.slot_dict)

    return 'hi' # 챗봇이 이용자에게 하는 말을 return
###############################################################################

@app.route("/test")
def TEST():
    tf.compat.v1.reset_default_graph()
    input_text = request.args.get('msg').strip()
    input_text = ' '.join(tokenizer.tokenize(input_text))

    data_text_arr = [input_text]
    data_input_ids, data_input_mask, data_segment_ids = bert_to_array.transform(data_text_arr)

    print("=============")
    print("input text:", input_text)
    print("data_text_arr:", data_text_arr)
    print("data_input_ids:", data_input_ids)
    print("data_input_mask:", data_input_mask)
    print("data_segment_ids:", data_segment_ids)
    print("=============")

    #예측 결과 출력
    inferred_tags, slots_score = model.predict_slots([data_input_ids, data_input_mask, data_segment_ids], tags_to_array)

    return render_template("test.html", input = input_text, input_tokens = data_text_arr, inferred_tags=inferred_tags, slots_score=slots_score)
