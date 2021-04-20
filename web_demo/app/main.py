# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import os
import sys
import pickle
import tensorflow as tf
sys.path.append(os.path.join(os.getcwd(), "Bert_fine_tuning"))
from to_array.bert_to_array import BERTToArray
from models.bert_slot_model import BertSlotModel
from to_array.tokenizationK import FullTokenizer
from sklearn import metrics

load_folder_path = os.path.join(os.getcwd(), "Fine_tuned") # 파인튜닝 경로
bert_model_hub_path = os.path.join(os.getcwd(), "Bert_pretrained") #프리트레인 경로
vocab_file = os.path.join(bert_model_hub_path, "assets/vocab.korean.rawtext.list")
is_bert = True

# 슬롯태깅 모델과 벡터라이저 불러오기
print("===============초기화 중===============")
global graph
graph = tf.get_default_graph()
# this line is to enable gpu
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = tf.ConfigProto(intra_op_parallelism_threads=0,
                        inter_op_parallelism_threads=0,
                        allow_soft_placement=True,
                        device_count = {'GPU': 1})
sess = tf.compat.v1.Session(config=config)
bert_to_array = BERTToArray(is_bert, vocab_file)
with open(os.path.join(load_folder_path, 'tags_to_array.pkl'), 'rb') as handle:
    tags_to_array = pickle.load(handle)
    slots_num = len(tags_to_array.label_encoder.classes_)
model = BertSlotModel.load(load_folder_path, sess)
tokenizer = FullTokenizer(vocab_file=vocab_file)
print("===============초기화 완료===============")


# 플라스크 앱 초기화
app = Flask("BERTsDay Chatbot")
app.static_folder = 'web_demo/app/static'
app.template_folder = "web_demo/app/templates"

@app.route("/")
def home():

############################### TODO ##########################################
# 슬롯 사전 만들기
    app.slot_dict = {'start': None, 'end': None, 'date':None, 'num':None ,'name': None, 'phone': None}

    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg').strip() # 사용자가 입력한 문장

    #벡터화
    input_text = ' '.join(tokenizer.tokenize(userText))
    tokens = input_text.split()
    data_text_arr = [input_text]
    data_input_ids, data_input_mask, data_segment_ids = bert_to_array.transform(data_text_arr)
    
    #모델 불러오고 슬롯태깅
    with graph.as_default():
        with sess.as_default():
            inferred_tags, slots_score = model.predict_slots([data_input_ids, data_input_mask, data_segment_ids], tags_to_array)

############################### TODO ##########################################
# 1. 사용자가 입력한 한 문장을 슬롯태깅 모델에 넣어서 결과 뽑아내기
# 2. 추출된 슬롯 정보를 가지고 더 필요한 정보 물어보는 규칙 만들기 (if문)
    try:
        app.slot_dict['start'] = ''
        print(app.slot_dict)
    except Exception as e:
        return "오류가 발생했습니다, 페이지를 다시 열어주세요"
    
    return f"input_text: {tokens}<br>inferred_tags: {inferred_tags} <br>slots_score: {slots_score}"



