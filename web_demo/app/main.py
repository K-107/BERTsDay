# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import os
import sys
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
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
    app.slot_dict = {'start': '', 'end': '', 'date': '', 'person': '', 'name': '', 'phone': ''}
    app.filled_num = 0

    return render_template("index.html")

score_limit = 0.75  
answer_name_arr = ['성함이 어떻게 되시나요?', '이름을 말해주세요.']
answer_phone_arr = ['연락 가능한 번호를 써주세요.(예시 : 010-1234-1234)', '전화번호를 알려주세요.(예시 : 010-1234-1234)', '예약자 분의 번호를 입력해주세요.(예시 : 010-1234-1234)']
answer_date_arr = ['몇 월 며칠에 예약하고 싶으신가요?', '예약하고 싶은 월일을 입력해주세요. (예시: 1월 3일)', '예약하시려는 날짜를 알려주세요.']
answer_start_arr = ['몇 시로 예약하실 건가요?', '몇 시부터 사용하실 건가요?', '사용 시작 시간을 알려주세요.']
answer_end_arr = ['몇 시까지 이용하실 건가요?', '언제까지 사용하실 건가요?', '종료 시간을 알려주세요.']
answer_person_arr = ['총 몇 명이신가요?', '몇 명이서 쓰실 건가요?', '이용 인원을 말씀해주세요?']

date_dict = {'오늘':0,'금일':0,'내일':1,'낼':1,'모레':2}
person_dict = {'혼자':1,'두명':2,'둘이':2,'세명':3,'셋이':3,'네명':4,'다섯':5,'여섯':6,'일곱':7,'여덟':8}

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg').strip() # 사용자가 입력한 문장

    #벡터화
    input_text = ' '.join(tokenizer.tokenize(userText))
    token_list = input_text.split()
    data_text_arr = [input_text]
    data_input_ids, data_input_mask, data_segment_ids = bert_to_array.transform(data_text_arr)
    
    #모델 불러오고 슬롯태깅
    with graph.as_default():
        with sess.as_default():
            inferred_tags, slots_score = model.predict_slots([data_input_ids, data_input_mask, data_segment_ids], tags_to_array)
    
    today = datetime.now()
    for key, value in enumerate(date_dict):
        if value in input_text:
            date_val = today + timedelta(days=date_dict[value])
            app.slot_dict['date'] = str(date_val.month) + "월" + str(date_val.day) + "일"
            
    for key, value in enumerate(person_dict):
        if value in input_text:
            app.slot_dict['person'] = str(person_dict[value])+'명'

    try:
        # 1. 사용자가 입력한 한 문장을 슬롯태깅 모델에 넣어서 결과 뽑아내기
        for i in range(0,len(inferred_tags[0])):
            if slots_score[0][i] >= score_limit:
                if inferred_tags[0][i]=='날짜':
                    if app.slot_dict['date'] == "": app.filled_num += 1
                    app.slot_dict['date'] += token_list[i]     
                elif inferred_tags[0][i]=='시작시간':
                    if app.slot_dict['start'] == "": app.filled_num += 1
                    app.slot_dict['start'] += token_list[i]     
                elif inferred_tags[0][i]=='종료시간':
                    if app.slot_dict['end'] == "": app.filled_num += 1
                    app.slot_dict['end'] += token_list[i]     
                elif inferred_tags[0][i]=='인원':
                    if app.slot_dict['person'] == "": app.filled_num += 1
                    app.slot_dict['person'] += token_list[i] 
                elif inferred_tags[0][i]=='이름':
                    if app.slot_dict['name'] == "": app.filled_num += 1
                    app.slot_dict['name'] += token_list[i]
                elif inferred_tags[0][i]=='번호':
                    if app.slot_dict['phone'] == "": app.filled_num += 1
                    app.slot_dict['phone'] += token_list[i]   
        
        # 디버깅용 상태 표시 문장
        if app.debug:
                response = f"<br><br>slot_dict: {app.slot_dict}<br>input_text: {token_list}<br>inferred_tags: {inferred_tags} <br>slots_score: {slots_score}"
        else:
                response = ""

        # 2. 추출된 슬롯 정보를 가지고 더 필요한 정보 물어보는 규칙 만들기 (if문)
        if ((app.slot_dict['start'] != "") and (app.slot_dict['end'] != "") and (app.slot_dict['person'] != "")and (app.slot_dict['date'] != "") and (app.slot_dict['name'] != "") and (app.slot_dict['phone'] != "")):
            return '예약이 완료되었습니다. 예약을 종료합니다.' + response
    
        elif ((app.slot_dict['start'] == "") and (app.slot_dict['end'] == "") and (app.slot_dict['person'] == "") and (app.slot_dict['date'] == "") and (app.slot_dict['name'] == "") and (app.slot_dict['phone'] == "")):
            return '죄송합니다 제가 이해를 잘 못해서 다시 한번 입력해주세요.' + response

        else:
            if app.slot_dict['date'] == '':
                return str(np.random.choice(answer_date_arr, 1)[0])+ response
            elif app.slot_dict['start'] == '':
                return str(np.random.choice(answer_start_arr, 1)[0]) + response
            elif app.slot_dict['end'] == '':
                return str(np.random.choice(answer_end_arr, 1)[0]) + response
            elif app.slot_dict['person'] == '':
                return str(np.random.choice(answer_person_arr, 1)[0]) + response
            elif app.slot_dict['name'] == '':
                return str(np.random.choice(answer_name_arr, 1)[0]) + response
            elif app.slot_dict['phone'] == '':
                return str(np.random.choice(answer_phone_arr, 1)[0]) + response

    except Exception as e:
        print(e)
        return str(e) + "<br>오류가 발생했습니다, 페이지를 다시 열어주세요"
    
    return "이 문장은 출력될 일이 없습니다."
