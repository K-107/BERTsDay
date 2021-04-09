import os

#이름
name_ls = []

#번호
phone_ls = []

#날짜
import calendar
from datetime import datetime
dt = datetime.today()
date_ls = []
for month in range(1, 13):
    for day in range(1, calendar.monthrange(dt.year, month)[1]+1):
        date_ls.append(f"{month}월 {day}일")

#시간
time_ls = []
for i in range(0,25):
    for j in range(i+1, 25):
        time_ls.append(f"{i}시 부터 {j}시 까지")

#인원
person_ls = []
for i in range(1,9):
    person_ls.append(f"{i}명")
    person_ls.append(f"{i}명이서")

#후미
tail1=["스터디룸", "방", ""]
tail2=["예약", ""]
tail3=["해줘", "해주세요", "하겠습니다", "할게" ,"할게요", "하겠습니다", "부탁", "부탁할게", "부탁할게요", "부탁드립니다", \
"부탁드리겠습니다.", "돼", "되나요", "됩니까", "되겠습니까", "있어", "있어요", "있나요", "있습니까", "잡을게", "잡으려고", "잡으려고요", \
"잡으려고 합니다", "잡을 수 있어", "잡을 수 있어요", "잡을 수 있나요", "잡을 수 있을까요", "잡을 수 있습니까", \
    "가능해", "가능한가", "가능한가요", "가능합니까", "가능해요", "남았어", "남았어요", "남았나요", "남았습니까", "남은 거 있어", \
        "남은 거 있나요", "남은 거 있어요", "남은 거 있습니까"]

# 
loop = 0 # 하나의 경우의 수에 몇개의 데이터가 생성됐는지?
total = 0 # 전체적으로 몇개의 데이터가 생성됐는지?

with open("output.txt", 'w') as f:
    # 1개의 슬롯만 주어진 경우 - 시간만 주어진 경우 ex) 10시부터 12시까지 스터디룸 예약 돼?
    loop = 0
    for time_element in time_ls:
        for tail1_element in tail1:
            for tail2_element in tail2:
                for tail3_element in tail3:
                    sentence = [f"/시간; {time_element}/ {tail1_element} {tail2_element} {tail3_element}\n"]
                    for sentence_element in sentence:
                        loop += 1
                        f.write(sentence_element)
    total += loop
    print(f"시간만 주어진 경우 문장 데이터 생성 완료, 만들어진 갯수: {loop}")
    print("예시)"+ sentence_element)


    # 2개의 슬롯만 주어진 경우 - 시간, 인원이 주어진 경우 ex) 10시부터 12시까지 3명이서 스터디룸 예약 돼?
    loop = 0
    for time_element in time_ls:
        for person_element in person_ls:
            for tail1_element in tail1:
                for tail2_element in tail2:
                    for tail3_element in tail3:
                        sentence = [
                            f"/시간; {time_element}/인원; {person_element}/ {tail1_element}{tail2_element} {tail3_element}\n",
                            f"/인원; {person_element}/시간; {time_element}/ {tail1_element} {tail2_element} {tail3_element}\n"
                        ]
                        for sentence_element in sentence:
                            loop += 1
                            f.write(sentence_element)
    total += loop
    print(f"시간만 주어진 경우 문장 데이터 생성 완료, 만들어진 갯수: {loop}")
    print("예시)"+ sentence_element)

    # 2개의 슬롯만 주어진 경우 - 번호, 시간이 주어진 경우 ex) 10시부터 12시까지 3명이서 스터디룸 예약 돼?

    
    f.flush()
    print("\f"+os.getcwd()+"\\output.txt에 저장됐습니다. 만들어진 문장 개수:", total)