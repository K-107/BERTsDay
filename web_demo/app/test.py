import os
import sys
# 프로젝트 폴더를 import가 이루어질 경로에 추가 
"""
sys.path.append(
    os.path.dirname(os.path.abspath(
        os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
"""
import finetuned

if __name__ == "__main__":
    print(__file__)
    print(os.getcwd())
    print()
