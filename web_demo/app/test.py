import os
import sys
sys.path.append(
    os.path.dirname(os.path.abspath(
        os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import Bert_fine_tuning as bert

if __name__ == "__main__":
    bert.to_array("asd")
    print(os.getcwd())
    print(os.get)
