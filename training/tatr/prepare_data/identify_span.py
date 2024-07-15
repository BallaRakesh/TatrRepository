import json
from utils import read_pascal_voc


"""
ToDo:
need to check if a string of words is a sentence or not
"""
def is_sentence(words):
    """
    checks if a set of strings is sentence or not
    """

def get_col_header_padding():
    """
    returns list of horizontal spaces between each header cell
    """
    pass


def get_data_cell_padding():
    """
    returns list of horizontal spaces between each data cell
    """
    pass

with open('../ocr_test_500/IM-000000009411510-AP1.json', 'r')  as f:
    data = json.load(f)
    
words = ' '.join([data[word]['text'] for word in data])
print(words)