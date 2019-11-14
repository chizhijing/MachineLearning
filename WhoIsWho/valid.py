# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:17:11 2019

@author: chizj
"""
from train import disambiguate_by_text_sim
from common import *

validate_pub_data = json.load(open(valid_pub_data_path, 'r', encoding='utf-8'))
validate_data = json.load(open(valid_row_data_path, 'r', encoding='utf-8'))
merge_data = {}
for author in validate_data:
    validate_data[author] = [validate_pub_data[paper_id] for paper_id in validate_data[author]]

res=disambiguate_by_text_sim(validate_data,0.15)
json.dump(res, open('result/disambiguate_by_text_sim2.json', 'w', encoding='utf-8'), indent=4)