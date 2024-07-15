import os 
import json
from PIL import Image

def contour_sort(a, b):
	if abs(a['y1'] - b['y1']) <= 15:
		return a['x1'] - b['x1']

	return a['y1'] - b['y1']

res_path = 'custom_result/custom_trial'
img_path = 'dataset/custom_geo/testing_data/images'
act_path = 'preprocess/custom/custom_data/data_in_funsd_format/testing_data/annotations/'

results = [item for item in os.listdir(res_path) if item.endswith('json')]

for res in results:
    with open(os.path.join(res_path, res), 'r') as f:
        data = json.load(f)

    act_res = res.replace('_tagging.json', '.json')
    with open(os.path.join(act_path, act_res), 'r') as f:
        actual_data = json.load(f)['form']

    actual_map = {item['label']: item['text'].lower() for item in actual_data if not item['label'] == "other"}

    #for item in actual_map:
    #    print(item, '  :::  ', actual_map[item])

    #break

    unique_keys = [item['pred_key'][2:] for item in data]

    fin = {}
    for item in data:
        if item['pred_key'][2:] != '':
            if item['pred_key'][2:] not in fin:
                fin[item['pred_key'][2:]] = item['text']
            else:
                fin[item['pred_key'][2:]] += f" {item['text']}"
           # print(item['pred_key'][2:],fin[item['pred_key'][2:]])

    #for item in fin:
    #    print(item, ' ::: ', fin[item])
    #print(unique_keys)
    
    #image_name = res.replace('_tagging.json','.png')
    comp = []

    for item in actual_map:
        complete_match = 0
        if item in fin:
            text = fin[item]
        else:
            text = ''

        if text == actual_map[item]:
            complete_match = 1
       comp.append({
            'kwy': item,
            'actual': actual_map[item],
            'pred': text,
            'complete_match': complete_match
           })

    #text = sorted(text, key=cmp_to_key(contour_sort))  

    break
