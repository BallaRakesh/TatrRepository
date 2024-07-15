import os
import json

actual_labels_path = '/home/gayathri/Downloads/CS/ANNOTATED_CS/ANNOTATED_VALIDATED_278/Labels'
prepared_data_path = '/home/gayathri/g3/extraction/GeoLayoutLM/dataset/custom_geo/preprocessed'

classes_path = '/home/gayathri/Downloads/CS/ANNOTATED_CS/ANNOTATED_VALIDATED_278/classes.txt'

with open(classes_path, 'r') as f:
    classes = f.readlines()

classes = [item.replace('\n', '').strip() for item in classes]
class_map = {i: item for i, item in enumerate(classes)}

print(class_map)

exit()

actual_labels_list = os.listdir(actual_labels_path)
prepared_data_list = os.listdir(prepared_data_path)


for labels in actual_labels_list:
    if labels.replace('txt', 'json') in prepared_data_list:
        # actual_classes = 
        with open(os.path.join(actual_labels_path, labels), 'r') as f:
            actual_labels = f.readlines()

        actual_labels = [class_map[int(item.split()[0])] for item in actual_labels]

        with open(os.path.join(prepared_data_path, labels.replace('txt', 'json')), 'r') as f:
            prepared_data = json.load(f)


        prepared_labels = [item for item in prepared_data['parse']['class'] 
                           if len(prepared_data['parse']['class'][item])!=0 and item!='O']
        
        missed_labels = list(set(actual_labels) - set(prepared_labels)) 
        extra_labels = list(set(prepared_labels) - set(actual_labels))
        print(missed_labels)
        print(extra_labels)

        exit()