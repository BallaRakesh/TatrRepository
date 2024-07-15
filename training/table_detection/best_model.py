import json

with open('output/train_may_24_2023/metrics.json', 'r') as f:
    log = f.readlines()

#log = log.split('/')

#print(eval(log[0]))

#exit()

aps = []
for item in log:
    try:
        #print(item)
        #item = eval(item)
        #print(item.keys())
        if "NaN" in item:
            #print(item)
            item = item.replace("NaN", "0")
            #print(item)
            item = eval(item)
            aps.append((item["iteration"], item["bbox/AP"], item["bbox/AP50"], item["bbox/AP75"]))

        if len(aps) == 7:
            break
    except:
        #print(eval(item))
        pass

print(aps)

