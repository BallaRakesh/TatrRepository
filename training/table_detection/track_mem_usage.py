import argparse
from datetime import datetime
import psutil
import time as t
import GPUtil as GPU

# Initialize parser
#parser = argparse.ArgumentParser()
#parser.add_argument('-p','--pid', help="Process ID of the program that need sto be tracked")
#args = parser.parse_args()

f = open('memory_tracker.txt', 'a')

for i in range(100000000):
    time = datetime.now()
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    print("GPU Usage:")
    #GPUtil.showUtilization()
    gpu = GPU.getGPUs()[0]
    #for gpu in GPUs:
    #    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))

    text = f"Time: {time} |  CPU usage: {cpu}% |  Virtual Memory: {mem}%  | GPU: {round(gpu.memoryUtil*100, 2)}% | Total GPU Memory: {round(gpu.memoryTotal/1024)} GB\n"
    print(text)
    f.write(text)
    t.sleep(30)

f.close()



