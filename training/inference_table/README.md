# Table Extraction

### Create virtual environment
`virtualenv env --python=3.8`

### Activate virtual environment
`source env/bin/activate`

### Install dependencies:
1. `pip install -r requirements.txt`
2. `pip install 'git+https://github.com/facebookresearch/detectron2.git'`


## Training

### Running train scripts
Trigger all the model training using the following bash script:
```
.train_table.sh
```


## Inference 

### Update config
Add the image and OCR files in `data/input` folder. And update the paths for these folders in `config.ini`.
Path to dump the result can also be specified in `config.ini` file.

Final result can be viewed at this location: `data/output/<dump_folder_name>/final/doc`


### Run inference script
```
python3 main.py
```