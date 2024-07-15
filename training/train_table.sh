python3 table_detection/train.py &
python3 tatr/src/main.py --data_type structure --config_file tatr/src/structure_config.json --data_root_dir tatr/final_dataset_26_sep_23/trash_train/clean_labels --epochs=2 --batch_size 2 --model_save_dir tatr/trainlogs/mar/kr
