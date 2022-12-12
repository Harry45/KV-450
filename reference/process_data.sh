#!/bin/bash/

cd RAW_DATA
python raw_data_to_data.py
python caculate_offset.py
python correction.py
python split_files.py
cp -r split_100000 ../DATA/split_100000