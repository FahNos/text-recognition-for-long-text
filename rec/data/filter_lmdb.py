import os
import lmdb
import cv2
import numpy as np
from tqdm import tqdm
import io
from PIL import Image
import argparse
import yaml

def filter_lmdb_dataset(data_dir, verbose=True):
    
    env_readonly = lmdb.open(data_dir, 
                    max_readers=32,
                    readonly=True,  
                    lock=False,                 
                    readahead=False,
                    meminit=False)
    
    # Start a read transaction to get total count
    with env_readonly.begin(write=False) as txn:
        num_samples = int(txn.get('num-samples'.encode(), b'0').decode())
        if verbose:
            print(f"Processing dataset: {data_dir}")
            print(f"Total samples before filtering: {num_samples}")
    
    valid_count = 0
    invalid_count = 0
    total_count = num_samples   
    keys_to_delete = []
    invalid_samples = []
    
    # Process all samples
    with env_readonly.begin(write=False) as txn:
        # Check each sample
        for idx in tqdm(range(1, num_samples + 1), 
                       desc="Validating samples", 
                       disable=not verbose):
            # Get keys for this sample
            image_key = f'image-{idx:09d}'.encode()
            label_key = f'label-{idx:09d}'.encode()
            wh_key = f'wh-{idx:09d}'.encode()
            
            # Get data
            image_bin = txn.get(image_key)
            label_bin = txn.get(label_key)
            
            is_valid = True
            reason = ""
            
            # Check if image exists
            if image_bin is None:
                is_valid = False
                reason = "Image is None, " 
            else:
                try:
                    # Convert binary data to image
                    img_buf = io.BytesIO(image_bin)
                    img = Image.open(img_buf)
                    img.verify()     
                except Exception as e:
                    is_valid = False
                    reason = f"Invalid image: {str(e)}"

            if label_bin is None:
                is_valid = False
                reason = reason + "Label is None, "
            else:
                label = label_bin.decode()
                if len(label) > 200:
                    is_valid = False
                    reason = reason + f"Label too long: {len(label)} chars, "
                if len(label) == 0:
                    is_valid = False
                    reason = reason + f"Label len = 0, "
                if label == ' ':
                    is_valid = False
                    reason = reason + f"Label is space, "   
                    
            if not is_valid:
                invalid_samples.append({
                    'index': idx,
                    'image_key': image_key,
                    'label_key': label_key,
                    'wh_key': wh_key,
                    'reason': reason.strip()
                })              
                invalid_count += 1
                if verbose:
                    print(f"Sample {idx} invalid: {reason}")
            else:
                valid_count += 1

    env_readonly.close()
    
    if verbose:
        print(f"\nDataset: {data_dir}")
        print(f"Total samples: {total_count}")
        print(f"Valid samples: {valid_count}")
        print(f"Invalid samples: {invalid_count}")
        print(f"Percentage invalid: {invalid_count/total_count*100:.2f}%")
    
    return valid_count, invalid_count, total_count, invalid_samples

def filter_hierarchical_lmdb_datasets(config_path): 
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dataset_config = config['Train']['dataset']
    data_dir_list = dataset_config['data_dir_list']

    total_valid = 0
    total_invalid = 0
    total_samples = 0
    all_invalid_samples = {}
    
    for data_dir in data_dir_list:
        print(f"\n{'-'*50}")
        print(f"Processing filter dataset: {data_dir}")
        valid, invalid, total, invalid_samples = filter_lmdb_dataset(data_dir)

        all_invalid_samples[data_dir] = invalid_samples        
        total_valid += valid
        total_invalid += invalid
        total_samples += total
    
    print(f"\n{'-'*50}")
    print("Summary:")
    print(f"Total datasets processed: {len(data_dir_list)}")
    print(f"Total samples: {total_samples}")
    print(f"Total valid samples: {total_valid}")
    print(f"Total invalid samples: {total_invalid}")
    print(f"Overall percentage invalid: {total_invalid/total_samples*100:.2f}%")

    return all_invalid_samples