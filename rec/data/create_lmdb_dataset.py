import os
import lmdb
import cv2
from tqdm import tqdm
import numpy as np
import io
from PIL import Image
import json
import argparse

def get_datalist(data_path, max_len):
   
    train_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(),
                          desc=f'load data from {data_path}'):
            if line:  # Check if line is not empty
                try:
                    line = json.loads(line)  # Parse JSON string into a dictionary
                    img_path = line["filename"]
                    label = line["text"]
                    if len(label) > max_len:
                        continue
                    train_data.append([str(img_path), label])
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}, line: {line}") # print error and line to debug
                    continue #skip the line if there is json decode error                
                   
    return train_data


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(data_list, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for imagePath, label in tqdm(data_list,
                                 desc=f'make dataset, save to {outputPath}'):
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
            buf = io.BytesIO(imageBin)
            w, h = Image.open(buf).size
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        whKey = 'wh-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()
        cache[whKey] = (str(w) + '_' + str(h)).encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create LMDB dataset.')
    parser.add_argument('-l', nargs='+', required=True, help='List of label files (JSON)')
    parser.add_argument('-s', required=True, help='Root path to save LMDB dataset')
    parser.add_argument('-m', type=int, default=800, help='Maximum length of data list')

    args = parser.parse_args()

    label_file_list = args.l
    save_path_root = args.s
    max_len = args.m

    for data_list in label_file_list:      
        save_path = save_path_root
        os.makedirs(save_path, exist_ok=True)
        print(save_path)
        train_data_list = get_datalist(data_list, max_len)

        createDataset(train_data_list, save_path)
