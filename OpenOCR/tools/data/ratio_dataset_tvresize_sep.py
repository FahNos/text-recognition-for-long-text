import io
import math
import random
import os

import cv2
import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
import json

from openrec.preprocess import create_operators, transform

def filter_lmdb_dataset(data_dir, verbose=True, char_dict=None, max_text_len = 25):
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
    invalid_samples = []

    # Process all samples
    with env_readonly.begin(write=False) as txn:
        # Check each sample
        for idx in range(1, num_samples + 1):
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
                    img_buf = io.BytesIO(image_bin)
                    img = Image.open(img_buf)
                    w, h = img.size
                    ratio = w / float(h)                    
                    img.verify()
                    if ratio > 30:
                        is_valid = False
                        reason = f"ratio too big > 30 - ratio = {ratio}"
                except Exception as e:
                    is_valid = False
                    reason = f"Invalid image: {str(e)}"

            if label_bin is None:
                is_valid = False
                reason = reason + "Label is None, "
            else:
                label = label_bin.decode()
                if len(label) > max_text_len:
                    is_valid = False
                    reason = reason + f"Label too long: {len(label)} chars, "

                label = label.strip()
                if len(label) == 0:
                    is_valid = False
                    reason = reason + f"Label len = 0, "
                if label == ' ':
                    is_valid = False
                    reason = reason + f"Label is space, "

                if char_dict and is_valid:
                    invalid_chars = []
                    for char in label:
                        if char not in char_dict:
                            invalid_chars.append(char)
                            is_valid = False
                            reason = reason + f"Invalid characters not in dictionary: {''.join(invalid_chars)}, "

            if not is_valid:
                invalid_samples.append({
                    'index': idx,
                    'image_key': image_key.decode('utf-8'),
                    'label_key': label_key.decode('utf-8'),
                    'wh_key': wh_key.decode('utf-8'),
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

def filter_hierarchical_lmdb_datasets(data_dir_list, char_dict_path, max_len):  

    if char_dict_path:
        try:
            with open(char_dict_path, 'r', encoding='utf-8') as f:
                content = f.read()
                chars = content.split()
                char_list = []
                for char in chars:
                    char_list.append(char)
                char_list = sorted(list(char_list))
        except Exception as e:
            print(f"Lỗi khi đọc file: {e}")

        print(f"Loaded dictionary with {len(char_list)} characters")

    char_list.append(' ')
    print(f'char list in filter_hierarchical_lmdb_datasets: {char_list}')

    char_dict = {}
    for i, char in enumerate(char_list):
        char_dict[char] = i


    total_valid = 0
    total_invalid = 0
    total_samples = 0
    all_invalid_samples = {}

    for data_dir in data_dir_list:
        print(f"\n{'-'*50}")
        print(f"Processing filter dataset: {data_dir}")
        valid, invalid, total, invalid_samples = filter_lmdb_dataset(data_dir, char_dict=char_dict, max_text_len = max_len)

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


class RatioDataSetTVResize_sep(Dataset):

    def __init__(self, config, mode, logger, seed=None, epoch=1, task='rec'):
        super(RatioDataSetTVResize_sep, self).__init__()
        
        self.mode = mode 
        self.ds_width = config[mode]['dataset'].get('ds_width', True)         

        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        max_ratio = loader_config.get('max_ratio', 10)
        min_ratio = loader_config.get('min_ratio', 1)
        data_dir_list = dataset_config['data_dir_list']


        if mode == 'Train':
            char_dict_path = global_config.get('character_dict_path')
            max_len = global_config.get('max_text_length')

            json_path = os.path.join('filtered_invalid_samples.json')

            # Nếu file json đã tồn tại, đọc từ đó
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.invalid_samples = json.load(f)
                print(f"Loaded invalid samples from {json_path}")
            else:           
                self.invalid_samples = filter_hierarchical_lmdb_datasets(data_dir_list, char_dict_path, max_len)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(self.invalid_samples, f, ensure_ascii=False, indent=2)
                print(f"Saved invalid samples to {json_path}")

        self.padding = dataset_config.get('padding', True)
        self.padding_rand = dataset_config.get('padding_rand', False)
        self.padding_doub = dataset_config.get('padding_doub', False)
        self.do_shuffle = loader_config['shuffle']
        self.seed = epoch
        data_source_num = len(data_dir_list)
        ratio_list = dataset_config.get('ratio_list', 1.0)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)
        assert (
            len(ratio_list) == data_source_num
        ), 'The length of ratio_list should be the same as the file_list.'
        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(
            data_dir_list, ratio_list)
        
        # Load rec_results.txt files  
        self.rec_results = {}
        if mode == 'Train':
            self.rec_results = self.load_rec_results(data_dir_list)

        for data_dir in data_dir_list:
            logger.info('Initialize indexs of datasets:%s' % data_dir)
        self.logger = logger
        self.data_idx_order_list = self.dataset_traversal()
        wh_ratio = np.around(np.array(self.get_wh_ratio()))
        self.wh_ratio = np.clip(wh_ratio, a_min=min_ratio, a_max=max_ratio)
        for i in range(max_ratio + 1):
            logger.info((1 * (self.wh_ratio == i)).sum())
        self.wh_ratio_sort = np.argsort(self.wh_ratio)
        self.ops = create_operators(dataset_config['transforms'],
                                    global_config)

        self.need_reset = True in [x < 1 for x in ratio_list]
        self.error = 0
        self.base_shape = dataset_config.get(
            'base_shape', [[64, 64], [96, 48], [112, 40], [128, 32]])
        self.base_h = dataset_config.get('base_h', 32)
        self.interpolation = T.InterpolationMode.BICUBIC
        transforms = []
        transforms.extend([
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        self.transforms = T.Compose(transforms)

    def load_rec_results(self, data_dir_list):        
        rec_results = {}
        for data_dir in data_dir_list:
            rec_results_path = os.path.join(data_dir, 'rec_results.txt')
            if os.path.exists(rec_results_path):
                with open(rec_results_path, 'r') as f:
                    lines = f.readlines()                
               
                dir_results = {}
                for line in lines:
                    parts = line.strip().split('\t')
                    if len(parts) < 2:
                        continue
                    
                    img_path = parts[0]
                    text = parts[1]
                    
                    # Extract image name and window index
                    img_name = os.path.basename(img_path)
                    if '_' in img_name:
                        base_name, window_idx = img_name.rsplit('_', 1)
                        base_name = base_name.split('.')[0]  # Remove file extension
                        window_idx = int(window_idx.split('.')[0])  # Get window index
                        
                        # Initialize dictionary for this image if it doesn't exist
                        if base_name not in dir_results:
                            dir_results[base_name] = {}
                        
                        # Store text for this window
                        dir_results[base_name][window_idx] = text
                
                rec_results[data_dir] = dir_results
            else:
                # self.logger.warning(f"rec_results.txt not found in {data_dir}")
                rec_results[data_dir] = {}
        
        return rec_results

    def get_wh_ratio(self):
        wh_ratio = []
        for idx in range(self.data_idx_order_list.shape[0]):
            lmdb_idx, file_idx = self.data_idx_order_list[idx]
            lmdb_idx = int(lmdb_idx)
            file_idx = int(file_idx)
            wh_key = 'wh-%09d'.encode() % file_idx
            wh = self.lmdb_sets[lmdb_idx]['txn'].get(wh_key)
            if wh is None:
                img_key = f'image-{file_idx:09d}'.encode()
                img = self.lmdb_sets[lmdb_idx]['txn'].get(img_key)
                buf = io.BytesIO(img)
                w, h = Image.open(buf).size
            else:
                wh = wh.decode('utf-8')
                w, h = wh.split('_')
            wh_ratio.append(float(w) / float(h))
        return wh_ratio

    def load_hierarchical_lmdb_dataset(self, data_dir_list, ratio_list):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, ratio in zip(data_dir_list, ratio_list):
            env = lmdb.open(dirpath,
                            max_readers=32,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            txn = env.begin(write=False)

            if self.mode == 'Train':
                invalid_indices = set()
                if dirpath in self.invalid_samples:
                    invalid_indices = {sample['index'] for sample in self.invalid_samples[dirpath]}
                

                num_samples = int(txn.get('num-samples'.encode()))
                valid_num_samples = num_samples - len(invalid_indices)

                lmdb_sets[dataset_idx] = {
                    'dirpath': dirpath,
                    'env': env,
                    'txn': txn,
                    'num_samples': num_samples,
                    'invalid_indices': invalid_indices,
                    'ratio_num_samples': int(ratio * valid_num_samples)
                }
                dataset_idx += 1
            else:
                num_samples = int(txn.get('num-samples'.encode()))
                lmdb_sets[dataset_idx] = {
                    'dirpath': dirpath,
                    'env': env,
                    'txn': txn,
                    'num_samples': num_samples,
                    'ratio_num_samples': int(ratio * num_samples)
                }
                dataset_idx += 1

        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['ratio_num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0

        if self.mode == 'Train':
            for lno in range(lmdb_num):
                tmp_sample_num = self.lmdb_sets[lno]['ratio_num_samples']

                invalid_indices = self.lmdb_sets[lno]['invalid_indices']
                valid_indices = [idx for idx in range(1, self.lmdb_sets[lno]['num_samples'] + 1)
                                if idx not in invalid_indices]
                if tmp_sample_num > len(valid_indices):
                    selected_indices = valid_indices
                else:
                    selected_indices = random.sample(valid_indices, tmp_sample_num)
                selected_indices = sorted(selected_indices)

                end_idx = beg_idx + len(selected_indices)
                data_idx_order_list[beg_idx:end_idx, 0] = lno
                data_idx_order_list[beg_idx:end_idx, 1] = selected_indices
                beg_idx = end_idx
        else:
            for lno in range(lmdb_num):
                tmp_sample_num = self.lmdb_sets[lno]['ratio_num_samples']
                end_idx = beg_idx + tmp_sample_num
                data_idx_order_list[beg_idx:end_idx, 0] = lno
                data_idx_order_list[beg_idx:end_idx, 1] = list(
                    random.sample(range(1, self.lmdb_sets[lno]['num_samples'] + 1),
                                self.lmdb_sets[lno]['ratio_num_samples']))
                beg_idx = beg_idx + tmp_sample_num

        return data_idx_order_list

    def get_img_data(self, value):
        """get_img_data."""
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype='uint8')
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            return None
        return imgori

    def resize_norm_img(self, data, gen_ratio, padding=True):
        img = data['image']
        w, h = img.size
        if self.padding_rand and random.random() < 0.5:
            padding = not padding


        # imgW, imgH = self.base_shape[gen_ratio - 1] if gen_ratio <= 4 else [
        #     self.base_h * gen_ratio, self.base_h
        # ]
        if gen_ratio <= 4:
            imgW, imgH = self.base_shape[gen_ratio - 1]
        else:
            gr = math.floor(gen_ratio)
            if gr % 2 != 0:
                gr += 1
            imgW, imgH = self.base_h * gr, self.base_h


        use_ratio = imgW // imgH # = 4
        if gen_ratio >= (w // h) + 2: # 4.9 + 2
            self.error += 1
            return None
        if not padding:
            resized_w = imgW
        else:
            ratio = w / float(h) # 5.9
            if math.ceil(imgH * ratio) > imgW:  # 188.8 > 192
                resized_w = imgW    # 128
            else:
                resized_w = int(
                    math.ceil(imgH * ratio * (random.random() + 0.5)))
                resized_w = min(imgW, resized_w)
        resized_image = F.resize(img, (imgH, resized_w),
                                 interpolation=self.interpolation)
        img = self.transforms(resized_image)
        if resized_w < imgW and padding:
            # img = F.pad(img, [0, 0, imgW-resized_w, 0], fill=0.)
            if self.padding_doub and random.random() < 0.5:
                img = F.pad(img, [0, 0, imgW - resized_w, 0], fill=0.)
            else:
                img = F.pad(img, [imgW - resized_w, 0, 0, 0], fill=0.)
        valid_ratio = min(1.0, float(resized_w / imgW))
        data['image'] = img
        data['valid_ratio'] = valid_ratio


        #* =========================================*#
        # Add the window-separated labels for images
        if gen_ratio <= 4:
            # For gen_ratio <= 4, keep the original behavior
            data['label_sep'] = data.get('ctc_label')
            data['length_sep'] = data.get('ctc_length')    
        else:
            if self.mode == 'Train' and len(self.rec_results) > 0:
                # For gen_ratio > 4, use the rec_results.txt file
                num_windows = use_ratio // 2
                lmdb_idx = int(data.get('lmdb_idx', -1))
                file_idx = int(data.get('file_idx', -1))             
                
                if lmdb_idx < 0 or file_idx < 0:
                    data['label_sep'] = None
                    data['length_sep'] = None
                    return None
                
                # Get the data directory for this sample
                data_dir = self.lmdb_sets[lmdb_idx]['dirpath']
                img_base_name = f"image-{file_idx:09d}"

                # print(f"--- data dir = {data_dir}")
                # print(f"    img_base_name = {img_base_name}")
                # print(f'    num windows = {num_windows}')            
                # print(f'    gen_ratio = {gen_ratio}')
                
                # Check if this image is in the rec_results file
                if data_dir not in self.rec_results or img_base_name not in self.rec_results[data_dir]:
                    data['label_sep'] = None
                    data['length_sep'] = None
                    return None
                
                # Check if the number of windows matches
                window_data = self.rec_results[data_dir][img_base_name]
                # print(f'window_data = {window_data}')
                if len(window_data) != num_windows:
                    data['label_sep'] = None
                    data['length_sep'] = None
                    # print("======================= debug 1: len(window_data) != num_windows")
                    return None
                
                # Check if any window has more than 16 characters
                if any(len(text) > 16 for text in window_data.values()):
                    data['label_sep'] = None
                    data['length_sep'] = None
                    # print("======================= debug 2: any(len(text) > 16 for text in window_data.values())")
                    return None
                
                # Everything is valid, create the label_sep and length_sep
                label_sep = []
                length_sep = []
                
                # Get ctc_label from data - this contains indices already
                ctc_label = data.get('ctc_label', [])
                if len(ctc_label) == 0:
                    data['label_sep'] = None
                    data['length_sep'] = None
                    # print("======================= debug 3: len(ctc_label) == 0")
                    return None
                    
                # Process each window
                char_index = 0
                for window_idx in range(1, num_windows + 1):
                    if window_idx not in window_data:
                        # Missing window
                        data['label_sep'] = None
                        data['length_sep'] = None
                        # print("======================= debug 4: window_idx not in window_data")
                        return None
                    
                    window_text = window_data[window_idx]
                    window_length = len(window_text)
                    
                    # Use char_index characters from ctc_label
                    if char_index + window_length > len(ctc_label):
                        # Not enough characters in ctc_label
                        data['label_sep'] = None
                        data['length_sep'] = None
                        # print("======================= debug 5: char_index + window_length > len(ctc_label)")
                        return None
                    
                    window_indices = ctc_label[char_index:char_index + window_length]
                    char_index += window_length
                    
                    # Pad to fixed_group_length (16)
                    fixed_group_length = 16
                    padded_indices = list(window_indices) + [0] * (fixed_group_length - len(window_indices))
                    
                    label_sep.append(padded_indices)
                    length_sep.append(window_length)
                
                data['label_sep'] = np.array(label_sep)
                data['length_sep'] = np.array(length_sep)
            else:               
                data['label_sep'] = data.get('ctc_label')
                data['length_sep'] = data.get('ctc_length')

            
            
            # print(f"    data[ctc_label] = {data.get('ctc_label')}")
            # print(f"    data[label_sep] = {data.get('label_sep')}")
            # print(f"    data[length_sep] = {data.get('length_sep')}") 
        
        if data['label_sep'] is None or data['length_sep'] is None:
            return None    
                    
        return data
    

    def get_lmdb_sample_info(self, txn, index):
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def __getitem__(self, properties):
        img_width = properties[0]
        img_height = properties[1]
        idx = properties[2]
        ratio = properties[3]
        lmdb_idx, file_idx = self.data_idx_order_list[idx]
        lmdb_idx = int(lmdb_idx)
        file_idx = int(file_idx)
        
        sample_info = self.get_lmdb_sample_info(
            self.lmdb_sets[lmdb_idx]['txn'], file_idx
        )
        
        if sample_info is None:
            ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
            ids = random.sample(ratio_ids, 1)
            return self.__getitem__([img_width, img_height, ids[0], ratio])
        img, label = sample_info        
      
        data = {'image': img, 'label': label, 'lmdb_idx': lmdb_idx, 'file_idx': file_idx}

        outs = transform(data, self.ops[:-1])
        if outs is not None:
            if 'lmdb_idx' not in outs:
                outs['lmdb_idx'] = lmdb_idx
            if 'file_idx' not in outs:
                outs['file_idx'] = file_idx

            outs = self.resize_norm_img(outs, ratio, padding=self.padding)
            if outs is None:
                ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
                ids = random.sample(ratio_ids, 1)
                return self.__getitem__([img_width, img_height, ids[0], ratio])
            outs = transform(outs, self.ops[-1:])
        if outs is None:
            ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
            ids = random.sample(ratio_ids, 1)
            return self.__getitem__([img_width, img_height, ids[0], ratio])
        return outs

    def __len__(self):
        return self.data_idx_order_list.shape[0]