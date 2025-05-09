import io
import math
import random
import os
import cv2
import lmdb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


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
                    img.verify()
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

class RatioDataSet(Dataset):

    def __init__(self, config, mode, logger, seed=None, epoch=1, task='rec'):
        super(RatioDataSet, self).__init__()
        self.ds_width = config[mode]['dataset'].get('ds_width', True)
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        max_ratio = loader_config.get('max_ratio', 10)
        min_ratio = loader_config.get('min_ratio', 1)
        syn = dataset_config.get('syn', False)
        if syn:
            data_dir_list = []
            data_dir = '../training_aug_lmdb_noerror/ep' + str(epoch)
            for dir_syn in os.listdir(data_dir):
                data_dir_list.append(data_dir + '/' + dir_syn)
        else:
            data_dir_list = dataset_config['data_dir_list']

        char_dict_path = dataset_config.get('transforms').get('GTCLabelEncode').get('character_dict_path')
        max_len = dataset_config.get('transforms').get('GTCLabelEncode').get('max_text_length')
        self.invalid_samples = filter_hierarchical_lmdb_datasets(data_dir_list, char_dict_path, max_len)


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
        for data_dir in data_dir_list:
            logger.info('Initialize indexs of datasets:%s' % data_dir)
        self.logger = logger
        self.data_idx_order_list = self.dataset_traversal()
        wh_ratio = np.around(np.array(self.get_wh_ratio()))
        self.wh_ratio = np.clip(wh_ratio, a_min=min_ratio, a_max=max_ratio)
        for i in range(max_ratio + 1):
            logger.info(f"Ratio {i}: {(1 * (self.wh_ratio == i)).sum()}")
        self.wh_ratio_sort = np.argsort(self.wh_ratio)
        self.ops = create_operators(dataset_config['transforms'],
                                    global_config)

        self.need_reset = True in [x < 1 for x in ratio_list]
        self.error = 0
        self.base_shape = dataset_config.get(
            'base_shape', [[64, 64], [96, 48], [112, 40], [128, 32]])
        self.base_h = 32

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
        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['ratio_num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
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
        h = img.shape[0]
        w = img.shape[1]
        if self.padding_rand and random.random() < 0.5:
            padding = not padding
        imgW, imgH = self.base_shape[gen_ratio - 1] if gen_ratio <= 4 else [
            self.base_h * gen_ratio, self.base_h
        ]
        use_ratio = imgW // imgH
        if use_ratio >= (w // h) + 2:
            self.error += 1
            return None
        if not padding:
            resized_image = cv2.resize(img, (imgW, imgH),
                                       interpolation=cv2.INTER_LINEAR)
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(
                    math.ceil(imgH * ratio * (random.random() + 0.5)))
                resized_w = min(imgW, resized_w)

            resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
        if self.padding_doub and random.random() < 0.5:
            padding_im[:, :, -resized_w:] = resized_image
        else:
            padding_im[:, :, :resized_w] = resized_image
        valid_ratio = min(1.0, float(resized_w / imgW))
        data['image'] = padding_im
        data['valid_ratio'] = valid_ratio
        data['real_ratio'] = round(w / h)
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
            self.lmdb_sets[lmdb_idx]['txn'], file_idx)
        if sample_info is None:
            ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
            ids = random.sample(ratio_ids, 1)
            return self.__getitem__([img_width, img_height, ids[0], ratio])
        img, label = sample_info
        data = {'image': img, 'label': label}
        outs = transform(data, self.ops[:-1])
        if outs is not None:
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
