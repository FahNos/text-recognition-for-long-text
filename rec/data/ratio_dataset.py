import os
import lmdb
import cv2
import numpy as np
from tqdm import tqdm
import io
from PIL import Image
import argparse
import yaml
import tensorflow as tf
import numpy as np
import random
import math

from data.aug_image import ImageAugmentation
from data.smtr_label_encoder import SMTRLabelEncoder

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

def filter_hierarchical_lmdb_datasets(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dataset_config = config['Train']['dataset']
    data_dir_list = dataset_config['data_dir_list']
    smtr_config = dataset_config.get('transforms').get('SMTRLabelEncode')
    max_len = smtr_config.get('max_text_length')
    print(f'max text length = {max_len}')

    char_dict_path = config.get('Global', {}).get('character_dict_path', None)

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

class ratioDataset(tf.keras.utils.Sequence):
    def __init__(self, config, mode='Train', seed=None, epoch=1, invalid_samples=None):
        super().__init__()
        self.is_training = True if mode == 'Train' else False

        self.max_len = config.get('Global', {}).get('max_text_length', 200)
        self.invalid_samples = invalid_samples or {}

        dataset_config = config[mode]['dataset']
        self.ds_width = dataset_config.get('ds_width', True)
        self.padding = dataset_config.get('padding', True)
        self.padding_rand = dataset_config.get('padding_rand', False)
        self.padding_doub = dataset_config.get('padding_doub', False)
        data_dir_list = dataset_config['data_dir_list']
        self.base_shape = dataset_config.get('base_shape', [[64, 64], [96, 48], [112, 40], [128, 32]])
        self.augment = dataset_config.get('augment_image', False)
        ratio_list = dataset_config.get('ratio_list', 1.0)

        sampler_config = config[mode]['sampler']
        self.scales = sampler_config.get('scales', [[128, 32]])
        self.divided_factor = sampler_config.get('divided_factor', [4, 16])

        loader_config = config[mode]['loader']
        self.do_shuffle = loader_config['shuffle']
        self.max_ratio = loader_config.get('max_ratio', 10)
        self.min_ratio = loader_config.get('min_ratio', 1)
        self.batch_size = loader_config.get('batch_size_per_card', 128)

        self.seed = epoch
        data_source_num = len(data_dir_list)
        if isinstance(ratio_list, (float, int)):
            ratio_list = [float(ratio_list)] * int(data_source_num)
        assert len(ratio_list) == data_source_num, 'The length of ratio_list should be the same as the file_list.'

        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir_list, ratio_list)
        self.data_idx_order_list = self.dataset_traversal()

        # Get width-height ratios
        wh_ratio = np.around(np.array(self.get_wh_ratio()))
        self.wh_ratio = np.clip(wh_ratio, a_min=self.min_ratio, a_max=self.max_ratio)

        self.error = 0
        self.base_h = 32
        # Create batch sampler
        self.max_bs =  1024
        self.batch_sampler = self.create_batch_sampler()

        smtr_config = dataset_config.get('transforms').get('SMTRLabelEncode')
        self.dict_path = smtr_config.get('character_dict_path', './dict.txt')
        if self.is_training:
            if os.path.exists(self.dict_path):
                print(f"File từ điển '{self.dict_path}' đã tồn tại, không cần tạo mới.")
                self.char_list = self.read_dict_file(self.dict_path)
            else:
                print(f"File từ điển '{self.dict_path}' không tồn tại, đang tạo mới...")
                self.char_list = self.create_dict_file()
        else:
            self.char_list = self.read_dict_file(self.dict_path)
            if not self.char_list:
                 print(f"Cảnh báo: Không thể đọc file từ điển '{self.dict_path}' ở chế độ Eval.")
        
        self.label_encoder = SMTRLabelEncoder(
            dict_character=self.char_list,
            use_space_char=smtr_config.get('use_space_char'),
            max_text_length=smtr_config.get('max_text_length'),
            sub_str_len=smtr_config.get('sub_str_len')
        )

    def read_dict_file(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                chars = content.split()
                char_list = []
                for char in chars:
                    char_list.append(char)

                char_list = sorted(list(char_list))
                return char_list
        except Exception as e:
            print(f"Lỗi khi đọc file: {e}")
            return []

    def create_dict_file(self):
        characters = set()
        print("Extracting labels from LMDB datasets...")

        try:
            # Iterate through all LMDB datasets
            for lmdb_idx in self.lmdb_sets:
                txn = self.lmdb_sets[lmdb_idx]['txn']
                num_samples = self.lmdb_sets[lmdb_idx]['num_samples']

                for file_idx in tqdm(range(1, num_samples + 1),
                                          desc=f'Processing LMDB {lmdb_idx+1}/{len(self.lmdb_sets)}',):
                                          # disable=True):
                    label_key = f'label-{file_idx:09d}'.encode()
                    label = txn.get(label_key)

                    if label is not None:
                        label = label.decode('utf-8')
                        # Update character set and max length
                        for char in label:
                            if not (0xE000 <= ord(char) <= 0xF8FF) and char != ' ' and not (0xF0000 <= ord(char) <= 0xFFFFD) and not (0x100000 <= ord(char) <= 0x10FFFD):
                                characters.add(char)

            # Sort characters for consistent output
            char_list = sorted(list(characters))

            # Write characters to dictionary file
            if self.is_training:
                current_dir = os.getcwd()
                file_path = os.path.join(current_dir, 'dict.txt')

                with open(file_path, 'w', encoding='utf-8') as f:
                        for idx, char in enumerate(char_list[:-1]):
                            f.write(f"{char}\n")
                        f.write(char_list[-1])

                print(f"Unique characters train dataset: {len(characters)}")
                return char_list

        except Exception as e:
            print(f"Error creating dictionary: {e}")
            return None

    def load_hierarchical_lmdb_dataset(self, data_dir_list, ratio_list):
        lmdb_sets = {}
        dataset_idx = 0
        for dirpath, ratio in zip(data_dir_list, ratio_list):

            if not os.path.exists(dirpath):
                print(f"Thư mục không tồn tại: {dirpath}")
                continue

            if not os.path.isdir(dirpath):
                print(f"Không phải là thư mục LMDB: {dirpath}")
                continue

            try:
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
            except Exception as e:
                print(f"Lỗi khi mở LMDB {dirpath}: {e}")

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

        # (total_sample_num, 2) is (lmdb_idx, file_idx), lmdb_idx is the index of lmdb_sets, file_idx is the index of lmdb,
        # e.g. data_idx_order_list = [[0, 5], [0, 2], [1, 9], [1, 1], [2, 15], [2, 3]]

        return data_idx_order_list

    def get_wh_ratio(self):
        wh_ratio = []
        i = 0
        for idx in range(self.data_idx_order_list.shape[0]):
            lmdb_idx, file_idx = self.data_idx_order_list[idx]
            lmdb_idx = int(lmdb_idx)
            file_idx = int(file_idx)
            wh_key = f'wh-{file_idx:09d}'.encode()
            wh = self.lmdb_sets[lmdb_idx]['txn'].get(wh_key)
            if wh is None:
                img_key = f'image-{file_idx:09d}'.encode()
                img = self.lmdb_sets[lmdb_idx]['txn'].get(img_key)
                if img is None:
                    print(f'img is none at lmdb_idx: {lmdb_idx} and file_idx: {file_idx}, fail get w_h ratio')
                    i += 1
                    continue
                try:
                    buf = io.BytesIO(img)
                    w, h = Image.open(buf).size
                except Exception as e:
                    print(f"Error opening image for sample {file_idx} in LMDB {lmdb_idx}: {e}")
                    continue
            else:
                wh = wh.decode('utf-8')
                w, h = wh.split('_')
            ratio = round(float(w) / float(h), 2)
            wh_ratio.append(ratio)

        print(f'* Tổng số ảnh bị lỗi = {i}')
        return wh_ratio

    def get_img_data(self, value):
        assert type(value) is bytes and len(value) > 0, "invalid input 'img' in Decode Image"

        imgdata = np.frombuffer(value, dtype='uint8')
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            print('imgori is NONE')
            return None
        return imgori

    def get_lmdb_sample_info(self, txn, index):
        label_key = f'label-{index:09d}'.encode()
        label = txn.get(label_key)
        if label is None:
            print('Label in get_lmdb_sample_info return NONE')
            return None
        label = label.decode('utf-8')
        img_key = f'image-{index:09d}'.encode()
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def resize_norm_img(self, img, gen_ratio, padding=True):
        if img is None:
            print('img in resize_norm_img return NONE')
            return None

        h = img.shape[0]
        w = img.shape[1]

        if self.padding_rand and random.random() < 0.5:
            padding = not padding

        imgW, imgH = self.base_shape[int(gen_ratio - 1)] if gen_ratio <= 4 else [
                        self.base_h * gen_ratio, self.base_h
                        ]
        imgW = int(imgW)
        imgH = int(imgH)

        use_ratio = imgW // imgH
        if use_ratio >= (w // h) + 2:
            print('use_ratio >= (w // h) + 2 in resize_norm_img, return None')
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
                resized_w = int(math.ceil(imgH * ratio * (random.random() + 0.5)))
                resized_w = min(imgW, resized_w)

            resized_image = cv2.resize(img, (resized_w, imgH))

        resized_image = resized_image.astype('float32')
        resized_image = resized_image / 255.0
        resized_image -= 0.5
        resized_image /= 0.5

        padding_im = np.zeros((imgH, imgW, 3), dtype=np.float32)
        if self.padding_doub and random.random() < 0.5:
            padding_im[:, -resized_w:, :] = resized_image
        else:
            padding_im[:, :resized_w, :] = resized_image

        return padding_im

    def create_batch_sampler(self):
        if isinstance(self.scales[0], list):
            base_im_w = self.scales[0][0]
            base_im_h = self.scales[0][1]
        elif isinstance(self.scales[0], int):
            base_im_w = self.scales[0]  # 128
            base_im_h = self.scales[0]  # 32

        base_batch_size = self.batch_size
        base_elements = base_im_w * base_im_h * base_batch_size

        # Get unique ratios, danh sách ratio duy nhất, loại bỏ ratio bị trùng nhau
        ratio_unique = np.unique(self.wh_ratio).tolist()

        # Create batches for each ratio
        batch_list = []
        for ratio in ratio_unique:
            ratio_ids = np.where(self.wh_ratio == ratio)[0] # lấy chỉ số của các phần tử trong mảng self.wh_ratio có giá trị = ratio

            if self.do_shuffle:
                random.shuffle(ratio_ids)

            # Adjust batch size based on ratio
            if ratio < 5:
                batch_size_ratio = base_batch_size
            else:
                batch_size_ratio = min(
                    self.max_bs,
                    int(max(1, (base_elements / (base_im_h * ratio * base_im_h))))
                )
                print(f'batch_size_ratio = {batch_size_ratio}')
            num_samples = len(ratio_ids)
            num_batches = (num_samples + batch_size_ratio - 1) // batch_size_ratio

            for i in range(num_batches):
                start_idx = i * batch_size_ratio
                end_idx = min((i + 1) * batch_size_ratio, num_samples)
                batch_indices = ratio_ids[start_idx:end_idx].tolist()

                if self.is_training and len(batch_indices) < batch_size_ratio:
                    padding_needed = batch_size_ratio - len(batch_indices)
                    repeated_indices = (ratio_ids.tolist() * ((padding_needed // len(ratio_ids)) + 1))[:padding_needed]
                    batch_indices.extend(repeated_indices)
                batch = {
                    'width': int(ratio * base_im_h),
                    'height': int(base_im_h),
                    'indices': batch_indices,
                    'ratio': ratio
                }
                batch_list.append(batch)

        return batch_list

    def augment_image(self, image, hparams=None, transforms=None, magnitude=5, num_layers=3, prob=0.5):
        if hparams is None:
            hparams = {'rotate_deg': 30.0,
                       'shear_x_pct': 0.9,
                       'shear_y_pct': 0.2,
                       'translate_x_pct': 0.10,
                       'translate_y_pct': 0.30,
                       'translate_pct': 0.15,
                       'shear_pct': 0.2}

        if transforms is None:
            transforms = ['AutoContrast','Equalize','Invert','Rotate','PosterizeIncreasing','SolarizeIncreasing',
                        'SolarizeAdd','ColorIncreasing','ContrastIncreasing','BrightnessIncreasing',
                        'ShearX','ShearY','TranslateXRel','TranslateYRel','GaussianBlur']

        # Convertir NumPy array a PIL Image
        if isinstance(image, np.ndarray):
            # Si es una imagen en formato OpenCV (BGR), convertir a RGB
            if image.shape[2] == 3:
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image_pil = Image.fromarray(image)
        else:
            image_pil = image

        # Aplicar augmentaciones
        augmenter = ImageAugmentation()

        for _ in range(num_layers):
            op_name = random.choice(transforms)
            image_pil = augmenter.apply_transform(image_pil, op_name, magnitude, hparams, prob)

        # Convertir de vuelta a NumPy array en formato BGR para OpenCV
        image_np = np.array(image_pil)
        if image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        return image_np

    def __len__(self):
        return len(self.batch_sampler)

    def on_epoch_end(self):
        # Shuffle data at the end of each epoch
        if self.do_shuffle:
            random.seed(self.seed)
            self.seed += 1
            self.batch_sampler = self.create_batch_sampler()
            random.shuffle(self.batch_sampler)

    def __getitem__(self, idx):
        batch_info = self.batch_sampler[idx]
        indices = batch_info['indices']
        ratio = batch_info['ratio']

        batch_size = len(indices)

        if ratio <= 4:
            imgW, imgH = self.base_shape[int(ratio - 1)]
        else:
            imgW, imgH = self.base_h * ratio, self.base_h

        imgW = int(imgW)
        imgH = int(imgH)

        # Initialize batch arrays
        batch_images = np.zeros((batch_size, imgH, imgW, 3), dtype=np.float32)
        batch_labels = []
        batch_label_subs = []
        batch_label_next = []
        batch_length_subs = []
        batch_label_subs_pre = []
        batch_label_next_pre = []
        batch_length_subs_pre = []
        batch_lengths = []

        for i, sample_idx in enumerate(indices):
            lmdb_idx, file_idx = self.data_idx_order_list[sample_idx]
            lmdb_idx = int(lmdb_idx)
            file_idx = int(file_idx)

            sample_info = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'], file_idx)
            if sample_info is None:
                # If sample is invalid, try to get another sample with the same ratio
                ratio_ids = np.where(self.wh_ratio == ratio)[0]
                if len(ratio_ids) > 0:
                    alt_idx = random.choice(ratio_ids)
                    lmdb_idx, file_idx = self.data_idx_order_list[alt_idx]
                    lmdb_idx = int(lmdb_idx)
                    file_idx = int(file_idx)
                    sample_info = self.get_lmdb_sample_info(self.lmdb_sets[lmdb_idx]['txn'], file_idx)

                if sample_info is None:
                    print('************* INVALID DATA sample_info is None *************')
                    print(f'lmdb_idx: {lmdb_idx}, file_idx: {file_idx}')
                    # If still invalid, use zeros
                    batch_images[i] = np.zeros((imgH, imgW, 3), dtype=np.float32)
                    batch_labels.append(np.zeros((self.max_len + 2), dtype=np.int64))
                    batch_label_subs.append(np.zeros(((self.max_len * 2) + 2, 5), dtype=np.int64))
                    batch_label_next.append(np.zeros(((self.max_len * 2) + 2), dtype=np.int64))
                    batch_length_subs.append(0)
                    batch_label_subs_pre.append(np.zeros(((self.max_len * 2) + 2, 5), dtype=np.int64))
                    batch_label_next_pre.append(np.zeros(((self.max_len * 2) + 2), dtype=np.int64))
                    batch_length_subs_pre.append(0)
                    batch_lengths.append(0)
                    continue

            imgbuf, label = sample_info
            img = self.get_img_data(imgbuf)
            if self.augment and self.is_training:
                img = self.augment_image(img)

            if img is None:
                print('************* INVALID DATA img is None *************')
                print(f'lmdb_idx: {lmdb_idx}, file_idx: {file_idx}')

                batch_images[i] = np.zeros((imgH, imgW, 3), dtype=np.float32)
                batch_labels.append(np.zeros((self.max_len + 2), dtype=np.int32))
                batch_label_subs.append(np.zeros(((self.max_len * 2) + 2, 5), dtype=np.int32))
                batch_label_next.append(np.zeros(((self.max_len * 2) + 2), dtype=np.int32))
                batch_length_subs.append(0)
                batch_label_subs_pre.append(np.zeros(((self.max_len * 2) + 2, 5), dtype=np.int32))
                batch_label_next_pre.append(np.zeros(((self.max_len * 2) + 2), dtype=np.int32))
                batch_length_subs_pre.append(0)
                batch_lengths.append(0)
                continue

            processed_img = self.resize_norm_img(img, ratio, padding=self.padding)

            if processed_img is None:
                print('************* INVALID DATA processed_img is None *************')
                print(f'lmdb_idx: {lmdb_idx}, file_idx: {file_idx}')

                batch_images[i] = np.zeros((imgH, imgW, 3), dtype=np.float32)
                batch_labels.append(np.zeros((self.max_len + 2), dtype=np.int32))
                batch_label_subs.append(np.zeros(((self.max_len * 2) + 2, 5), dtype=np.int32))
                batch_label_next.append(np.zeros(((self.max_len * 2) + 2), dtype=np.int32))
                batch_length_subs.append(0)
                batch_label_subs_pre.append(np.zeros(((self.max_len * 2) + 2, 5), dtype=np.int32))
                batch_label_next_pre.append(np.zeros(((self.max_len * 2) + 2), dtype=np.int32))
                batch_length_subs_pre.append(0)
                batch_lengths.append(0)
                continue

            batch_images[i] = processed_img

            sub_str_len = 5
            label = label.strip()
            if len(label) > 200 :
                print('************* INVALID DATA len(label) > self.max_len *************')
                print(f'lmdb_idx: {lmdb_idx}, file_idx: {file_idx}')
            if len(label) == 0 :
                print('************* INVALID DATA len(label) = 0 *************')
                print(f'lmdb_idx: {lmdb_idx}, file_idx: {file_idx}')

            label_encode = self.label_encoder.encode(label)
            if label_encode is None:
                print('************* INVALID DATA label_encode is None *************')
                print(f'lmdb_idx: {lmdb_idx}, file_idx: {file_idx}')

                batch_images[i] = np.zeros((imgH, imgW, 3), dtype=np.float32)
                batch_labels.append(np.zeros((self.max_len + 2), dtype=np.int32))
                batch_label_subs.append(np.zeros(((self.max_len * 2) + 2, 5), dtype=np.int32))
                batch_label_next.append(np.zeros(((self.max_len * 2) + 2), dtype=np.int32))
                batch_length_subs.append(0)
                batch_label_subs_pre.append(np.zeros(((self.max_len * 2) + 2, 5), dtype=np.int32))
                batch_label_next_pre.append(np.zeros(((self.max_len * 2) + 2), dtype=np.int32))
                batch_length_subs_pre.append(0)
                batch_lengths.append(0)
                continue

            encoded_label, encoded_subs, encoded_next, length_subs, encoded_subs_pre, encoded_next_pre, length_subs_pre, length = label_encode

            batch_labels.append(encoded_label)
            batch_label_subs.append(encoded_subs)
            batch_label_next.append(encoded_next)
            batch_length_subs.append(length_subs)
            batch_label_subs_pre.append(encoded_subs_pre)
            batch_label_next_pre.append(encoded_next_pre)
            batch_length_subs_pre.append(length_subs_pre)
            batch_lengths.append(length)

        batch_labels = np.array(batch_labels)
        batch_label_subs = np.array(batch_label_subs)
        batch_label_next = np.array(batch_label_next)
        batch_length_subs = np.array(batch_length_subs)
        batch_label_subs_pre = np.array(batch_label_subs_pre)
        batch_label_next_pre = np.array(batch_label_next_pre)
        batch_length_subs_pre = np.array(batch_length_subs_pre)
        batch_lengths = np.array(batch_lengths)

        return (
            batch_images,
            {
                'label': batch_labels,
                'label_subs': batch_label_subs,
                'label_next': batch_label_next,
                'length_subs': batch_length_subs,
                'label_subs_pre': batch_label_subs_pre,
                'label_next_pre': batch_label_next_pre,
                'length_subs_pre': batch_length_subs_pre,
                'length': batch_lengths
            }
        )

# Helper function to create a TensorFlow dataset from our Sequence class
def create_tf_dataset(dataset, batch_size=None):
    """
    Create a tf.data.Dataset from our custom Sequence class
    """
    def generator():
        for i in range(len(dataset)):
            yield dataset[i]

    # Determine output types and shapes based on a sample
    sample_x, sample_y = dataset[0]

    # Define output types
    output_types = (
        tf.float32,  # Images
        {
            'label': tf.int64,
            'label_subs': tf.int64,
            'label_next': tf.int64,
            'length_subs': tf.int64,
            'label_subs_pre': tf.int64,
            'label_next_pre': tf.int64,
            'length_subs_pre': tf.int64,
            'length': tf.int64
        }
    )

    # Define output shapes
    max_len = dataset.max_len
    batch_size = len(dataset.batch_sampler[0]['indices'])

    output_signature = (
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),  # Images shape
        {
            'label': tf.TensorSpec(shape=(None, max_len + 2), dtype=tf.int64),  # +2 for BOS and EOS
            'label_subs': tf.TensorSpec(shape=(None, (max_len * 2) + 2, 5), dtype=tf.int64),
            'label_next': tf.TensorSpec(shape=(None, (max_len * 2) + 2), dtype=tf.int64),
            'length_subs': tf.TensorSpec(shape=(None,), dtype=tf.int64),
            'label_subs_pre': tf.TensorSpec(shape=(None, (max_len * 2) + 2, 5), dtype=tf.int64),
            'label_next_pre': tf.TensorSpec(shape=(None, (max_len * 2) + 2), dtype=tf.int64),
            'length_subs_pre': tf.TensorSpec(shape=(None,), dtype=tf.int64),
            'length': tf.TensorSpec(shape=(None,), dtype=tf.int64)
        }
    )

    # Create the dataset
    tf_dataset = tf.data.Dataset.from_generator(
        generator,
        # output_types=output_types,
        output_signature=output_signature
    )

    # Apply batching if batch_size is provided
    # if batch_size is not None:
    #     tf_dataset = tf_dataset.batch(batch_size)

    # Prefetch for better performance
    tf_dataset = tf_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return tf_dataset

def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

def create_datasets_from_config(config_path):
    print('START FILTER\n')
    config = load_config(config_path)
    if config is None:
        raise ValueError("Failed to load configuration")

    invalid_samples = filter_hierarchical_lmdb_datasets(config_path)
    print('\nFINISH FILTER')
    print(f"\n{'-'*100}\n{'-'*100}\n{'-'*100}")

    # Create datasets
    train_dataset = ratioDataset(config, mode='Train', invalid_samples = invalid_samples)
    val_dataset = ratioDataset(config, mode='Eval')

    # Create TensorFlow datasets
    train_tf_dataset = create_tf_dataset(train_dataset)
    val_tf_dataset = create_tf_dataset(val_dataset)

    return train_tf_dataset, val_tf_dataset, train_dataset, val_dataset