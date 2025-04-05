import os
import ssl
import urllib.request
from tqdm import tqdm
import yaml
from argparse import ArgumentParser
import shutil
from pathlib import Path
import json
from typing import List, Dict, Tuple
import kagglehub
import magic
import numpy as np
from crop_img import process_image_file

def get_dataset_info(cfg):
    download_urls = cfg.get("download_links", [])
    filenames = cfg.get("filenames", [])   
    check_validity = cfg.get("check_validity", True)
    return download_urls, filenames, check_validity

def urlretrieve(url, filename, chunk_size=1024 * 32, check_validity=True):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    ctx = ssl.create_default_context()
    if not check_validity:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, context=ctx) as response:
        with open(filename, "wb") as fh, tqdm(total=response.length, unit="B", unit_scale=True) as pbar:
            while chunk := response.read(chunk_size):
                fh.write(chunk)
                pbar.update(len(chunk))

def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def check_archive_file(file_path):
  """Kiểm tra xem file có phải là file nén (zip, rar, tar) hay không."""
  try:
    file_type = magic.from_file(file_path, mime=True)
    return file_type in ['application/zip', 'application/x-rar-compressed', 'application/x-tar']
  except FileNotFoundError:
    return False

def extract_and_remove(file_path, extract_to_dir):   
    if not os.path.exists(file_path):
        print(f"Tệp nén không tồn tại: {file_path}")
        return   
    os.makedirs(extract_to_dir, exist_ok=True)
  
    if file_path.endswith('.zip'):
        print(f"Đang giải nén ZIP {file_path} vào {extract_to_dir}...")
        shutil.unpack_archive(file_path, extract_to_dir)
    elif file_path.endswith('.tar') or file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
        print(f"Đang giải nén TAR {file_path} vào {extract_to_dir}...")
        import tarfile
        with tarfile.open(file_path) as tar:
            tar.extractall(path=extract_to_dir)
    else:
        print(f"Không hỗ trợ định dạng tệp nén: {file_path}")
        return

    print("Giải nén hoàn tất.")
   
    print(f"Đang xóa tệp nén {file_path}...")
    os.remove(file_path)
    print("Đã xóa tệp nén.")


def change_image_paths(input_file, output_file, old_path, new_path):
    """
    Thay đổi đường dẫn trong file
    """
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    image_path, text = parts
                    new_image_path = image_path.replace(old_path, new_path)
                    outfile.write(f"{new_image_path}\t{text}\n")
                else:  # Xử lý các dòng không có tab
                    continue
    except FileNotFoundError:
        print(f"File not found: {input_file}")

def download_kaggle_vietocr(extract_to_dir):
    try:
        
        print("Downloading VietOCR dataset from Kaggle...")
        cache_path = kagglehub.dataset_download("vulamnguyen/vietocr-dataset")
        destination_path = os.path.join(extract_to_dir, "vietocr-dataset")
       
        try:
            shutil.copytree(cache_path, destination_path)
            print(f"Dataset copied to: {destination_path}")
        except FileExistsError:
            print(f"Directory {destination_path} already exists. Skipping copy.")
        
        # Xóa cache
        shutil.rmtree(cache_path)
        
        #  Kiểm tra lại xem dữ liệu đã được sao chép thành công chưa
        if os.path.exists(destination_path):
            print("Dataset successfully downloaded and copied.")
        else:
            print("Error downloading or copying dataset.")
            
        # Xử lý các đường dẫn
        datasets = [
            {"name": "ink", "folder": "InkData_line_processed"},
            {"name": "en_00", "folder": "en_00"},
            {"name": "en01", "folder": "en_01"},
            {"name": "meta", "folder": "meta"},
            {"name": "random", "folder": "random"},
            {"name": "vi00", "folder": "vi_00"},
            {"name": "vi01", "folder": "vi_01"}
        ]
        
        for dataset in datasets:
            input_file = f"{destination_path}/VietOCR-Paddle/{dataset['folder']}/rec/rec_gt_train.txt"
            output_file = f"{destination_path}/VietOCR-Paddle/{dataset['folder']}/rec/rec_gt_train_{dataset['name']}.txt"
            old_path = "train"
            new_path = f"{destination_path}/VietOCR-Paddle/{dataset['folder']}/rec/train"
            
            change_image_paths(input_file, output_file, old_path, new_path)
            print(f"Đã ghi file mới vào: {output_file}")
            
    except ImportError:
        print("Không thể import kagglehub. Hãy cài đặt bằng lệnh 'pip install kagglehub'")
    except Exception as e:
        print(f"Lỗi khi tải VietOCR dataset: {e}")

def copy_file(extract_to_dir): 

    destination_folder = os.path.join(extract_to_dir, 'data/images')
    os.makedirs(destination_folder, exist_ok=True)    

    # Danh sách các thư mục cần copy
    source_folders = [
        f'{extract_to_dir}/vietnamese/test_image', 
        f'{extract_to_dir}/vietnamese/train_images',
        f'{extract_to_dir}/vietnamese/unseen_test_images',
        f'{extract_to_dir}/vietnamese/labels',  # 2000 images street views
        f"{extract_to_dir}/My data and handwritting ICDAR2021",    # 261 images (create by myself) + 687 images handwritting ICDAR2021
        f'{extract_to_dir}/Doc_Scan',  # 200 images scan document 
        ]

    # Copy toàn bộ ảnh từ các thư mục nguồn sang 'images'
    for folder in source_folders:
        if os.path.exists(folder):
            for file_name in os.listdir(folder):
                source_file = os.path.join(folder, file_name)
                destination_file = os.path.join(destination_folder, file_name)
                # Copy file nếu là file (tránh copy folder con nếu có)
                if os.path.isfile(source_file):
                    shutil.copy2(source_file, destination_file)

def make_crop_img(extract_to_dir):
    crop_img_folder = os.path.join(extract_to_dir, 'data/crop_images')
    os.makedirs(crop_img_folder, exist_ok=True)

    crop_label_path =  f'{extract_to_dir}/data/crop_label.txt'
    crop_label = open(crop_label_path,'w', encoding='utf8')
    labels_folder_path = f"{extract_to_dir}/data/images" # Label Folder    
    if not os.path.exists(labels_folder_path):
        print(f"Error: Labels folder not found: {labels_folder_path}")
    image_folder = f'{extract_to_dir}/data/images' # Images Folder

    txt_paths = [os.path.join(labels_folder_path, f) for f in os.listdir(labels_folder_path) if f.endswith('.txt')]

    for txt_path in txt_paths:
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            dt_boxes = []
            texts = []

            for line in lines:
                parts = line.strip().split(",")
                if len(parts) != 9:
                    continue

                try:
                    coords = [int(round(float(x))) for x in parts[:8]]
                    text = parts[8]
                    text = text.strip()

                    if '###' in text or text == ' ':
                        # print('invalid path : ', txt_path)
                        continue

                    pts = np.array([[coords[0], coords[1]],
                                [coords[2], coords[3]],
                                [coords[4], coords[5]],
                                [coords[6], coords[7]]], np.float32)

                    dt_boxes.append(pts)
                    texts.append(text)

                except (ValueError, IndexError) as e:
                    print(f"Error processing line in {txt_path}: {line}")
                    continue

            filename = os.path.basename(txt_path)

            img_number = os.path.splitext(os.path.basename(txt_path))[0].split('_')[1]
            if int(img_number) < 10000:
                img_name = f'im{int(img_number):04d}.jpg'
            else:
                img_name = f'im{int(img_number):05d}.jpg'
            image_file = os.path.join(image_folder, img_name) 
         
            if not process_image_file(image_file, dt_boxes, texts, crop_img_folder, crop_label):
                print(f"Skipping file due to errors: {image_file}")

        except Exception as e:
            print(f"Error processing {txt_path}: {str(e)}")
            continue

    crop_label.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', help='configuration file to use', default='download_config.yaml')
    args = parser.parse_args()
    
    config_path = args.config    

    
    # Load the YAML config
    cfg = load_yaml_config(config_path)
    
 
    if "root" in cfg:
        os.makedirs(cfg["root"], exist_ok=True)
        
    extract_to_dir = None
    if "extraction_paths" in cfg:
        os.makedirs(cfg["extraction_paths"], exist_ok=True)
        extract_to_dir = cfg["extraction_paths"]
    
    # Get dataset info and download files
    urls, filename_paths, check_validity = get_dataset_info(cfg)
    
    if len(urls) != len(filename_paths):
        print(f"Error: Number of URLs, filenames and extraction paths don't match")
        return    
   
    for i, (url, filename_path) in enumerate(zip(urls, filename_paths)):
        print(f"\n[{i+1}/{len(urls)}] Downloading {filename_path} from {url} ...")
        try:
            urlretrieve(url=url, filename=filename_path, check_validity=check_validity)
            print(f"Successfully downloaded {filename_path}") 
          
            is_archive = check_archive_file(filename_path)
            if is_archive:
                extract_and_remove(filename_path, extract_to_dir)
            
        except Exception as e:
            print(f"Error downloading {filename_path}: {e}")
    
    print("Downloads and extractions finished!")    
    if extract_to_dir is not None:
        print("Download Kaggle Vietocr")
        download_kaggle_vietocr(extract_to_dir)    
        print("\nAll dataset processing completed!")

        copy_file(extract_to_dir)
        print("\nCopy completed!")
        make_crop_img(extract_to_dir)
        print("\nCrop images completed!")

if __name__ == "__main__":
    main()