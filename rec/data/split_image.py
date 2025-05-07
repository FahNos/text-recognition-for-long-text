import os
import math
import lmdb
from io import BytesIO
from PIL import Image


def split_and_resize_lmdb_images(data_dir_list, base_h=32, existing_splits_file=None):
    """
    Đọc ảnh từ các LMDB folders, với ảnh có gen_ratio > 4:
    - Resize về chiều cao = base_h, chiều rộng tính toán.
    - Chia nhỏ ảnh thành các cửa sổ con kích thước imgW x imgH, lưu ra thư mục cùng LMDB.

    Args:
        data_dir_list (list of str): Danh sách đường dẫn tới thư mục chứa LMDB.
        base_h (int): Chiều cao cơ sở cho ảnh con.
    """    
    existing_images = set()
    if existing_splits_file and os.path.isfile(existing_splits_file):
        print(f"Đọc danh sách ảnh đã tách từ file: {existing_splits_file}")
        with open(existing_splits_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 1:
                    # Lấy tên file cơ sở từ đường dẫn
                    img_path = parts[0]
                    base_filename = os.path.basename(img_path)
                    # Trích xuất ID gốc từ tên file (image-000000029_1.png -> image-000000029)
                    if '_' in base_filename:
                        base_id = base_filename.split('_')[0]
                        existing_images.add(base_id)
        print(f"Đã tìm thấy {len(existing_images)} ảnh gốc đã được xử lý")

    for lmdb_dir in data_dir_list:
        if not os.path.isdir(lmdb_dir):
            print(f"Bỏ qua, không phải thư mục LMDB: {lmdb_dir}")
            continue
        print(f"Mở LMDB: {lmdb_dir}")

        env = lmdb.open(
            lmdb_dir,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        with env.begin(write=False) as txn:
            num_samples = int(txn.get(b'num-samples'))
            print(f"Số mẫu: {num_samples}")

            out_dir = lmdb_dir
            os.makedirs(out_dir, exist_ok=True)
            label_path = os.path.join(out_dir, 'labels.txt')
            f_label = open(label_path, 'w', encoding='utf-8')
            processed = set()

            for idx in range(1, num_samples + 1):
                img_base_id = f'image-{idx:09d}'
                # Kiểm tra xem ảnh đã tồn tại trong danh sách đã xử lý hay chưa
                if img_base_id in existing_images:
                    print(f"Bỏ qua {img_base_id}, đã xử lý trước đó")
                    continue

                key = img_base_id.encode()
                imgbuf = txn.get(key)
                if imgbuf is None:
                    continue
                
                label_key = f'label-{idx:09d}'.encode()
                lb_data = txn.get(label_key)
                label = lb_data.decode('utf-8') if lb_data else ''

                try:
                    img = Image.open(BytesIO(imgbuf))
                except Exception as e:
                    print(f"Lỗi mở ảnh {key.decode()}: {e}")
                    continue

                w, h = img.size
                gen_ratio = round(w / float(h))
                
                if gen_ratio <= 4:
                    continue
                else:                    
                    gr = math.floor(gen_ratio)
                    if gr % 2 != 0:
                        gr += 1
                    imgW, imgH = base_h * gr, base_h
                                            
                if idx not in processed:
                    f_label.write(f"image-{idx:09d}.png\t{label}\n")
                    processed.add(idx)
                
                resized_w = imgW               

                resized = img.resize((resized_w, imgH), Image.BICUBIC)

                if resized.mode != 'RGB':
                    resized = resized.convert('RGB')
             
                num_windows = resized_w // (base_h * 2)
                if num_windows < 1:
                    continue

                out_dir = os.path.join(lmdb_dir, 'splits')
                os.makedirs(out_dir, exist_ok=True)

                base_name = os.path.splitext(f'image-{idx:09d}')[0]

                # out_name_0 = f"{base_name}_0.png"
                # out_path_0 = os.path.join(out_dir, out_name_0)

                # try:
                #     resized.save(out_path_0)
                # except Exception as e:
                #     print(f"Lỗi lưu {out_path_0}: {e}")

                for win in range(num_windows):
                    left = win * (base_h * 2)
                    right = left + (base_h * 2)
                    crop = resized.crop((left, 0, right, imgH))
                 
                    if crop.mode != 'RGB':
                        crop = crop.convert('RGB')
                    out_name = f"{base_name}_{win+1}.png"
                    out_path = os.path.join(out_dir, out_name)
                    try:
                        crop.save(out_path)
                    except Exception as e:
                        print(f"Lỗi lưu {out_path}: {e}")
                        
            f_label.close()
        env.close()
        print(f"Hoàn thành LMDB: {lmdb_dir}\n")


def main():
    data_dirs = [
        # '../Union14M-L-LMDB-Filtered/train_challenging',
        # # '../Union14M-L-LMDB-Filtered/train_hard',
        # # '../Union14M-L-LMDB-Filtered/train_medium',
        # # '../Union14M-L-LMDB-Filtered/train_normal',
        # # '../Union14M-L-LMDB-Filtered/train_easy',
        '../Union14M-L-LMDB-Filtered/train_vnese',
    ]
    # Đường dẫn tới file chứa danh sách ảnh đã tách
    existing_splits_file = 'processed_splits.txt'  # Thay đổi tên file theo nhu cầu
    
    split_and_resize_lmdb_images(data_dirs, base_h=32, existing_splits_file=existing_splits_file)


if __name__ == '__main__':
    main()
