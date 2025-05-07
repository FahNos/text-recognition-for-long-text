import os
import re
import argparse # Import thư viện argparse

def find_image_with_highest_suffix(filepath):
    """
    Đọc file txt, tìm ảnh gốc có số đuôi (_X) lớn nhất và giá trị lớn nhất đó.

    Args:
        filepath (str): Đường dẫn đến file txt cần đọc.

    Returns:
        tuple: (tên_ảnh_gốc, số_đuôi_lớn_nhất) hoặc (None, -1) nếu không tìm thấy
               hoặc có lỗi.
    """
    max_suffixes = {}  # Dictionary để lưu trữ số đuôi lớn nhất cho mỗi ảnh gốc
                       # Key: tên ảnh gốc (ví dụ: image-000000009)
                       # Value: số đuôi lớn nhất tìm thấy cho ảnh đó

    try:
        # Sử dụng encoding utf-8 để hỗ trợ nhiều loại ký tự
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: # Bỏ qua các dòng trống
                    continue

                parts = line.split('\t')
                if len(parts) < 1: # Kiểm tra xem dòng có ít nhất 1 phần không
                    print(f"Cảnh báo: Dòng {line_num} không hợp lệ (thiếu dữ liệu): {line}")
                    continue

                full_path = parts[0]
                # Lấy tên file từ đường dẫn đầy đủ
                filename = os.path.basename(full_path)

                # Sử dụng regular expression để trích xuất tên gốc và số đuôi
                # Pattern: (tên gốc)_ (số đuôi).đuôi_file
                match = re.match(r'^(.*?)_(\d+)(\.[^.]+)$', filename)

                if match:
                    base_name = match.group(1) # Phần tên gốc
                    suffix_str = match.group(2) # Phần số đuôi dạng chuỗi
                    try:
                        suffix_num = int(suffix_str) # Chuyển thành số nguyên

                        # Cập nhật số đuôi lớn nhất cho ảnh gốc này
                        current_max = max_suffixes.get(base_name, -1)
                        if suffix_num > current_max:
                            max_suffixes[base_name] = suffix_num

                    except ValueError:
                        print(f"Cảnh báo: Không thể chuyển đổi suffix thành số ở dòng {line_num}: {filename}")
                # else:
                    # Tùy chọn: Bỏ qua hoặc ghi log các file không khớp định dạng
                    # print(f"Thông tin: Bỏ qua file không khớp định dạng tên ở dòng {line_num}: {filename}")
                    # pass

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{filepath}'")
        return None, -1
    except IsADirectoryError:
         print(f"Lỗi: Đường dẫn '{filepath}' là một thư mục, không phải file.")
         return None, -1
    except PermissionError:
         print(f"Lỗi: Không có quyền đọc file '{filepath}'.")
         return None, -1
    except Exception as e:
        print(f"Lỗi không xác định khi đọc file '{filepath}': {e}")
        return None, -1

    # Sau khi đọc hết file, tìm ảnh gốc có giá trị đuôi lớn nhất tổng thể
    if not max_suffixes:
        print("Không tìm thấy ảnh nào có định dạng suffix hợp lệ trong file.")
        return None, -1

    overall_max_suffix = -1
    image_with_max_suffix = None

    # Duyệt qua dictionary để tìm giá trị lớn nhất
    for base_name, max_suffix in max_suffixes.items():
        if max_suffix > overall_max_suffix:
            overall_max_suffix = max_suffix
            image_with_max_suffix = base_name

    return image_with_max_suffix, overall_max_suffix

# --- Phần xử lý đối số dòng lệnh và thực thi chương trình ---
if __name__ == "__main__":
    # Tạo parser
    parser = argparse.ArgumentParser(description="Tìm ảnh có số đuôi (_X) lớn nhất từ file txt.")

    # Thêm đối số --path
    parser.add_argument(
        "--path",
        type=str,          # Kiểu dữ liệu là chuỗi (string)
        required=True,     # Bắt buộc phải cung cấp đối số này
        help="Đường dẫn đến file txt cần xử lý." # Mô tả trợ giúp
    )

    # Phân tích các đối số được cung cấp từ dòng lệnh
    args = parser.parse_args()

    # Lấy đường dẫn file từ đối số đã parse
    file_path = args.path

    print(f"Đang xử lý file: {file_path}")

    # Gọi hàm và nhận kết quả
    image_name, max_val = find_image_with_highest_suffix(file_path)

    # In kết quả
    if image_name is not None:
        print(f"\n--- Kết quả ---")
        print(f"Ảnh gốc có số đuôi lớn nhất là: '{image_name}'")
        print(f"Giá trị đuôi lớn nhất tìm thấy là: {max_val}")
    else:
        print("\nKhông thể xác định ảnh có đuôi lớn nhất hoặc đã xảy ra lỗi.")