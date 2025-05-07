import re
from collections import defaultdict

def process_files(rec_results_path, labels_path, mismatch_output_path, corrected_output_path):   
    original_labels = {}
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                # Lấy tên ảnh không có đuôi .png
                image_name = parts[0].replace('.png', '')
                original_labels[image_name] = parts[1]    
  
    image_parts = defaultdict(list)
    
    with open(rec_results_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:  
                path = parts[0]
                text = parts[1]
                confidence = float(parts[2])                 
                
                # lấy tên ảnh và số thứ tự
                match = re.search(r'(image-\d+)_(\d+)', path)
                if match:
                    base_image = match.group(1)
                    order_num = int(match.group(2))
                    image_parts[base_image].append((order_num, path, text, confidence))
    
    # Mở file để ghi kết quả
    mismatch_file = open(mismatch_output_path, 'w', encoding='utf-8')
    corrected_file = open(corrected_output_path, 'w', encoding='utf-8')
    
    # Xử lý từng ảnh gốc
    for base_image, parts in image_parts.items():
        skip_processing = False
        for score in image_parts[base_image]:        
            if score[3] < 0.15:
                skip_processing = True
                break
        if skip_processing:
            continue

        # Sắp xếp các phần theo số thứ tự
        parts_by_order = sorted(parts, key=lambda x: x[0])
        
        # Ghép text lại với nhau
        combined_text = ''.join(part[2] for part in parts_by_order)
        
        # Lấy text gốc từ file labels.txt
        original_text = original_labels.get(base_image, "")
        reject = ['%NA%', '%SC%', '-', 'I', '/', '....', '000','i', '%math%','...', '###', '####', 'None']

        if original_text in reject or len(original_text) < 4:       
            continue
        
        # So sánh text đã ghép với text gốc
        if combined_text != original_text and original_text:
            # Ghi vào file mismatches.txt
            for _, path, text, _ in parts_by_order:
                mismatch_file.write(f"{path}\t{text}\t{original_text}\n")
            
            # Thực hiện phân chia lại text
            corrected_parts = simple_text_division(parts_by_order, original_text)
            
            # Ghi kết quả được sửa vào file
            for order_num, path, original_pred, confidence in parts_by_order:
                corrected_text = corrected_parts.get(order_num, original_pred)
                corrected_file.write(f"{path}\t{corrected_text}\t{confidence}\t#####\t{original_text}\n")
                                
        else:
            # Nếu giống với text gốc, giữ nguyên
            for _, path, text, confidence in parts_by_order:
                corrected_file.write(f"{path}\t{text}\t{confidence}\n")
    
    mismatch_file.close()
    corrected_file.close()
    
    print(f"Đã hoàn thành! Kết quả được lưu trong {mismatch_output_path} và {corrected_output_path}")

def simple_text_division(parts, original_text):
  
    if not original_text:
        return {part[0]: part[2] for part in parts}
    
    # Sắp xếp theo thứ tự
    parts_by_order = sorted(parts, key=lambda x: x[0])
    num_parts = len(parts_by_order)    
    
    # Khởi tạo mảng đánh dấu vị trí
    char_owners = [None] * len(original_text)
    
    # Xác định hướng xử lý: từ đầu đến cuối hoặc từ cuối đến đầu
    first_part = parts_by_order[0]
    last_part = parts_by_order[-1]
    
    process_from_end = last_part[3] >= first_part[3]
    
    if process_from_end:      
        current_parts = list(reversed(parts_by_order))
    else:      
        current_parts = parts_by_order.copy()
    
    # Xử lý phần đầu tiên (hoặc phần cuối nếu process_from_end=True)
    first_part_to_process = current_parts[0]
    order_num = first_part_to_process[0]
    text = first_part_to_process[2]
    
    if process_from_end:      
        start_idx = max(0, len(original_text) - len(text))
        end_idx = len(original_text)
    else:       
        start_idx = 0
        end_idx = min(len(text), len(original_text))
    
    # Đánh dấu các vị trí thuộc về phần đầu tiên được xử lý
    for i in range(start_idx, end_idx):
        char_owners[i] = order_num
    
    # Xử lý các phần ở giữa (bỏ qua phần đầu tiên và phần cuối cùng)
    for part_idx in range(1, len(current_parts) - 1):
        current_part = current_parts[part_idx]
        order_num = current_part[0]
        text = current_part[2]
        
        if not text:  
            continue        
      
        match_found = False        
       
        for length in range(len(text), 0, -1):
            for start_pos in range(len(text) - length + 1):
                substring = text[start_pos:start_pos + length]                
              
                if process_from_end:
                    # Tìm từ cuối lên đầu, bỏ qua vị trí đã đăng ký
                    for i in range(len(original_text) - length, -1, -1):
                        if all(char_owners[j] is None for j in range(i, i + length)) and \
                           original_text[i:i + length] == substring:
                          
                            for j in range(i, i + length):
                                char_owners[j] = order_num                            
                         
                            if start_pos > 0:                               
                                prefix = text[:start_pos]
                                prefix_start = max(0, i - len(prefix))
                                for j in range(prefix_start, i):
                                    if char_owners[j] is None:
                                        char_owners[j] = order_num
                            
                            if start_pos + length < len(text):                               
                                suffix = text[start_pos + length:]
                                suffix_end = min(len(original_text), i + length + len(suffix))
                                for j in range(i + length, suffix_end):
                                    if char_owners[j] is None:
                                        char_owners[j] = order_num
                            
                            match_found = True
                            break
                else:
                    # Tìm từ đầu xuống cuối, bỏ qua vị trí đã đăng ký
                    for i in range(len(original_text) - length + 1):
                        if all(char_owners[j] is None for j in range(i, i + length)) and \
                           original_text[i:i + length] == substring:                          
                            for j in range(i, i + length):
                                char_owners[j] = order_num                            
                         
                            if start_pos > 0:                                
                                prefix = text[:start_pos]
                                prefix_start = max(0, i - len(prefix))
                                for j in range(prefix_start, i):
                                    if char_owners[j] is None:
                                        char_owners[j] = order_num
                            
                            if start_pos + length < len(text):                               
                                suffix = text[start_pos + length:]
                                suffix_end = min(len(original_text), i + length + len(suffix))
                                for j in range(i + length, suffix_end):
                                    if char_owners[j] is None:
                                        char_owners[j] = order_num
                            
                            match_found = True
                            break
                
                if match_found:
                    break
            
            if match_found:
                break
        
        # Nếu không tìm thấy trùng khớp, gán các vị trí trống liên tiếp
        if not match_found:
            if process_from_end:
                # Tìm vị trí trống gần nhất tính từ cuối lên
                empty_positions = [i for i in range(len(original_text)) if char_owners[i] is None]
                if empty_positions:
                    # Lấy vị trí trống xa nhất tính từ cuối
                    start_idx = empty_positions[0]
                    end_idx = min(start_idx + len(text), len(original_text))
                    
                    # Đánh dấu vị trí
                    for i in range(start_idx, end_idx):
                        char_owners[i] = order_num
            else:
                # Tìm vị trí trống gần nhất tính từ đầu xuống
                empty_positions = [i for i in range(len(original_text)) if char_owners[i] is None]
                if empty_positions:
                    # Lấy vị trí trống đầu tiên
                    start_idx = empty_positions[0]
                    end_idx = min(start_idx + len(text), len(original_text))
                    
                    # Đánh dấu vị trí
                    for i in range(start_idx, end_idx):
                        char_owners[i] = order_num
    
    # Xử lý phần còn lại (phần đầu tiên nếu process_from_end=True, phần cuối cùng nếu process_from_end=False)
    last_part_to_process = current_parts[-1]
    order_num = last_part_to_process[0]
    
    # Tìm vị trí còn trống nằm ở đầu hoặc cuối (tuỳ hướng xử lý)
    if process_from_end:  # Xử lý từ cuối lên đầu, nên phần còn lại là phần đầu
        # Tìm vị trí trống liên tục từ đầu
        start_empty = 0
        # while start_empty < len(char_owners) and char_owners[start_empty] is not None:
        #     start_empty += 1
            
        # Tìm vị trí đã được gán đầu tiên sau vị trí trống
        end_empty = start_empty
        while end_empty < len(char_owners) and char_owners[end_empty] is None:
            end_empty += 1
            
        # Gán tất cả vị trí trống này cho phần cuối
        for i in range(start_empty, end_empty):
            char_owners[i] = order_num
    else:  # Xử lý từ đầu xuống cuối, nên phần còn lại là phần cuối
        # Tìm vị trí trống liên tục từ cuối
        end_empty = len(char_owners) - 1
        # while end_empty >= 0 and char_owners[end_empty] is not None:
        #     end_empty -= 1
            
        # Tìm vị trí đã được gán đầu tiên trước vị trí trống
        start_empty = end_empty
        while start_empty >= 0 and char_owners[start_empty] is None:
            start_empty -= 1
        start_empty += 1  # Điều chỉnh lại vị trí bắt đầu
        
        # Gán tất cả vị trí trống này cho phần đầu
        for i in range(start_empty, end_empty + 1):
            char_owners[i] = order_num
    
    # Xử lý các vị trí còn trống
    empty_regions = []
    start_idx = None
    
    for i in range(len(char_owners)):
        if char_owners[i] is None and start_idx is None:
            start_idx = i
        elif (char_owners[i] is not None or i == len(char_owners) - 1) and start_idx is not None:
            end_idx = i if char_owners[i] is not None else i + 1
            empty_regions.append((start_idx, end_idx))
            start_idx = None
    
    # Phân phối các vùng trống cho các phần kề cạnh
    for start_idx, end_idx in empty_regions:
        # Tìm phần trước và phần sau vùng trống
        prev_part = None
        next_part = None
        
        if start_idx > 0:
            prev_part = char_owners[start_idx - 1]
        
        if end_idx < len(char_owners):
            next_part = char_owners[end_idx]
        
        if prev_part is not None and next_part is not None:
            # Có cả phần trước và phần sau
            empty_length = end_idx - start_idx
            
            # Đếm số ký tự đã gán cho mỗi phần
            prev_count = sum(1 for i in range(len(char_owners)) if char_owners[i] == prev_part)
            next_count = sum(1 for i in range(len(char_owners)) if char_owners[i] == next_part)
            
            # Chia đều hoặc ưu tiên phần có ít ký tự hơn
            if empty_length % 2 == 0:
                # Chia đều nếu số ký tự chẵn
                mid_point = start_idx + empty_length // 2
                for i in range(start_idx, mid_point):
                    char_owners[i] = prev_part
                for i in range(mid_point, end_idx):
                    char_owners[i] = next_part
            else:
                # Nếu số ký tự lẻ, ưu tiên phần có ít ký tự hơn
                mid_point = start_idx + empty_length // 2
                if prev_count <= next_count:
                    for i in range(start_idx, mid_point + 1):
                        char_owners[i] = prev_part
                    for i in range(mid_point + 1, end_idx):
                        char_owners[i] = next_part
                else:
                    for i in range(start_idx, mid_point):
                        char_owners[i] = prev_part
                    for i in range(mid_point, end_idx):
                        char_owners[i] = next_part
        elif prev_part is not None:
            # Chỉ có phần trước
            for i in range(start_idx, end_idx):
                char_owners[i] = prev_part
        elif next_part is not None:
            # Chỉ có phần sau
            for i in range(start_idx, end_idx):
                char_owners[i] = next_part
    
    # Tạo kết quả cuối cùng
    corrected_parts = {}
    
    for order_num, _, _, _ in parts:
        # Tìm tất cả vị trí thuộc về phần này
        positions = [i for i in range(len(char_owners)) if char_owners[i] == order_num]
        
        if positions:
            # Lấy text từ vị trí đầu đến vị trí cuối
            start = min(positions)
            end = max(positions) + 1
            corrected_parts[order_num] = original_text[start:end]
        else:
            # Nếu không có vị trí nào, giữ nguyên text gốc
            corrected_parts[order_num] = next((part[2] for part in parts if part[0] == order_num), "")
    
    return corrected_parts

# Đường dẫn đến các file
rec_results_path = r"C:\Users\has11\Desktop\content\label image sep\rec_results_train_vnese.txt"
labels_path = r"C:\Users\has11\Desktop\content\label image sep\labels_train_vnese.txt"
mismatch_output_path = r"C:\Users\has11\Desktop\content\label image sep\mismatches_train_vnese.txt"
corrected_output_path = r"C:\Users\has11\Desktop\content\label image sep\corrected_results_train_vnese.txt"

# Chạy hàm xử lý
process_files(rec_results_path, labels_path, mismatch_output_path, corrected_output_path)