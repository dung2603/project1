import re

input_file = r"C:\filemohinh\modelmoi\train_test_inputs\nyudepthv2_test_files_with_gt.txt"    # Thay 'input.txt' bằng tên file của bạn
output_file = r"C:\Users\buiti\Downloads\ZoeDepth-main\ZoeDepth-main\train_test_inputs\output.txt"  # Thay 'output.txt' bằng tên file bạn muốn lưu kết quả

with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
    for line in fin:
        parts = line.strip().split(' ')
        if len(parts) >= 3:
            # Thêm 'sync/' vào hai đường dẫn đầu tiên
            parts[0] = 'nyu2_test/' + parts[0].lstrip('/')
            parts[1] = 'nyu2_test/' + parts[1].lstrip('/')
            # Ghi lại dòng đã chỉnh sửa
            fout.write(' '.join(parts) + '\n')
        else:
            # Nếu dòng không đúng định dạng, ghi lại nguyên vẹn
            fout.write(line)