# import os
# import pandas as pd
# import numpy as np
# import argparse

# def load_hierarchy(hierarchy_file, delimiter=';'):
#     hierarchy = {}
#     try:
#         with open(hierarchy_file, 'r', encoding='utf-8') as f:
#             lines = f.read().splitlines()
#             for line in lines:
#                 levels = line.split(delimiter)
#                 key = str(levels[0]).strip()  # Đảm bảo key là chuỗi và giữ nguyên định dạng
#                 hierarchy[key] = [str(level).strip() for level in levels]
#     except Exception as e:
#         raise ValueError(f"Không thể đọc tệp hierarchy '{hierarchy_file}': {str(e)}")
#     return hierarchy

# def load_hierarchy_from_csv_directory(directory, prefix=None):
#     hierarchies = {}
#     if not os.path.exists(directory):
#         raise FileNotFoundError(f"Thư mục hierarchies không tồn tại: {directory}")

#     if prefix is None:
#         prefixes = ['hierarchy_', 'italia_hierarchy_', 'adult_hierarchy_']
#     else:
#         prefixes = [prefix]

#     for filename in os.listdir(directory):
#         for prefix in prefixes:
#             if filename.startswith(prefix) and filename.endswith('.csv'):
#                 attr_name = filename.replace(prefix, '').replace('.csv', '')
#                 hierarchy_file = os.path.join(directory, filename)
#                 hierarchies[attr_name] = load_hierarchy(hierarchy_file)
#                 break
#     if not hierarchies:
#         raise ValueError(f"Không tìm thấy tệp hierarchy nào trong thư mục '{directory}' với tiền tố {prefixes}")
#     return hierarchies

# def get_qi_attributes(input_file, id_column=None, target_column=None, delimiter=';'):
#     try:
#         df = pd.read_csv(input_file, delimiter=delimiter, dtype=str)
#     except Exception as e:
#         raise ValueError(f"Không thể đọc tệp đầu vào '{input_file}': {str(e)}")

#     columns_to_drop = []
#     if id_column and id_column in df.columns:
#         columns_to_drop.append(id_column)
#     if target_column and target_column in df.columns:
#         columns_to_drop.append(target_column)
#     if columns_to_drop:
#         df = df.drop(columns=columns_to_drop)
#     return list(df.columns)

# def apply_generalization(value, hierarchy, level):
#     # Đảm bảo giá trị là chuỗi và giữ nguyên định dạng
#     value_str = str(value).strip()
#     if value_str not in hierarchy:
#         print(f"Giá trị '{value_str}' không có trong hierarchy, giữ nguyên.")
#         return value_str
#     levels = hierarchy[value_str]
#     if level >= len(levels):
#         print(f"Mức {level} vượt quá số mức trong hierarchy cho giá trị '{value_str}', sử dụng mức cao nhất: {len(levels)-1}")
#         level = len(levels) - 1
#     return levels[level]

# def anonymize_dataset(generalization_levels, hierarchies, input_file, output_file, delimiter=';'):
#     # Đọc dữ liệu từ input_file, ép tất cả cột thành chuỗi
#     try:
#         df = pd.read_csv(input_file, delimiter=delimiter, dtype=str)
#     except Exception as e:
#         raise ValueError(f"Không thể đọc tệp đầu vào '{input_file}': {str(e)}")

#     # Xử lý giá trị NaN
#     for col in df.columns:
#         df[col] = df[col].fillna('Unknown')

#     # Áp dụng tổng quát hóa cho từng cột
#     for attr, level in generalization_levels.items():
#         # Kiểm tra xem cột có tồn tại trong dữ liệu không
#         if attr not in df.columns:
#             print(f"Cảnh báo: Cột '{attr}' không tồn tại trong dữ liệu. Bỏ qua.")
#             continue
        
#         # Kiểm tra xem hierarchy có tồn tại cho cột không
#         if attr not in hierarchies:
#             print(f"Cảnh báo: Không tìm thấy hierarchy cho cột '{attr}'. Bỏ qua.")
#             continue
        
#         # Áp dụng tổng quát hóa nếu level > 0
#         if level > 0:
#             hierarchy = hierarchies[attr]
#             df[attr] = df[attr].apply(lambda x: apply_generalization(x, hierarchy, level))
#         else:
#             print(f"Không áp dụng tổng quát hóa cho cột '{attr}' (level={level}).")

#     # Tạo thư mục đầu ra nếu chưa tồn tại
#     output_dir = os.path.dirname(output_file)
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)

#     # Lưu dữ liệu đã ẩn danh vào output_file
#     try:
#         df.to_csv(output_file, sep=delimiter, index=False)
#         print(f"Dữ liệu đã ẩn danh được lưu vào '{output_file}'.")
#     except Exception as e:
#         raise ValueError(f"Không thể ghi vào tệp đầu ra '{output_file}': {str(e)}")




import os
import pandas as pd
import numpy as np
import argparse

def load_hierarchy(hierarchy_file, delimiter=';'):
    """
    Load hierarchy from a CSV file.

    Parameters:
    - hierarchy_file (str): Đường dẫn đến tệp hierarchy CSV.
    - delimiter (str): Ký tự phân tách trong tệp hierarchy, mặc định là ';'.

    Returns:
    - dict: Từ điển ánh xạ giá trị gốc sang danh sách các mức tổng quát hóa.
    """
    hierarchy = {}
    try:
        with open(hierarchy_file, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
            for line in lines:
                levels = line.split(delimiter)
                key = str(levels[0]).strip()  # Đảm bảo key là chuỗi và giữ nguyên định dạng
                hierarchy[key] = [str(level).strip() for level in levels]
    except Exception as e:
        raise ValueError(f"Không thể đọc tệp hierarchy '{hierarchy_file}': {str(e)}")
    return hierarchy

def load_hierarchy_from_csv_directory(directory, prefix=None):
    """
    Load all hierarchies from CSV files in the specified directory.

    Parameters:
    - directory (str): Thư mục chứa các tệp hierarchy.
    - prefix (str, optional): Tiền tố của tên tệp hierarchy (ví dụ: 'italia_hierarchy_').

    Returns:
    - dict: Từ điển ánh xạ tên thuộc tính (attribute) sang hierarchy.
    """
    hierarchies = {}
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Thư mục hierarchies không tồn tại: {directory}")

    if prefix is None:
        prefixes = ['hierarchy_', 'italia_hierarchy_', 'adult_hierarchy_']
    else:
        prefixes = [prefix]

    for filename in os.listdir(directory):
        for prefix in prefixes:
            if filename.startswith(prefix) and filename.endswith('.csv'):
                attr_name = filename.replace(prefix, '').replace('.csv', '')
                hierarchy_file = os.path.join(directory, filename)
                hierarchies[attr_name] = load_hierarchy(hierarchy_file)
                break
    if not hierarchies:
        raise ValueError(f"Không tìm thấy tệp hierarchy nào trong thư mục '{directory}' với tiền tố {prefixes}")
    return hierarchies

def get_qi_attributes(input_file, id_column=None, target_column=None, delimiter=';'):
    """
    Get quasi-identifier attributes from the input file.

    Parameters:
    - input_file (str): Đường dẫn đến tệp dữ liệu đầu vào.
    - id_column (str, optional): Tên cột định danh (ID) để loại bỏ.
    - target_column (str, optional): Tên cột mục tiêu (target) để loại bỏ.
    - delimiter (str): Ký tự phân tách trong tệp CSV, mặc định là ';'.

    Returns:
    - list: Danh sách các cột quasi-identifier.
    """
    try:
        df = pd.read_csv(input_file, delimiter=delimiter, dtype=str)
    except Exception as e:
        raise ValueError(f"Không thể đọc tệp đầu vào '{input_file}': {str(e)}")

    columns_to_drop = []
    if id_column and id_column in df.columns:
        columns_to_drop.append(id_column)
    if target_column and target_column in df.columns:
        columns_to_drop.append(target_column)
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    return list(df.columns)

def apply_generalization(value, hierarchy, level):
    """
    Áp dụng tổng quát hóa cho một giá trị dựa trên hierarchy và mức tổng quát hóa.

    Parameters:
    - value: Giá trị cần tổng quát hóa.
    - hierarchy (dict): Hierarchy của thuộc tính (từ load_hierarchy).
    - level (int): Mức tổng quát hóa.

    Returns:
    - str: Giá trị sau khi tổng quát hóa.
    """
    # Đảm bảo giá trị là chuỗi và giữ nguyên định dạng
    value_str = str(value).strip()
    # Đảm bảo các key trong hierarchy cũng được chuẩn hóa dưới dạng chuỗi
    hierarchy = {str(k).strip(): v for k, v in hierarchy.items()}
    if value_str not in hierarchy:
        print(f"Giá trị '{value_str}' không có trong hierarchy, giữ nguyên.")
        return value_str
    levels = hierarchy[value_str]
    if level >= len(levels):
        print(f"Mức {level} vượt quá số mức trong hierarchy cho giá trị '{value_str}', sử dụng mức cao nhất: {len(levels)-1}")
        level = len(levels) - 1
    return levels[level]

def anonymize_dataset(generalization_levels, hierarchies, input_file, output_file, delimiter=';'):
    """
    Ẩn danh dữ liệu từ input_file và lưu kết quả vào output_file dựa trên các mức tổng quát hóa.

    Parameters:
    - generalization_levels (dict): Từ điển ánh xạ thuộc tính với mức tổng quát hóa (ví dụ: {'age': 3, 'city_birth': 4}).
    - hierarchies (dict): Từ điển chứa hierarchy của các thuộc tính (tất cả các mức).
    - input_file (str): Đường dẫn đến tệp dữ liệu đầu vào.
    - output_file (str): Đường dẫn đến tệp đầu ra.
    - delimiter (str): Ký tự phân tách trong tệp CSV, mặc định là ';'.
    """
    # Đọc dữ liệu từ input_file, ép tất cả cột thành chuỗi
    try:
        df = pd.read_csv(input_file, delimiter=delimiter, dtype=str)
    except Exception as e:
        raise ValueError(f"Không thể đọc tệp đầu vào '{input_file}': {str(e)}")

    # Xử lý giá trị NaN
    for col in df.columns:
        df[col] = df[col].fillna('Unknown')

    # Áp dụng tổng quát hóa cho từng cột
    for attr, level in generalization_levels.items():
        # Kiểm tra xem cột có tồn tại trong dữ liệu không
        if attr not in df.columns:
            print(f"Cảnh báo: Cột '{attr}' không tồn tại trong dữ liệu. Bỏ qua.")
            continue
        
        # Kiểm tra xem hierarchy có tồn tại cho cột không
        if attr not in hierarchies:
            print(f"Cảnh báo: Không tìm thấy hierarchy cho cột '{attr}'. Bỏ qua.")
            continue
        
        # Áp dụng tổng quát hóa nếu level > 0
        if level > 0:
            hierarchy = hierarchies[attr]
            
            # Áp dụng tổng quát hóa cho từng giá trị trong cột
            df[attr] = df[attr].apply(lambda x: apply_generalization(x, hierarchy, level))
        # else:
        #     print(f"Không áp dụng tổng quát hóa cho cột '{attr}' (level={level}).")

    # Tạo thư mục đầu ra nếu chưa tồn tại
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Lưu dữ liệu đã ẩn danh vào output_file
    try:
        df.to_csv(output_file, sep=delimiter, index=False)
        # print(f"Dữ liệu đã ẩn danh được lưu vào '{output_file}'.")
    except Exception as e:
        raise ValueError(f"Không thể ghi vào tệp đầu ra '{output_file}': {str(e)}")