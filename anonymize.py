import os
import pandas as pd
import numpy as np

def load_hierarchy(hierarchy_file, delimiter=';'):
    hierarchy = {}
    try:
        with open(hierarchy_file, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
            for line in lines:
                levels = line.split(delimiter)
                key = str(levels[0]).strip() 
                hierarchy[key] = [str(level).strip() for level in levels]
    except Exception as e:
        raise ValueError(f"Không thể đọc tệp hierarchy '{hierarchy_file}': {str(e)}")
    return hierarchy

def load_hierarchy_from_csv_directory(directory, prefix=None):
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
    value_str = str(value).strip()
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
    try:
        df = pd.read_csv(input_file, delimiter=delimiter, dtype=str)
    except Exception as e:
        raise ValueError(f"Không thể đọc tệp đầu vào '{input_file}': {str(e)}")

    for col in df.columns:
        df[col] = df[col].fillna('Unknown')

    for attr, level in generalization_levels.items():
        if attr not in df.columns:
            print(f"Cảnh báo: Cột '{attr}' không tồn tại trong dữ liệu. Bỏ qua.")
            continue
        if attr not in hierarchies:
            print(f"Cảnh báo: Không tìm thấy hierarchy cho cột '{attr}'. Bỏ qua.")
            continue
        
        if level > 0:
            hierarchy = hierarchies[attr]
            
            df[attr] = df[attr].apply(lambda x: apply_generalization(x, hierarchy, level))
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        df.to_csv(output_file, sep=delimiter, index=False)
    except Exception as e:
        raise ValueError(f"Không thể ghi vào tệp đầu ra '{output_file}': {str(e)}")