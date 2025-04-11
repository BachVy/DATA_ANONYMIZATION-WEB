import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn import svm
import pickle
from anonymize import anonymize_dataset, load_hierarchy_from_csv_directory, load_hierarchy, get_qi_attributes
from apo_aro import APO_ARO
from fitness import fitness, check_k_anonymity
from split import split
import argparse

class SVM:
    def __init__(self):
        self.model = svm.SVC(decision_function_shape="ovr")
        
    def fit(self, inputs, targets):
        self.model = self.model.fit(inputs, targets)

    def eval(self, inputs, targets, class_names=[]):
        if len(class_names) == 0:
            class_names = None
        preds = self.model.predict(inputs)
        return f1_score(targets, preds, labels=np.unique(preds), average='weighted')

    def save_model(self, path):
        pickle.dump(self.model, open(path, 'wb'))

    def load_model(self, path):
        self.model = pickle.load(open(path, 'rb'))

def one_hot_encoding(df):
    cat_attrs = []
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category' or len(df[col].unique()) < 0.1 * len(df):
            cat_attrs.append(col)
    if cat_attrs:
        df = pd.get_dummies(df, columns=cat_attrs)
    return df

def embed_target(targets):
    unique_labels = set(targets)
    label_to_idx = {v: i for i, v in enumerate(unique_labels)} 
    new_targets = [label_to_idx[i] for i in targets]
    return new_targets, label_to_idx

def replace_generalization(anon_df, original_df):
    cat_attrs = []
    num_attrs = []
    for col in original_df.columns:
        if (original_df[col].dtype == 'object' or 
            original_df[col].dtype.name == 'category' or 
            len(original_df[col].unique()) < 0.1 * len(original_df)):
            cat_attrs.append(col)
        else:
            num_attrs.append(col)

    def get_mean(value):
        if isinstance(value, str):
            tmp = value.split('~')
            if len(tmp) == 2:
                low, high = tmp
                mean = float(low) + (float(high) - float(low))/2
                return mean
            else:
                if '*' in value:
                    low = float(value.replace('*', '0'))
                    high = float(value.replace('*', '9'))
                    return low + (high - low) / 2
                elif not value.isnumeric():
                    return 0
        return float(value)

    anon_df_processed = anon_df.copy()
    if num_attrs:
        anon_df_numeric = anon_df[num_attrs].copy()
        for col in num_attrs:
            anon_df_processed[col] = anon_df_numeric[col].apply(get_mean)

    if cat_attrs:
        anon_df_cat = anon_df[cat_attrs].copy()
        anon_df_cat = pd.get_dummies(anon_df_cat, columns=cat_attrs)
        anon_df_processed = pd.concat([anon_df_processed.drop(columns=cat_attrs, errors='ignore'), anon_df_cat], axis=1)

    one_hot_original = one_hot_encoding(original_df.copy())
    missing_cols = [col for col in one_hot_original.columns if col not in anon_df_processed.columns]
    if missing_cols:
        missing_df = pd.DataFrame(0, index=anon_df_processed.index, columns=missing_cols)
        anon_df_processed = pd.concat([anon_df_processed, missing_df], axis=1)

    anon_df_processed = anon_df_processed[one_hot_original.columns]
    return anon_df_processed

def auto_select_target_column(df, target_column=None):
    if target_column and target_column in df.columns:
        return target_column

    common_target_names = ['target', 'label', 'class', 'disease', 'income', 'severity']
    columns_lower = [col.lower() for col in df.columns]

    for col in df.columns:
        if col.lower() in common_target_names:
            print(f"Tự động chọn cột target: '{col}' (dựa trên tên phổ biến).")
            return col

    selected_column = df.columns[-1]
    print(f"Không tìm thấy cột target phổ biến. Tự động chọn cột cuối: '{selected_column}'.")
    return selected_column

def auto_configure_params(input_file, df, id_column, target_column):
    dataset_name = os.path.basename(input_file).replace('.csv', '')
    dataset_dir = os.path.dirname(input_file)
    base_dir = os.path.dirname(dataset_dir)

    hierarchy_dir = os.path.join(dataset_dir, 'hierarchies')
    output_dir = os.path.join(base_dir.replace('data', 'results'), dataset_name)

    if id_column is None:
        columns_lower = [col.lower() for col in df.columns]
        for col in df.columns:
            if col.lower() in ['id', 'identifier']:
                id_column = col
                break
        if id_column is None:
            print("Không tìm thấy cột ID. Đặt id-column=None.")
            id_column = None

    target_column = auto_select_target_column(df, target_column)

    return hierarchy_dir, output_dir, id_column, target_column

def anonymize_data(input_file, k_anonymity, population_size=10, max_iter=500, k_share=1, id_column=None, target_column=None, hierarchy_prefix=None, ignore_missing_hierarchy=False):
    df = pd.read_csv(input_file, delimiter=';')

    hierarchy_dir, output_dir, id_column, target_column = auto_configure_params(
        input_file, df, id_column, target_column
    )

    if not os.path.exists(hierarchy_dir):
        raise FileNotFoundError(f"Thư mục hierarchies không tồn tại: {hierarchy_dir}. Vui lòng tạo thư mục và thêm các tệp hierarchy.")

    os.makedirs(output_dir, exist_ok=True)

    dataset_name = os.path.basename(input_file).replace('.csv', '')

    train_file = os.path.join(output_dir, f'{dataset_name}_train.txt')
    val_file = os.path.join(output_dir, f'{dataset_name}_val.txt')
    if not (os.path.exists(train_file) and os.path.exists(val_file)):
        args = argparse.Namespace(
            input=input_file,
            output=output_dir,
            train_ratio=0.8,
            seed=2020
        )
        split(args)

    with open(train_file, 'r') as f:
        train_indexes = [int(i) for i in f.read().splitlines()]
    with open(val_file, 'r') as f:
        val_indexes = [int(i) for i in f.read().splitlines()]

    if id_column and id_column in df.columns:
        df = df.drop([id_column], axis=1)
    if target_column not in df.columns:
        raise ValueError(f"Cột nhãn '{target_column}' không tồn tại trong dữ liệu.")
    targets = list(df[target_column])
    df = df.drop([target_column], axis=1)

    original_df = df.copy()
    targets_encoded, label_to_idx = embed_target(targets)
    original_df_encoded = one_hot_encoding(original_df)

    hierarchies = load_hierarchy_from_csv_directory(hierarchy_dir, prefix=hierarchy_prefix)
    
    if not hierarchies:
        raise ValueError(f"Không tìm thấy hierarchy nào trong thư mục '{hierarchy_dir}' với tiền tố '{hierarchy_prefix}'.")

    qi_attributes = get_qi_attributes(input_file, id_column=id_column, target_column=target_column, delimiter=';')
    print(f"Quasi-identifiers ban đầu: {qi_attributes}")

    qi_attributes = [attr for attr in qi_attributes if attr in hierarchies]
    if not qi_attributes:
        raise ValueError("Không có cột nào trong qi_attributes có hierarchy. Vui lòng kiểm tra thư mục hierarchies.")
    print(f"Quasi-identifiers sau khi lọc (chỉ giữ các cột có hierarchy): {qi_attributes}")

    dim = len(qi_attributes)
    print(f"Số chiều (dim): {dim}")

    lb = np.array([0] * dim)
    ub = []
    for attr in qi_attributes:
        max_level = max(len(levels) for levels in hierarchies[attr].values()) - 1
        if max_level <= 0:
            raise ValueError(f"Hierarchy cho thuộc tính '{attr}' không hợp lệ: max_level={max_level}. Phải có ít nhất 1 mức tổng quát hóa.")
        ub.append(max_level)
    ub = np.array(ub)
    print(f"ub: {ub}")
    print(f"lb: {lb}")

    if len(ub) != dim:
        raise ValueError(f"Kích thước của ub ({len(ub)}) không khớp với số chiều (dim={dim}).")

    if any(u <= 0 for u in ub):
        raise ValueError(f"Giới hạn trên (ub) không hợp lệ: {ub}. Kiểm tra hierarchies để đảm bảo có ít nhất 1 mức tổng quát hóa cho mỗi thuộc tính.")

    zero_levels = np.zeros(dim)
    satisfies_k = check_k_anonymity(zero_levels, hierarchies, input_file, k_anonymity, load_hierarchy, qi_attributes)
    print(f"Dữ liệu gốc có thỏa mãn {k_anonymity}-anonymity không? {satisfies_k}")

    for attr in qi_attributes:
        if attr in df.columns:
            unique_values = df[attr].astype(str).unique()
            hierarchy_values = set()
            for value, levels in hierarchies[attr].items():
                hierarchy_values.add(str(value).strip())
            missing_values = [val for val in unique_values if val not in hierarchy_values]
            if missing_values:
                print(f"Cảnh báo: Các giá trị sau trong cột '{attr}' không có trong hierarchy: {missing_values}")

    fitness_func_args = (hierarchies, input_file, k_anonymity, load_hierarchy, ub, target_column, id_column, qi_attributes)
    apo_aro = APO_ARO(
        population_size=population_size,
        dim=dim,
        max_iter=max_iter,
        lb=lb,
        ub=ub,
        fitness_func=lambda x: fitness(x, *fitness_func_args),
        k_share=k_share
    )

    min_iter = 20 
    num_iterations = max_iter // k_share  
    for i in range(num_iterations):
        apo_aro.iterate()
        current_best_solution = apo_aro.best_solution
        current_best_fitness = apo_aro.best_fitness

        current_total_iter = (i + 1) * k_share
        if current_total_iter >= min_iter:
            if current_best_solution is None:
                print("Giải pháp hiện tại là None, tiếp tục chạy...")
                continue
            satisfies_k = check_k_anonymity(current_best_solution, hierarchies, input_file, k_anonymity, load_hierarchy, qi_attributes)
            if satisfies_k:
                print(f"Đã tìm thấy giải pháp thỏa mãn {k_anonymity}-anonymity sau {current_total_iter} vòng lặp. Dừng thuật toán.")
                break

    best_solution = apo_aro.best_solution
    best_fitness = apo_aro.best_fitness

    if best_solution is None:
        raise ValueError("Không tìm thấy giải pháp hợp lệ sau khi chạy thuật toán APO_ARO.")

    print(f"Best solution found by APO_ARO: {best_solution}")
    print(f"Best fitness: {best_fitness}")

    fitness_value, il, m, ca_anon = fitness(best_solution, hierarchies, input_file, k_anonymity, load_hierarchy, ub, target_column, id_column, qi_attributes)

    generalization_levels = {qi_attributes[i]: int(best_solution[i]) for i in range(dim)}
    anon_output_file = os.path.join(output_dir, f'{dataset_name}_anonymized.csv')
    anonymize_dataset(generalization_levels, hierarchies, input_file, anon_output_file, delimiter=';')

    anon_df = pd.read_csv(anon_output_file, delimiter=';')
    anon_df = anon_df.drop([target_column], axis=1, errors='ignore')
    anon_df_encoded = replace_generalization(anon_df, original_df)
    anon_df_encoded = anon_df_encoded[original_df_encoded.columns]

    train_inputs = original_df_encoded.iloc[train_indexes]
    train_targets = [targets_encoded[i] for i in train_indexes]
    val_inputs = original_df_encoded.iloc[val_indexes]
    val_targets = [targets_encoded[i] for i in val_indexes]

    anon_train_inputs = anon_df_encoded.iloc[train_indexes]
    anon_val_inputs = anon_df_encoded.iloc[val_indexes]

    # Huấn luyện và đánh giá trên dữ liệu gốc
    svm_model = SVM()
    svm_model.fit(train_inputs, train_targets)
    ca_original = svm_model.eval(val_inputs, val_targets)

    results = {
        'il': il, 
        'ca_original': ca_original,
        'ca_anon': ca_anon, 
        'generalization_levels': generalization_levels, 
        'best_fitness': best_fitness, 
        'target_column': target_column,
        'm': m 
    }

    return anon_output_file, results