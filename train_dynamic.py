# import os
# import pandas as pd
# import numpy as np
# from fitness import fitness, check_k_anonymity
# from anonymize import anonymize_dataset, load_hierarchy
# from apo_aro import APO_ARO
# import subprocess  # Thêm để gọi split.py

# # Hàm mã hóa one-hot
# def one_hot_encoding(df):
#     return pd.get_dummies(df, drop_first=True)

# # Hàm nhúng nhãn
# def embed_target(targets):
#     label_to_idx = {label: idx for idx, label in enumerate(sorted(set(targets)))}
#     embedded_targets = [label_to_idx[label] for label in targets]
#     return embedded_targets, label_to_idx

# # Hàm thay thế dữ liệu tổng quát hóa
# def replace_generalization(anon_df, original_df):
#     return one_hot_encoding(anon_df)

# # Lớp SVM đơn giản
# class SVM:
#     def __init__(self):
#         from sklearn.svm import SVC
#         self.model = SVC(kernel='linear', probability=True)
    
#     def fit(self, X, y):
#         self.model.fit(X, y)
    
#     def eval(self, X, y, labels=None):
#         from sklearn.metrics import f1_score
#         y_pred = self.model.predict(X)
#         return f1_score(y, y_pred, average='weighted')

# # Hàm chọn cột target tự động
# def select_target_column(df, qi_attributes):
#     """
#     Chọn một cột làm target nếu người dùng không cung cấp.
#     Tiêu chí: Cột có số giá trị duy nhất từ 2 đến 20, ưu tiên cột không phải QI.
#     """
#     candidate_columns = [col for col in df.columns if col not in qi_attributes]
#     if not candidate_columns:
#         candidate_columns = df.columns.tolist()  # Nếu không phân biệt được QI, dùng tất cả cột
    
#     for col in candidate_columns:
#         unique_values = df[col].nunique()
#         if 2 <= unique_values <= 20:  # Tiêu chí phân loại hợp lý
#             print(f"Chọn tự động cột '{col}' làm target (số giá trị duy nhất: {unique_values}).")
#             return col
    
#     # Nếu không tìm thấy cột nào phù hợp, chọn cột cuối cùng
#     selected_col = df.columns[-1]
#     print(f"Không tìm thấy cột phù hợp, chọn cột cuối cùng '{selected_col}' làm target.")
#     return selected_col

# def anonymize_data(input_file, k_anonymity, target_column=None):
#     # Đọc dữ liệu
#     df = pd.read_csv(input_file, delimiter=';')
#     id_column = 'ID'  # Cột ID, nếu có, sẽ được loại bỏ khỏi QI
    
#     # Sao chép dữ liệu gốc
#     original_df = df.copy()

#     # Xử lý cột target
#     if target_column and target_column in df.columns:
#         print(f"Sử dụng cột '{target_column}' do người dùng cung cấp làm target.")
#     else:
#         # Nếu không có target_column hoặc không tồn tại, tự động chọn
#         qi_attributes = [col for col in df.columns]  # Ban đầu coi tất cả là QI
#         target_column = select_target_column(df, qi_attributes)
    
#     targets = list(df[target_column])
#     df = df.drop([target_column], axis=1)

#     # Loại bỏ cột ID nếu có
#     if id_column and id_column in df.columns:
#         print(f"Loại bỏ cột '{id_column}' khỏi danh sách các cột cần ẩn danh.")
#         df = df.drop([id_column], axis=1)

#     # Xác định các cột quasi-identifier và lọc các cột có file hierarchy
#     all_attributes = [col for col in df.columns]
#     hierarchies = {}
#     hierarchy_dir = os.path.join(os.path.dirname(input_file), 'hierarchies')
#     qi_attributes = []
    
#     for attr in all_attributes:
#         hierarchy_file = os.path.join(hierarchy_dir, f'hierarchy_{attr}.csv')
#         if os.path.exists(hierarchy_file):
#             hierarchies[attr] = pd.read_csv(hierarchy_file, delimiter=';').values.tolist()
#             qi_attributes.append(attr)
#         else:
#             print(f"Không tìm thấy file hierarchy cho thuộc tính '{attr}' ({hierarchy_file}), bỏ qua cột này.")

#     if not qi_attributes:
#         raise ValueError("Không có cột nào có file hierarchy để ẩn danh. Vui lòng kiểm tra thư mục hierarchies.")

#     dim = len(qi_attributes)
#     print(f"Các cột được chọn để ẩn danh (có file hierarchy): {qi_attributes}")

#     # Tính max_levels (ub) từ hierarchies
#     ub = np.array([len(hierarchy) - 1 for hierarchy in hierarchies.values()])

#     # Mã hóa dữ liệu gốc
#     original_df_encoded = one_hot_encoding(df)
#     targets_encoded, label_to_idx = embed_target(targets)

#     # Đọc chỉ số train và val
#     dataset_name = os.path.basename(input_file).replace('.csv', '')
#     output_dir = os.path.join(os.path.dirname(os.path.dirname(input_file)).replace('data', 'results'), dataset_name)
#     os.makedirs(output_dir, exist_ok=True)
#     train_file = os.path.join(output_dir, f'{dataset_name}_train.txt')
#     val_file = os.path.join(output_dir, f'{dataset_name}_val.txt')

#     # Nếu file train.txt hoặc val.txt không tồn tại, tự động chạy split.py
#     if not os.path.exists(train_file) or not os.path.exists(val_file):
#         print(f"Không tìm thấy file {train_file} hoặc {val_file}. Tự động chạy split.py để tạo file train/val...")
#         try:
#             subprocess.run([
#                 'python', 'split.py',
#                 '--input', input_file,
#                 '--output', output_dir,
#                 '--train_ratio', '0.8',
#                 '--seed', '2020'
#             ], check=True)
#             print(f"Đã tạo file {train_file} và {val_file} thành công.")
#         except subprocess.CalledProcessError as e:
#             raise RuntimeError(f"Lỗi khi chạy split.py: {e}")

#     # Kiểm tra lại sau khi chạy split.py
#     if not os.path.exists(train_file) or not os.path.exists(val_file):
#         raise FileNotFoundError(f"Không tìm thấy file train.txt hoặc val.txt trong {output_dir} sau khi chạy split.py")

#     with open(train_file, 'r') as f:
#         train_indexes = [int(i) for i in f.read().splitlines()]
#     with open(val_file, 'r') as f:
#         val_indexes = [int(i) for i in f.read().splitlines()]

#     # Kiểm tra k-anonymity trên dữ liệu gốc
#     zero_levels = np.zeros(dim)
#     satisfies_k = check_k_anonymity(zero_levels, hierarchies, input_file, k_anonymity, load_hierarchy, qi_attributes)
#     print(f"Dữ liệu gốc có thỏa mãn {k_anonymity}-anonymity không? {satisfies_k}")

#     # Nếu dữ liệu gốc không thỏa mãn, chạy thuật toán APO_ARO
#     if not satisfies_k:
#         apo_aro = APO_ARO(
#             population_size=50,
#             dim=dim,
#             max_iter=100,
#             lb=np.zeros(dim),
#             ub=ub,
#             fitness_func=lambda x: fitness(x, hierarchies, input_file, k_anonymity, load_hierarchy, ub, target_column, id_column, qi_attributes)
#         )
#         apo_aro.run()

#         # Lấy giải pháp tốt nhất sau khi chạy
#         best_solution = apo_aro.best_solution
#         best_fitness = apo_aro.best_fitness

#         if best_solution is None:
#             raise ValueError("Không tìm thấy giải pháp hợp lệ sau khi chạy thuật toán APO_ARO.")

#         print(f"Best solution found by APO_ARO: {best_solution}")
#         print(f"Best fitness: {best_fitness}")

#         # Ẩn danh dữ liệu với giải pháp tốt nhất
#         generalization_levels = {qi_attributes[i]: int(best_solution[i]) for i in range(dim)}
#         anon_output_file = os.path.join(output_dir, f'{dataset_name}_anonymized.csv')
#         anonymize_dataset(generalization_levels, hierarchies, input_file, anon_output_file, delimiter=';')

#         # Gọi lại hàm fitness để lấy đầy đủ các giá trị
#         f_x, il, m, ca_anon = fitness(best_solution, hierarchies, input_file, k_anonymity, load_hierarchy, ub, target_column, id_column, qi_attributes)

#         # Tính Classification Accuracy (CA) trên dữ liệu gốc
#         anon_df = pd.read_csv(anon_output_file, delimiter=';')
#         if target_column in anon_df.columns:
#             anon_df = anon_df.drop([target_column], axis=1)
#         anon_df_encoded = replace_generalization(anon_df, original_df)
#         anon_df_encoded = anon_df_encoded[original_df_encoded.columns]

#         # Chia dữ liệu thành tập train và test
#         train_inputs = original_df_encoded.iloc[train_indexes]
#         train_targets = [targets_encoded[i] for i in train_indexes]
#         val_inputs = original_df_encoded.iloc[val_indexes]
#         val_targets = [targets_encoded[i] for i in val_indexes]

#         # Huấn luyện và đánh giá trên dữ liệu gốc
#         svm_model = SVM()
#         svm_model.fit(train_inputs, train_targets)
#         ca_original = svm_model.eval(val_inputs, val_targets)

#         # Tạo dictionary kết quả
#         results = {
#             'il': il,
#             'ca_original': ca_original,
#             'ca_anon': ca_anon,
#             'm': m,
#             'generalization_levels': generalization_levels,
#             'best_fitness': f_x,
#             'target_column': target_column
#         }

#         return anon_output_file, results

#     else:
#         # Nếu dữ liệu gốc đã thỏa mãn k-anonymity
#         f_x, il, m, ca_anon = fitness(zero_levels, hierarchies, input_file, k_anonymity, load_hierarchy, ub, target_column, id_column, qi_attributes)

#         # Tính CA trên dữ liệu gốc
#         train_inputs = original_df_encoded.iloc[train_indexes]
#         train_targets = [targets_encoded[i] for i in train_indexes]
#         val_inputs = original_df_encoded.iloc[val_indexes]
#         val_targets = [targets_encoded[i] for i in val_indexes]

#         svm_model = SVM()
#         svm_model.fit(train_inputs, train_targets)
#         ca_original = svm_model.eval(val_inputs, val_targets)

#         results = {
#             'il': il,
#             'ca_original': ca_original,
#             'ca_anon': ca_anon,
#             'm': m,
#             'generalization_levels': {qi_attributes[i]: 0 for i in range(dim)},
#             'best_fitness': f_x,
#             'target_column': target_column
#         }

#         return input_file, results







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
    """
    Support Vector Machine Classifier
    """
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
    """
    Tự động chọn cột target nếu người dùng không cung cấp hoặc cột không tồn tại.

    Parameters:
    - df (pandas.DataFrame): DataFrame chứa dữ liệu.
    - target_column (str, optional): Cột target do người dùng cung cấp.

    Returns:
    - str: Tên cột target được chọn.
    """
    if target_column and target_column in df.columns:
        return target_column

    # Danh sách các tên cột thường dùng cho target
    common_target_names = ['target', 'label', 'class', 'disease', 'income', 'severity']
    columns_lower = [col.lower() for col in df.columns]

    # Tìm cột target dựa trên tên phổ biến
    for col in df.columns:
        if col.lower() in common_target_names:
            print(f"Tự động chọn cột target: '{col}' (dựa trên tên phổ biến).")
            return col

    # Nếu không tìm thấy, chọn cột cuối cùng
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

    # Tự động chọn target_column nếu không được cung cấp hoặc không tồn tại
    target_column = auto_select_target_column(df, target_column)

    return hierarchy_dir, output_dir, id_column, target_column

def anonymize_data(input_file, k_anonymity, population_size=10, max_iter=500, k_share=1, id_column=None, target_column=None, hierarchy_prefix=None, ignore_missing_hierarchy=False):
    """
    Hàm ẩn danh dữ liệu với các tham số mặc định.

    Parameters:
    - input_file (str): Đường dẫn đến file CSV đầu vào.
    - k_anonymity (int): Hệ số k cho k-anonymity.
    - population_size (int): Kích thước quần thể (mặc định: 10).
    - max_iter (int): Số vòng lặp tối đa (mặc định: 500).
    - k_share (int): Số lần chia sẻ trong APO_ARO (mặc định: 1).
    - id_column (str, optional): Cột ID (mặc định: None).
    - target_column (str, optional): Cột mục tiêu (mặc định: None).
    - hierarchy_prefix (str, optional): Tiền tố của file hierarchy (mặc định: None).
    - ignore_missing_hierarchy (bool): Bỏ qua các cột không có hierarchy (mặc định: False).

    Returns:
    - tuple: (anon_output_file, results)
        - anon_output_file: Đường dẫn đến file CSV đã ẩn danh.
        - results: Dictionary chứa các thông số kết quả (IL, CA, Generalization Levels, Best Fitness, Penalty).
    """
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

    # Chuẩn bị dữ liệu để tính CA
    original_df = df.copy()
    targets_encoded, label_to_idx = embed_target(targets)
    original_df_encoded = one_hot_encoding(original_df)

    # Tải hierarchies trước
    hierarchies = load_hierarchy_from_csv_directory(hierarchy_dir, prefix=hierarchy_prefix)
    
    # Kiểm tra hierarchies
    if not hierarchies:
        raise ValueError(f"Không tìm thấy hierarchy nào trong thư mục '{hierarchy_dir}' với tiền tố '{hierarchy_prefix}'.")

    # Lấy qi_attributes ban đầu
    qi_attributes = get_qi_attributes(input_file, id_column=id_column, target_column=target_column, delimiter=';')
    print(f"Quasi-identifiers ban đầu: {qi_attributes}")

    # Chỉ giữ lại các thuộc tính có hierarchy
    qi_attributes = [attr for attr in qi_attributes if attr in hierarchies]
    if not qi_attributes:
        raise ValueError("Không có cột nào trong qi_attributes có hierarchy. Vui lòng kiểm tra thư mục hierarchies.")
    print(f"Quasi-identifiers sau khi lọc (chỉ giữ các cột có hierarchy): {qi_attributes}")

    # Tính số chiều (dim) dựa trên các thuộc tính có hierarchy
    dim = len(qi_attributes)
    print(f"Số chiều (dim): {dim}")

    # Tính ub và lb chỉ cho các thuộc tính có hierarchy
    lb = np.array([0] * dim)  # lb mặc định là 0
    ub = []
    for attr in qi_attributes:
        max_level = max(len(levels) for levels in hierarchies[attr].values()) - 1
        if max_level <= 0:
            raise ValueError(f"Hierarchy cho thuộc tính '{attr}' không hợp lệ: max_level={max_level}. Phải có ít nhất 1 mức tổng quát hóa.")
        ub.append(max_level)
    ub = np.array(ub)
    print(f"ub: {ub}")
    print(f"lb: {lb}")

    # Kiểm tra ub
    if len(ub) != dim:
        raise ValueError(f"Kích thước của ub ({len(ub)}) không khớp với số chiều (dim={dim}).")

    if any(u <= 0 for u in ub):
        raise ValueError(f"Giới hạn trên (ub) không hợp lệ: {ub}. Kiểm tra hierarchies để đảm bảo có ít nhất 1 mức tổng quát hóa cho mỗi thuộc tính.")

    zero_levels = np.zeros(dim)
    satisfies_k = check_k_anonymity(zero_levels, hierarchies, input_file, k_anonymity, load_hierarchy, qi_attributes)
    print(f"Dữ liệu gốc có thỏa mãn {k_anonymity}-anonymity không? {satisfies_k}")

    # Kiểm tra giá trị trong dữ liệu có khớp với hierarchy không
    for attr in qi_attributes:
        if attr in df.columns:
            unique_values = df[attr].astype(str).unique()
            hierarchy_values = set()
            for value, levels in hierarchies[attr].items():
                hierarchy_values.add(str(value).strip())  # Chuẩn hóa giá trị trong hierarchy thành chuỗi
            missing_values = [val for val in unique_values if val not in hierarchy_values]
            if missing_values:
                print(f"Cảnh báo: Các giá trị sau trong cột '{attr}' không có trong hierarchy: {missing_values}")

    # Chạy APO_ARO với điều kiện dừng
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

    # Chạy từng vòng lặp và kiểm tra điều kiện dừng
    min_iter = 1  # Khôi phục về giá trị cũ
    num_iterations = max_iter // k_share  # Số lần gọi iterate() để đạt max_iter vòng lặp
    for i in range(num_iterations):
        apo_aro.iterate()  # Gọi phương thức iterate() để thực hiện k_share vòng lặp
        current_best_solution = apo_aro.best_solution
        current_best_fitness = apo_aro.best_fitness

        # Kiểm tra k-anonymity cho giải pháp hiện tại
        current_total_iter = (i + 1) * k_share
        if current_total_iter >= min_iter:  # Sau ít nhất 50 vòng lặp
            if current_best_solution is None:
                print("Giải pháp hiện tại là None, tiếp tục chạy...")
                continue
            satisfies_k = check_k_anonymity(current_best_solution, hierarchies, input_file, k_anonymity, load_hierarchy, qi_attributes)
            if satisfies_k:
                print(f"Đã tìm thấy giải pháp thỏa mãn {k_anonymity}-anonymity sau {current_total_iter} vòng lặp. Dừng thuật toán.")
                break

    # Lấy giải pháp tốt nhất sau khi chạy
    best_solution = apo_aro.best_solution
    best_fitness = apo_aro.best_fitness

    if best_solution is None:
        raise ValueError("Không tìm thấy giải pháp hợp lệ sau khi chạy thuật toán APO_ARO.")

    print(f"Best solution found by APO_ARO: {best_solution}")
    print(f"Best fitness: {best_fitness}")

    # Tính lại fitness để lấy các giá trị il, m, ca
    fitness_value, il, m, ca_anon = fitness(best_solution, hierarchies, input_file, k_anonymity, load_hierarchy, ub, target_column, id_column, qi_attributes)

    # Ẩn danh dữ liệu với giải pháp tốt nhất
    generalization_levels = {qi_attributes[i]: int(best_solution[i]) for i in range(dim)}
    anon_output_file = os.path.join(output_dir, f'{dataset_name}_anonymized.csv')
    anonymize_dataset(generalization_levels, hierarchies, input_file, anon_output_file, delimiter=';')

    # Tính Classification Accuracy (CA) trên dữ liệu gốc
    anon_df = pd.read_csv(anon_output_file, delimiter=';')
    anon_df = anon_df.drop([target_column], axis=1, errors='ignore')
    anon_df_encoded = replace_generalization(anon_df, original_df)
    anon_df_encoded = anon_df_encoded[original_df_encoded.columns]

    # Chia dữ liệu thành tập train và test
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

    # Tạo dictionary kết quả
    results = {
        'il': il,  # Information Loss
        'ca_original': ca_original,  # CA trên dữ liệu gốc
        'ca_anon': ca_anon,  # CA trên dữ liệu ẩn danh (lấy từ fitness)
        'generalization_levels': generalization_levels,  # Mức độ ẩn danh cho từng thuộc tính
        'best_fitness': best_fitness,  # Độ tối ưu tốt nhất của fitness
        'target_column': target_column,  # Cột mục tiêu
        'm': m  # Hệ số phạt (Penalty)
    }

    return anon_output_file, results