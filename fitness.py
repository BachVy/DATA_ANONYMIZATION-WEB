import pandas as pd
import numpy as np
from anonymize import anonymize_dataset, load_hierarchy

def check_lengths(mapped_qi_attributes, generalization_levels):
    if len(mapped_qi_attributes) != len(generalization_levels):
        raise ValueError(
            f"Độ dài của mapped_qi_attributes ({len(mapped_qi_attributes)}) "
            f"không khớp với độ dài của generalization_levels ({len(generalization_levels)}). "
            f"mapped_qi_attributes: {mapped_qi_attributes}, "
            f"generalization_levels: {generalization_levels}"
        )

def calculate_ncp(generalization_levels, max_levels):
    ncp_values = []
    max_levels = max_levels.tolist() if isinstance(max_levels, np.ndarray) else max_levels
    max_level = max(max_levels) if max_levels else 1
    for applied_level in generalization_levels:
        if max_level == 0:
            ncp = 0
        else:
            ncp = applied_level / max_level
        ncp_values.append(ncp)
    return ncp_values

def calculate_il(generalization_levels, max_levels):
    ncp_values = calculate_ncp(generalization_levels, max_levels)
    n = len(ncp_values)
    if n == 0:
        return 0
    return (1 / n) * sum(ncp_values)

def check_k_anonymity(generalization_levels, hierarchies, input_file, k, load_hierarchy_func, mapped_qi_attributes):
    check_lengths(mapped_qi_attributes, generalization_levels)

    temp_output_file = "temp_anonymized.csv"
    generalization_dict = {mapped_qi_attributes[i]: int(level) for i, level in enumerate(generalization_levels)}
    anonymize_dataset(generalization_dict, hierarchies, input_file, temp_output_file, delimiter=';')

    anon_df = pd.read_csv(temp_output_file, delimiter=';')

    qi_columns = mapped_qi_attributes
    grouped = anon_df.groupby(qi_columns).size()
    min_group_size = grouped.min()
    satisfies_k_anonymity = all(count >= k for count in grouped.values)

    import os
    os.remove(temp_output_file)

    return satisfies_k_anonymity

def calculate_m(generalization_levels, hierarchies, input_file, k, load_hierarchy_func, mapped_qi_attributes):
    satisfies_k_anonymity = check_k_anonymity(generalization_levels, hierarchies, input_file, k, load_hierarchy_func, mapped_qi_attributes)
    m = 0 if satisfies_k_anonymity else 1
    return m

def calculate_ca(generalization_levels, hierarchies, input_file, target_column, id_column, mapped_qi_attributes):
    import os
    from train_dynamic import one_hot_encoding, embed_target, replace_generalization, SVM

    df = pd.read_csv(input_file, delimiter=';')
    if id_column and id_column in df.columns:
        df = df.drop([id_column], axis=1)
    
    targets = list(df[target_column])
    df = df.drop([target_column], axis=1)

    check_lengths(mapped_qi_attributes, generalization_levels)

    temp_output_file = "temp_anonymized.csv"
    generalization_dict = {mapped_qi_attributes[i]: int(level) for i, level in enumerate(generalization_levels)}
    anonymize_dataset(generalization_dict, hierarchies, input_file, temp_output_file, delimiter=';')
   
    anon_df = pd.read_csv(temp_output_file, delimiter=';')
    if id_column and id_column in anon_df.columns:
        anon_df = anon_df.drop([id_column], axis=1)
    anon_df = anon_df.drop([target_column], axis=1, errors='ignore')

    one_hot_df = one_hot_encoding(df.copy())
    one_hot_anon_df = replace_generalization(anon_df, df)
    embeded_targets, label_to_idx = embed_target(targets)

    dataset_name = os.path.basename(input_file).replace('.csv', '')
    output_dir = os.path.join(os.path.dirname(os.path.dirname(input_file)).replace('data', 'results'), dataset_name)
    train_file = os.path.join(output_dir, f'{dataset_name}_train.txt')
    val_file = os.path.join(output_dir, f'{dataset_name}_val.txt')
    with open(train_file, 'r') as f:
        train_indexes = [int(i) for i in f.read().splitlines()]
    with open(val_file, 'r') as f:
        val_indexes = [int(i) for i in f.read().splitlines()]

    clf = SVM()
    train_features = one_hot_anon_df.iloc[train_indexes]
    train_targets = [embeded_targets[i] for i in train_indexes]
    val_features = one_hot_anon_df.iloc[val_indexes]
    val_targets = [embeded_targets[i] for i in val_indexes]

    clf.fit(train_features, train_targets)
    f1_score = clf.eval(val_features, val_targets, label_to_idx.keys())

    os.remove(temp_output_file)

    return f1_score

def fitness(generalization_levels, hierarchies, input_file, k, load_hierarchy_func, max_levels, target_column, id_column, mapped_qi_attributes):
    alpha = 0.5
    lambda_ = 10.0
    gamma = 0.5

    m = calculate_m(generalization_levels, hierarchies, input_file, k, load_hierarchy_func, mapped_qi_attributes)
    il = calculate_il(generalization_levels, max_levels)
    ca = calculate_ca(generalization_levels, hierarchies, input_file, target_column, id_column, mapped_qi_attributes)

    f_x = alpha * il + lambda_ * m + gamma * (1 - ca)

    return f_x, il, m, ca