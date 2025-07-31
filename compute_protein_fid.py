import torch
import numpy as np
from esm import pretrained
import os
from scipy.linalg import sqrtm
import argparse  
def sort_pdb(pdb_directory):
    file_info = []  # 存储文件名和对应的整数
    pdb_info = [] 

    # 遍历目录下的所有 PDB 文件
    for filename in os.listdir(pdb_directory):
        if filename.endswith(".pdb"):
            num_char = next((c for c in filename if c.isdigit()), None)
            if num_char is not None:
                num = int(num_char)
                file_info.append((filename, num))

    # Sort files by the integer value extracted
    file_info.sort(key=lambda x: x[1])

    # Read sorted PDB files
    for filename, _ in file_info:
        pdb_file_path = os.path.join(pdb_directory, filename)
        pdb_info.append(pdb_file_path)
    return pdb_info

def check_for_nan_and_infs(tensor):
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)

    if nan_mask.any():
        print("Found NaN values in the tensor.")
    else:
        print("No NaN values found.")

    if inf_mask.any():
        print("Found infinite values in the tensor.")
    else:
        print("No infinite values found.")

def create_nac_tensor(pdb_file_path):
    nac_tensor = []
    with open(pdb_file_path, 'r') as file:
        lines = file.readlines()

    # 存储 N、CA 和 C 原子的坐标
    coordinates_dict = {'N': [], 'CA': [], 'C': []}

    for line in lines:
        if line.startswith("ATOM"):
            parts = line.split()
            if len(parts) >= 7 and parts[2] in coordinates_dict:
                coordinates = parts[6:9]  # 提取 x, y, z 坐标
                try:
                    x, y, z = [float(coord) for coord in coordinates]
                    coordinates_dict[parts[2]].append((x, y, z))
                except ValueError:
                    print(f"无法转换坐标：{coordinates}")

    # 创建 NCAC 张量
    n_coords = np.array(coordinates_dict['N'])
    ca_coords = np.array(coordinates_dict['CA'])
    c_coords = np.array(coordinates_dict['C'])

    min_length = min(len(n_coords), len(ca_coords), len(c_coords))
    if min_length > 0:
        nac_tensor = np.zeros((min_length, 3, 3))
        for i in range(min_length):
            nac_tensor[i, 0] = n_coords[i]
            nac_tensor[i, 1] = ca_coords[i]
            nac_tensor[i, 2] = c_coords[i]
        nac_tensor = np.expand_dims(nac_tensor, axis=0)  # Adding batch dimension

    return nac_tensor

def process_pdb_file(pdb_file_path, encoder):
    ref_embedding = create_nac_tensor(pdb_file_path)
    if ref_embedding.shape[0] == 0:
        print(f"No data in file: {pdb_file_path}")
        return None

    # print("NCAC_shape:", ref_embedding.shape)

    # 假设 ref_embedding 的形状为 (1, length, 3, 3)
    batch_size, length, _, _ = ref_embedding.shape

    # 创建 padding_mask
    padding_mask = torch.zeros(batch_size, length, dtype=torch.bool)

    # 创建 confidence 张量
    confidence = torch.ones(batch_size, length)

    # 将 ref_embedding 转换为张量
    ref_embedding_tensor = torch.tensor(ref_embedding, dtype=torch.float)

    # 执行前向传播
    encoder_output = encoder(ref_embedding_tensor, encoder_padding_mask=padding_mask, confidence=confidence)

    # 获取 encoder 的输出
    encoder_out = encoder_output['encoder_out'][0]  # 选择第一个 batch

    return encoder_out

def calculate_fid(ref_embedding, data_embedding):
    ref_embedding_np = ref_embedding.detach().numpy()
    data_embedding_np = data_embedding.detach().numpy()

    mu_ref = np.mean(ref_embedding_np, axis=0)
    mu_data = np.mean(data_embedding_np, axis=0)

    ref_embedding_reshaped = ref_embedding_np.reshape(-1, ref_embedding_np.shape[-1])
    data_embedding_reshaped = data_embedding_np.reshape(-1, data_embedding_np.shape[-1])

    sigma_ref = np.cov(ref_embedding_reshaped, rowvar=False)
    sigma_data = np.cov(data_embedding_reshaped, rowvar=False)

    fid = np.sum((mu_ref - mu_data) ** 2)
    sigma_product = sqrtm(sigma_ref.dot(sigma_data))

    if np.isnan(sigma_product).any():
        print("Warning: sigma_product contains NaN values!")
        sigma_product = np.nan_to_num(sigma_product)

    fid += np.trace(sigma_ref) + np.trace(sigma_data) - 2 * np.trace(sigma_product)

    return fid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process PDB directories for embeddings.')
    
    # 定义命令行参数
    parser.add_argument('--ref_pdb_directory', type=str, required=True, 
                        help='Directory containing reference PDB files.')
    parser.add_argument('--pre_pdb_directory', type=str, required=True, 
                        help='Directory containing data PDB files.')

    # 解析命令行参数
    args = parser.parse_args()
    ref_pdb_directory =args.ref_pdb_directory# "/liuyunfan/qianyunhang/ckpt_samples/foldflow-base/sample_100/samples/"
    pre_pdb_directory = args.pre_pdb_directory#"/liuyunfan/qianyunhang/ckpt_samples/foldflow-base/sample_100/designs/"
    ref_pdb_info = sort_pdb(ref_pdb_directory)
    pre_pdb_info = sort_pdb(pre_pdb_directory)
    
    # 加载模型和字母表
    model, alphabet = pretrained.load_model_and_alphabet('/liuyunfan/qianyunhang/esm_if1_gvp4_t16_142M_UR50.pt')
    model.eval()
    encoder = model.encoder
    fid_values = []

    # Process each corresponding pair of PDB files
    for i, ref_file_path in enumerate(ref_pdb_info):  # Use enumerate to get index and path
        pre_file_path = pre_pdb_info[i]
        print(f"data_file_path:{pre_file_path}")
        print(f"ref_file_path:{ref_file_path}")

        # Process both files
        ref_embedding = process_pdb_file(ref_file_path, encoder)
        pre_embedding = process_pdb_file(pre_file_path, encoder)

        if ref_embedding is not None and pre_embedding is not None:
            # Calculate FID
            fid_value = calculate_fid(ref_embedding, pre_embedding)
            fid_values.append(fid_value)
            print(f"FID Value between {os.path.basename(ref_file_path)} and {os.path.basename(pre_file_path)}: {fid_value}")

    # Calculate average FID
    if fid_values:
        average_fid = np.mean(fid_values)
        average_fid_real = average_fid.real
        print(f"Average FID Value: {average_fid}")
        print(f"Average FID_real Value: {average_fid_real}")
    else:
        print("No FID values were calculated.")
