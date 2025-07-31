import torch
import numpy as np
from esm import pretrained
import os
import argparse  
from scipy.linalg import sqrtm
def check_for_nan_and_infs(tensor):
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)

    if nan_mask.any():
        print("found NaN values in the tensor")
    else:
        print("No NaN values found")

    if inf_mask.any():
        print("found infinite values in the tensor")
    else:
        print("No infinite values found")

def create_nac_tensor(pdb_directory):
    """
    从指定的 PDB 目录生成 N、CA 和 C 原子的 NCAC 张量。

    :param pdb_directory: 包含 PDB 文件的目录路径
    :return: 一个列表，包含每个 PDB 文件的 NCAC 张量
    """
    nac_tensors = []  # 存储每个文件的 NCAC 张量

    # 遍历目录下的所有 PDB 文件
    for filename in os.listdir(pdb_directory):
        if filename.endswith(".pdb"):
            pdb_file_path = os.path.join(pdb_directory, filename)

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
                nac_tensor = np.expand_dims(nac_tensor, axis=0)
                nac_tensors.append(nac_tensor)

    if nac_tensors:
        combined_tensor = np.concatenate(nac_tensors, axis=0)
    else:
        combined_tensor = np.empty((0, 3, 3))

    return combined_tensor

def process_pdb_directory(pdb_directory,  encoder):
    """
    处理给定的 PDB 目录，并返回 encoder 的输出。

    :param pdb_directory: PDB 文件的目录路径
    :param model: 预训练的模型
    :param encoder: 模型中的 encoder
    :return: encoder 的输出
    """
    # 生成 NCAC 张量
    ref_embedding = create_nac_tensor(pdb_directory)
    print("NCAC_shape:", ref_embedding.shape)

    # 假设 ref_embedding 的形状为 (b, length, 3, 3)
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
    """
    计算两组嵌入之间的 FID 值。
    
    :param ref_embedding: 参考嵌入，形状为 (b, length, feature_dim)
    :param data_embedding: 要比较的嵌入，形状为 (b, length, feature_dim)
    :return: FID 值
    """
    # 将张量转为 NumPy 数组
    ref_embedding_np = ref_embedding.detach().numpy()
    data_embedding_np = data_embedding.detach().numpy()

    # 计算均值
    mu_ref = np.mean(ref_embedding_np, axis=0)  # 计算均值，结果为 (length, feature_dim)
    mu_data = np.mean(data_embedding_np, axis=0)  # (length, feature_dim)

    # 为了计算协方差，重塑嵌入以消除 batch_size 维度
    # 合并 batch_size 和 length 维度
    ref_embedding_reshaped = ref_embedding_np.reshape(-1, ref_embedding_np.shape[-1])  # 变形为 (b * length, feature_dim)
    data_embedding_reshaped = data_embedding_np.reshape(-1, data_embedding_np.shape[-1])  # 变形为 (b * length, feature_dim)

    # 计算协方差矩阵
    sigma_ref = np.cov(ref_embedding_reshaped, rowvar=False)  # (feature_dim, feature_dim)
    sigma_data = np.cov(data_embedding_reshaped, rowvar=False)

    # 计算 FID
    fid = np.sum((mu_ref - mu_data) ** 2)  # 计算均值的平方差
    sigma_product = sqrtm(sigma_ref.dot(sigma_data))

    # 检查协方差的平方根是否返回 NaN
    if np.isnan(sigma_product).any():
        print("Warning: sigma_product contains NaN values!")
        sigma_product = np.nan_to_num(sigma_product)

    # 计算 FID
    fid += np.trace(sigma_ref) + np.trace(sigma_data) - 2 * np.trace(sigma_product)

    return fid
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='Process PDB directories for embeddings.')
    
    # 定义命令行参数
    parser.add_argument('--ref_pdb_directory', type=str, required=True, 
                        help='Directory containing reference PDB files.')
    parser.add_argument('--pre_pdb_directory', type=str, required=True, 
                        help='Directory containing data PDB files.')

    # 解析命令行参数
    args = parser.parse_args()
    
    # 使用命令行参数替换硬编码的目录
    ref_pdb_directory = args.ref_pdb_directory
    data_pdb_directory = args.pre_pdb_directory
    
    # 加载模型和字母表
    model, alphabet = pretrained.load_model_and_alphabet('/liuyunfan/qianyunhang/esm_if1_gvp4_t16_142M_UR50.pt')

    # 设置模型为评估模式
    model.eval()

    # 获取 GVPTransformerModel 中的 encoder
    encoder = model.encoder

    # 调用处理函数并获取 encoder 的输出
    ref_embedding = process_pdb_directory(ref_pdb_directory, encoder)
    data_embedding = process_pdb_directory(data_pdb_directory, encoder)

    # 打印输出
    print("ref_embedding:", ref_embedding.shape)  # 输出形状
    print("data_embedding:", data_embedding.shape) 
    check_for_nan_and_infs(ref_embedding)
    check_for_nan_and_infs(data_embedding)
    fid_value = calculate_fid(ref_embedding, data_embedding)

    print("FID Value:", fid_value)
# if __name__ == "__main__":
#     # PDB 文件目录
#     ref_pdb_directory = "/liuyunfan/qianyunhang/ckpt_samples/foldflow-base/sample_100/samples/"
#     data_pdb_directory = "/liuyunfan/qianyunhang/ckpt_samples/foldflow-base/sample_100/designs/"
    
#     # 加载模型和字母表
#     model, alphabet = pretrained.load_model_and_alphabet('/liuyunfan/qianyunhang/esm_if1_gvp4_t16_142M_UR50.pt')

#     # 设置模型为评估模式
#     model.eval()

#     # 获取 GVPTransformerModel 中的 encoder
#     encoder = model.encoder

#     # 调用处理函数并获取 encoder 的输出
#     ref_embedding = process_pdb_directory(ref_pdb_directory,  encoder)
#     data_embedding = process_pdb_directory(data_pdb_directory, encoder)

#     # 打印输出
#     print("ref_embedding:", ref_embedding.shape)  # 输出形状
#     print("data_embedding:", data_embedding.shape) 
#     check_for_nan_and_infs(ref_embedding)
#     check_for_nan_and_infs(data_embedding)
#     fid_value = calculate_fid(ref_embedding, data_embedding)

#     print("FID Value:", fid_value)
  