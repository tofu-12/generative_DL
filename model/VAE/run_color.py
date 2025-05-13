import os
import sys

import torch.optim as optim

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
from color_model import ColorVAE
from dataset.CelebA_dataset import get_CelebA_data
from module_blocks.loss_functions import bce_reconstruction_loss
from utils import select_run_mode, vae_run_with_selected_mode
from vae_run_client import VAERunClient


if __name__ == "__main__":
    # 保存先のパス
    weights_file_name = "color_vae.pth"
    model_file_name = "color_vae.pth"

    weights_file_path = os.path.join(os.path.dirname(__file__), "weights_file", weights_file_name)
    model_file_path = os.path.join(os.path.dirname(__file__), "model_file", model_file_name)

    # モデルを実行するクライエントの準備
    client = VAERunClient()
    client.set_model(ColorVAE, 200)

    optimizer = optim.Adam(client.model.parameters(), lr=0.001)
    loss_function = bce_reconstruction_loss
    client.set_loss_function_and_optimizer(loss_function, optimizer)

    batch_size = 32
    epoch = 5
    client.set_data(batch_size, get_CelebA_data)

    # モードの選択と実行
    mode = select_run_mode()
    vae_run_with_selected_mode(
        mode,
        client,
        batch_size,
        epoch,
        weights_file_path,
        model_file_path
    )

    if mode == "train_with_weights_file" or mode == "train_without_weights_file":
        client.visualize_final_z(False)
