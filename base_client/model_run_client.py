import os
import sys
from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from schemas import Dataloaders, History


class ModelRunClient:
    # 初期化関連
    def __init__(self):
        """
        モデルを実行するクライエントのインスタンスの初期化
        """
        self.device: torch.device = None

        self.dataloaders: Dataloaders = None

        self.model: nn.Module = None
        self.optimizer: optim = None
        self.loss_function: Callable = None
        
        self.history = None
    
    def _get_device(self) -> torch.device:
        """
        使用するデバイスを取得

        Returns:
            device (mps or cpu)
        """
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("MPS device is available and being used.")
        else:
            device = torch.device("cpu")
            print("MPS device is not available, using CPU instead.")
        
        return device
    
    def set_model(self, model: nn.Module) -> None:
        """
        モデルの設定

        Args:
            model: モデル
        
        Raise:
            Exception: 任意のエラー
        """
        try:
            # デバイスの設定
            self.device = self._get_device()

            # モデルの設定
            self.model = model().to(self.device)
        
        except Exception as e:
            raise Exception(f"モデルの設定に失敗しました:\n{str(e)}")
    
    def set_loss_function_and_optimizer(self, loss_function: Callable, optimizer: optim) -> None:
        """
        損失関数と最適化手法の設定

        Args:
            loss_function: 損失関数
            optimizer: 最適化手法
        
        Raise:
            Exception: 任意のエラー
        """
        try:
            self.loss_function = loss_function
            self.optimizer = optimizer
        
        except Exception as e:
            raise Exception(f"損失関数と最適化手法の設定に失敗しました:\n{str(e)}")
    
    def set_data(self, batch_size: int, get_data_function: Callable) -> None:
        """
        データの設定

        Args:
            batch_size: バッチサイズ
            get_data_function: データ取得関数
        
        Raise:
            Exception: 任意のエラー
        """
        try:
            self.dataloaders = get_data_function(batch_size)
        
        except Exception as e:
            raise Exception(f"データの設定に失敗しました:\n{str(e)}")
    

    # 訓練関連
    def _training(self):
        raise NotImplementedError

    def _train_loop(self):
        raise NotImplementedError
    
    def _val_loop(self):
        raise NotImplementedError
    

    # ファイル関連
    def _save_weights(self, weights_file_path: str) -> None:
        """
        モデルのパラメータを保存する

        Args:
            weights_file_path: パラメータファイルパス

        Raise:
            Exception: 任意のエラー
        """
        try:
            print("-" * 50)
            print("Saving Parameter")
            torch.save(self.model.state_dict(), weights_file_path)
            print("Done!")
            print("-" * 50)
        
        except Exception as e:
            raise Exception(f"パラメータの保存に失敗しました: {str(e)}")
    
    def _save_model(self, model_file_path: str) -> None:
        """
        モデルを保存する

        Args:
            model_file_path: モデルファイルパス

        Raise:
            Exception: 任意のエラー
        """
        try:
            print("-" * 50)
            print("Saving Model")
            torch.save(self.model, model_file_path)
            print("Done!")
            print("-" * 50)
        
        except Exception as e:
            raise Exception(f"モデルの保存に失敗しました: {str(e)}")
    
    def _save_with_checking(self, weights_file_path: str, model_file_path: str) -> None:
        """
        保存するか確認したのちに、パラメータとモデルを保存

        Args:
            weights_file_path: パラメータファイルパス
            model_file_path: モデルファイルパス

        Raise:
            Exception: 任意のエラー
        """
        try:
            is_save = input("モデルを保存しますか？ (y/n) >> ")
            if is_save == "y":
                self._save_weights(weights_file_path)
                self._save_model(model_file_path)

        except Exception as e:
            raise e
    
    def _load_params(self, weights_file_path: str) -> None:
        """
        モデルにパラメータをロードする

        Args:
            weights_file_path: パラメータファイルパス
        """
        try:
            if os.path.exists(weights_file_path):
                print("-" * 50)
                print("Load Parameter")
                self.model.load_state_dict(torch.load(weights_file_path))
                print("Done!")
                print("-" * 50)
            
            else:
                raise Exception(f"パラメータファイル ({weights_file_path}) が存在しません")
        
        except Exception as e:
            print("-" * 50)
            print(f"パラメータのロードに失敗しました: {str(e)}")
            print("パラメータをロードしていないモデルを使用します")
            print("-" * 50)
        
    def _load_model(self, model_file_path: str) -> None:
        """
        モデルをロードする

        Args:
            model_file_path: モデルファイルパス
        
        Raise:
            Exception: 任意のエラー
        """
        try:
            if os.path.exists(model_file_path):
                print("-" * 50)
                print("Load Model")
                self.model = torch.load(model_file_path, weights_only=False)
                print("Done!")
                print("-" * 50)
            else: 
                raise Exception("モデルファイルが存在しません")
        
        except Exception as e:
            raise Exception(f"モデルのロードに失敗しました: {str(e)}")
    

    # 実行関連
    def run_training(self):
        raise NotImplementedError
    
    def run_test(self):
        raise NotImplementedError
    

    # 可視化関連
    def visualize_model_size(self, checking_model_structure: bool=False) -> None:
        """
        モデルのサイズを確認

        Args:
            is_checking_model_structure: モデルの構造を確認するか否か
        """
        try:
            if self.model:
                # トータルパラメータ数
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"モデル全体のパラメータ数: {total_params}")

                # 学習可能パラメータ数
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"学習対象のパラメータ数: {trainable_params}")

                # モデルの構造
                if checking_model_structure:
                    for name, parameter in self.model.named_parameters():
                        print(f"Layer: {name} | Size: {parameter.size()} | Numel: {parameter.numel()} | Requires_grad: {parameter.requires_grad}")

            else:
                print("モデルが設定されていません")
        
        except Exception as e:
            print(f"モデルのサイズを確認に失敗しました: {str(e)}")

    def visualize_loss_history(self, history: History=None) -> None:
        """
        学習履歴の可視化

        Args:
            history: 学習履歴 (Noneの場合はself.historyにする)
        """
        try:
            if not history:
                history = self.history
            
            if history is None:
                print("学習履歴がありません。モデルを訓練してから実行してください。")
                return

            num_plots = 0
            if history.train_loss_per_batch and len(history.train_loss_per_batch) > 0:
                num_plots += 1
            if history.train_loss_per_epoch and len(history.train_loss_per_epoch) > 0:
                num_plots += 1
            if history.val_loss_per_epoch and len(history.val_loss_per_epoch) > 0:
                num_plots += 1

            if num_plots == 0:
                print("可視化する損失データがありません。")
                return

            plt.figure(figsize=(12, 5 * num_plots))
            plot_index = 1

            # 訓練損失 (バッチごと) の可視化
            if history.train_loss_per_batch:
                plt.subplot(num_plots, 1, plot_index)
                plt.plot(history.train_loss_per_batch)
                plt.title('Training Loss per Batch')
                plt.xlabel('Batch Index')
                plt.ylabel('Loss')
                plt.grid(True)
                plot_index += 1

            # 訓練損失 (エポックごと) の可視化
            if history.train_loss_per_epoch:
                plt.subplot(num_plots, 1, plot_index)
                plt.plot(history.train_loss_per_epoch, marker='o')
                plt.title('Training Loss per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Average Loss')
                plt.grid(True)
                plot_index += 1

            # 検証損失 (エポックごと) の可視化
            if history.val_loss_per_epoch:
                plt.subplot(num_plots, 1, plot_index)
                plt.plot(history.val_loss_per_epoch, marker='o', color='orange')
                plt.title('Validation Loss per Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Average Loss')
                plt.grid(True)
                plot_index += 1

            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f"学習履歴の可視化に失敗しました: {str(e)}")
