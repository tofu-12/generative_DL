import torch
import torch.nn as nn


# デバイスの取得
def get_device() -> torch.device:
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

# モデルの保存
def save_model(model: nn.Module, weights_file_path: str, model_file_path:str) -> None:
    """
    重みとモデルを保存する

    Args:
        model: モデル
        weights_file_path: 重みファイルのパス
        model_file_path: モデルファイルのパス
    
    Raise:
        Exception: エラーが発生した場合
    """
    try:
        torch.save(model.state_dict(), weights_file_path)
        torch.save(model, model_file_path)
    except Exception as e:
        raise Exception(f"モデルの保存に失敗しました: {str(e)}")
    
def save_model_with_check(model: nn.Module, weights_file_path: str, model_file_path:str) -> None:
    """
    確認して、重みとモデルを保存する

    Args:
        model: モデル
        weights_file_path: 重みファイルのパス
        model_file_path: モデルファイルのパス

    Raise:
        Exception: エラーが発生した場合
    """
    try:
        is_save_str = input("モデルを保存しますか (y/n) >> ")
        is_save = False if is_save_str == "n" else True

        if is_save:
            save_model(model, weights_file_path, model_file_path)

    except Exception as e:
        raise e
