from vae_run_client import VAERunClient


def select_run_mode() -> str:
    """
    実行するモードの選択

    Returns:
        実行するモード:
            train_with_weights_file,
            train_without_weights_file,
            test,
            test_only_pic
    """
    try:
        mode_dict = {"1": "train_with_weights_file", "2": "train_without_weights_file", "3": "test", "4": "test_only_pic"}

        print("-" * 50)
        print("1: train with weights file")
        print("2: train without weights file")
        print("3: test")
        print("4: test only pic")
        mode_num = input("モードを選択してください >> ")
        print("-" * 50)
        mode = mode_dict[mode_num]

        return mode

    except KeyError:
        print("入力した番号が誤っています")
        return select_run_mode()
    
    except Exception as e:
        raise Exception(f"モードの選択に失敗しました: {str(e)}")

def vae_run_with_selected_mode(
        mode: str,
        client: VAERunClient,
        batch_size: int,
        epoch: int,
        weights_file_path: str,
        model_file_path: str
) -> None:
    """
    選ばれたモードを実行

    Args:
        mode: 実行するモード
        client: ModelRunClient
        batch_size: バッチサイズ
        epoch: エポック数
        weights_file_name: パラメータファイル名
        model_file_name: モデルファイル名
    """
    if mode == "train_with_weights_file":
        client.run_training(
            batch_size,
            epoch,
            weights_file_path, 
            model_file_path,
            loading_weights=True
        )
        client.visualize_loss_history()
    
    if mode == "train_without_weights_file":
        client.run_training(
            batch_size,
            epoch,
            weights_file_path,
            model_file_path,
            loading_weights=False
        )
        client.visualize_loss_history()

    elif mode == "test":
        client.run_test(model_file_path, checking_test_loss=True)
    
    elif mode == "test_only_pic":
        client.run_test(model_file_path, checking_test_loss=False)
