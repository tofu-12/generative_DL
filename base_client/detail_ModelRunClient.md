# ModelRunClientの詳細

## 使用の流れ
1. クライエントの作成
    ```python
    class ChildClient(ModelRunClient):
        def __init__(self):
            super().__init__()
    ```

2. インスタンスの生成
    ```python
    client = ChildClient()
    ```

3. 機械学習モデルの設定
    ```python
    # Model: nn.Module
    client.set_model(Model)
    ```

4. 損失関数と最適化手法の設定
    ```python
    # 損失関数の例
    """
    損失関数の型

    Callable:
        predict, target, その他の要素 -> loss
    """
    loss_function = bce_reconstruction_loss

    # 最適化手法の例
    optimizer = optim.Adam(client.model.parameters(), lr=0.001)

    # 損失関数と最適化手法の設定
    client.set_loss_function_and_optimizer(loss_function, optimizer)
    ```

5. データの設定
    ```python
    # Dataloadersの設定
    lass Dataloaders(BaseModel):
        """
        データローダをまとめるデータ型

        Args:
            train: 訓練用データローダ
            val: 検証用データローダ
            test: テスト用データローダ
        """
        model_config = ConfigDict(arbitrary_types_allowed=True)

        train: torch.utils.data.DataLoader
        val: torch.utils.data.DataLoader
        test: torch.utils.data.DataLoader

    # データ取得関数の設定
    """
    データ取得関数の型

    Callable:
        batch_size: int -> Dataloaders
    """
    get_data_function = sample_get_data_function

    # バッチサイズとエポック数の設定
    batch_size = 32
    epoch = 5

    # データの設定
    client.set_data(batch_size, get_data_function)

6. 学習の実行
    ```python
    client.run_training(
        batch_size,
        epoch,
        weights_file_name, 
        model_file_name,
        loading_weights=True
    )
    ```

7. 学習履歴の可視化
    ```python
    class History(BaseModel):
        """
        history（run_trainingの際にこれをインスタンスの変数に設定する）
    
        Args:
            train_loss_per_batch: バッチごとの訓練の損失
            train_loss_per_epoch: エポックごとの訓練の損失
            val_loss_per_epoch: 検証の損失
            test_loss: テストの損失
        """
        train_loss_per_batch: list = []
        train_loss_per_epoch: list = []
        val_loss_per_epoch: list = []
        test_loss: list = []
    
    client.visualize_loss_history()
    ```

8. 精度の検証
    ```python
    client.run_test(model_file_name, checking_test_loss=True)
    ```

## 継承した際に実装する関数
```python
# 訓練関連
def _training(self):
    raise NotImplementedError

def _train_loop(self):
    raise NotImplementedError

def _val_loop(self):
    raise NotImplementedError

# 実行関連
def run_training(self):
    raise NotImplementedError

def run_test(self):
    raise NotImplementedError
```

## 継承した際に実装ずみの関数
```python
# 初期化関連
def __init__(self):
def _get_device(self) -> torch.device:
def set_model(self, model: nn.Module) -> None:
def set_loss_function_and_optimizer(self, loss_function: Callable, optimizer: optim) -> None:
def set_data(self, batch_size: int, get_data_function: Callable) -> None:

# ファイル関連
def _save_weights(self, weights_file_name: str) -> None:
def _save_model(self, model_file_name: str) -> None:
def _save_with_checking(self, weights_file_name: str, model_file_name: str) -> None:
def _load_params(self, weights_file_name: str) -> None:
def _load_model(self, model_file_name: str) -> None:

# 可視化関連
def visualize_model_size(self, checking_model_structure: bool=False) -> None:
def visualize_loss_history(self, history: History=None) -> None:
```
