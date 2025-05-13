# run_simple_gan.py

## Scope
SimpleGANを動かすプログラムを実装する。

## Backgroud
SimpleGANのモデルの実装を終えたため、次にSimpleGANを動かしたい。

## Proposal
### 概要
main関数で基本的に全体を動作させる。  
標準入力でモードを選択する。
1. 訓練
2. テスト
3. 画像の生成

### 主な関数
- **main(): 全体を制御する関数**
- **train(): 訓練を行う関数**
- **test(): テストを行う関数**
- **visualize_history(): 学習履歴を可視化する関数**
- **generate(): 画像を生成する関数**

### main()
- 基本的な流れは以下の通り
    1. データの設定
    2. モデルの設定
    3. 損失関数の設定
    4. 最適化手法の設定
    5. モードの選択 (select_mode())
    6. モードの実行

- どこでエラーが発生したか分かるような例外処理

### train()
- 引数と返り値
    ```python
    def train(
        dataloaders: Dataloaders,
        discriminator: nn.Module,
        generator: nn.Module,
        loss_function: Callable,
        discriminator_optim: optim,
        generator_optim: optim
    ) -> None:
    ```

- epoch単位の基本的な流れは以下の通り
    1. 識別器のラベルの作成
    2. 実画像に対する識別器の損失の算出
    3. 生成器による偽画像の生成
    4. 偽画像に対する識別器の損失の算出
    5. 識別器の重みの更新
    6. 生成器の重みの更新

- 学習の過程で学習履歴を保存
- エラーの際にはデータを保存するか確認するようにする
