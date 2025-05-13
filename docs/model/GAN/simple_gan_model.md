# simple_gan_model.py

## Scope
32x32の白黒imageに対応した最もシンプルなGANのモデルを実装する。

## Background
生成モデルの一種である基本的なGANの実装を通して、GANについての理解を深めるとともに、よりPytorchの実装に慣れる。  
また、次回以降でより発展したモデルを作成するための土台を作る。

## Proposal
### 実装するクラス
- Discriminator
- Generator
- SimpleGAN

### Discriminator
**本物の画像か生成機が生成した画像かを分類する分類器**
- 画像データから2値分類を行う。  
- モデルのアーキテクチャは基本的なConv -> bn -> leaky_reluのブロックを重ねる。  
- 過学習防止のために、Dropoutを途中に挟む。
- loss_function: bce

### Generator
**多変量正規分布からランダムに選ばれたベクトルから画像を生成する生成機**
- conv.t -> bn -> rleaky_eluを重ねてサイズを大きくしながら画像を生成する。
- 過学習防止のために、Dropoutを途中に挟む。
- loss_function: bce

### SimpleGAN
**基本的なGAN**
- インスタンス変数として、DiscriminatorとGeneratorを持つ。

## アーキテクチャ
### Discriminator
1. conv(kernel_size=3, stride=2, paddong=1) -> leaky_relu -> dropout
2. ( conv(kernel_size=3, stride=2, padding=1) -> bn -> leaky_relu -> dropout ) x n
3. conv -> view(-1, m)
4. linear(m, 1)

### Generator
1. view(-1, latent_dim, 1, 1)
2. ( conv.t -> bn -> leaky_relu ) x n
3. conv.t
