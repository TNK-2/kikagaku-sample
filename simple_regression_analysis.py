import numpy as np
# ベクトルの定義
x = np.array([1, 2, 3])
y = np.array([2, 3.9, 6.1])

# データの中心化
# 平均の算出
x.mean()
y.mean()

xc = x - x.mean()
yc = y - y.mean()

# パラメータaの計算
# 要素ごとの掛け算
xx = xc * xc
xy = xc * yc
xx.sum()
xy.sum()

# 傾きの算出
a = xy.sum() / xx.sum()
# print(a)

import pandas as pd
# pandas ; データベースの操作
# df ; data frame
df = pd.read_csv('sample.csv')
# print(df)
# print(df.head(3))

# データの抽出
x = df['x']
y = df['y']
# print(y)

import matplotlib.pyplot as plt
# Matplotlib : グラフの描画
# 横軸をx、縦軸をyの散布図でプロット
# plt.scatter(x,y)
# plt.show()

# 単回帰分析の実装
# データの中心化
# データの概要を表示
# print(df.describe())
# print(df.mean())

# 中心化
df_c = df - df.mean()
# print(df_c.head(3))
# print(df_c.describe())

# データの抽出
x = df_c['x']
y = df_c['y']

# x y の散布図をプロット
# plt.scatter(x,y)
# plt.show()

# パラメータaの計算
xx = x * x
xy = x * y
a = xy.sum() / xx.sum()
# print(a)

## プロットして確認
# plt.scatter(x, y, label='y') # 実測値
# plt.plot(x, a*x, label='y_hat', color='red') # 予測値
# plt.legend() # 判例の表示
# plt.show()

# 予測値の計算
x_new = 40
mean = df.mean()
# print(mean['x'])

# 中心化
xc = x_new - mean['x']
# print(xc)

# 単回帰分析による予測
yc = a * xc

# 元のスケールの予測値
y_hat = a * xc + mean['y']
# print(y_hat)

# 予測値を研鑽する関数の作成¥
def predict(x):
    # 定数項
    a = 10069.022519284063
    xm = 37.62221999999999
    ym = 121065.0
    # 中心化
    xc = x - xm
    # 予測値の計算
    y_hat = a * xc + ym
    # 出力
    return y_hat

print(predict(40))