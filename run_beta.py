import pandas_datareader as pdr
import FinanceDataReader as fdr
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from statsmodels import regression
import statsmodels.api as sm
import numpy as np


start = datetime.datetime(2024, 1, 7)
end = datetime.datetime(2025, 1, 7)

# df = fdr.DataReader("078930", start, end)
# print(df)

# 아래는 차근차근 설명
# df_stk = pdr.DataReader("078930", "naver", start, end)
# df_stk['Close']=df_stk['Close'].astype('int') # 'Close' 종가 int형으로 변환
# df_stk['Open']=df_stk['Open'].astype('int') # 'Close' 종가 int형으로 변환
# # print(df.dtypes)
# df_stk['Return'] = df_stk['Close'].diff() # 금일수익 = 금일종가 - 전일종가
# df_stk['ReturnRate'] = df_stk['Return']/df_stk['Open'] #수익률 = 금일수익/금일시작가
# df_stk['PctChange'] = df_stk['Close'].pct_change() # 변화율 (위 두줄 한번에)
# asset = df_stk.PctChange.values
# 위 전체를 한줄에

asset = pdr.DataReader("078930", "naver", start, end).Close.astype('float').pct_change().values[1:]
# print(asset) 

benchmark = pdr.DataReader("KOSPI", "naver", start, end).Close.astype('float').pct_change().values[1:]
# print(benchmark)

# 이건 선형 모델 구하는 함수
def linreg(x, y):
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()

    x=x[:,1]
    return model.params[0], model.params[1]

alpha, beta = linreg(benchmark, asset)
print(alpha)
print(beta)


# 그래프로 확인
X = benchmark
Y = asset
X2 = np.linspace(X.min(), X.max(), 100)
Y_hat = X2*beta + alpha

plt.scatter(X, Y, alpha=0.2)
plt.xlabel("KOSPI daily return")
plt.ylabel("Asset daily return")
plt.plot(X2, Y_hat, 'r', alpha=0.9)
plt.show()