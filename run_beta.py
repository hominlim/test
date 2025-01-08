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


# asset = pdr.DataReader("078930", "naver", start, end).Close.astype('float').pct_change().values[1:]
# print(asset) 

# benchmark = pdr.DataReader("KOSPI", "naver", start, end).Close.astype('float').pct_change().values[1:]
# # print(benchmark)

# 이건 선형 모델 구하는 함수, 베타계수 리턴
def linreg(x, y):
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()

    x=x[:,1]
    # return model.params[0], model.params[1]
    return model.params[1] #베타만 리턴




# # 그래프로 확인
# X = benchmark
# Y = asset
# X2 = np.linspace(X.min(), X.max(), 100)
# Y_hat = X2*beta + alpha

# plt.scatter(X, Y, alpha=0.2)
# plt.xlabel("KOSPI daily return")
# plt.ylabel("Asset daily return")
# plt.plot(X2, Y_hat, 'r', alpha=0.9)
# plt.show()



# 전체 종목 가져오기
df_kospi = fdr.StockListing('KOSPI')
df_kosdaq = fdr.StockListing('KOSDAQ')
# print(df_kospi)

asset = []
for code in df_kospi['Code']: #전체
# for code in df_kospi['Code'][0:2]: #두개 종목만
    asset.append(pdr.DataReader(code, "naver", start, end).Close.astype('float').pct_change().values[1:])
df_asset = pd.DataFrame(asset)
# print(type(df_asset))
# print(df_asset)

#결측치 처리
# df_asset.fillna(1) # 단일값으로 채우기
# df_asset.fillna(method='pad') #이전값으로 채우기
# df_asset.fillna(method='bfill') #이후값으로 채우기
# print(df_asset)

# ndaary로 변환
asset=df_asset.values
# print(type(asset))
# print(asset)

# 벤치마크 구하기
benchmark = pdr.DataReader("KOSPI", "naver", start, end).Close.astype('float').pct_change().values[1:]
# print(type(benchmark))
# print(benchmark)


# 베타계수 구하기
beta = []
for row in asset:
    beta.append(linreg(benchmark, row))

# print(type(beta))
# print(beta)
    
df_beta = pd.DataFrame({'Beta': beta})
df_beta.insert(0, 'Name', df_kospi['Name'])
df_beta.insert(1, 'Code', df_kospi['Code'])
print(df_beta)