import pandas_datareader as pdr
import FinanceDataReader as fdr
import pandas as pd
from datetime import datetime, timedelta
from statsmodels import regression
import statsmodels.api as sm
import time


end = datetime.today()
start = end-timedelta(days=365)


# 선형 모델 구하는 함수, 베타계수 리턴
def linreg(x, y):
    x = sm.add_constant(x)
    model = regression.linear_model.OLS(y, x).fit()

    x=x[:,1]
    # return model.params[0], model.params[1]
    return model.params[1] #베타만 리턴


# 전체 종목 가져오기
df_kospi = fdr.StockListing('KOSPI')
df_kosdaq = fdr.StockListing('KOSDAQ')

df_stock = df_kospi

asset = []
for code in df_stock['Code']: #전체
    asset.append(pdr.DataReader(code, "naver", start, end).Close.astype('float').pct_change().values[1:])
    

# #결측치 처리
df_asset = pd.DataFrame(asset)
df_asset.fillna(method='ffill') #이전값으로 채우기
# df_asset.fillna(method='bfill') #이후값으로 채우기
# df_asset.fillna(1) # 단일값으로 채우기
asset=df_asset.values # ndaary로 변환

# 벤치마크 구하기
benchmark = pdr.DataReader("KOSPI", "naver", start, end).Close.astype('float').pct_change().values[1:]


# 베타계수 구하기
beta = []
for row in asset:
    beta.append(linreg(benchmark, row))

# 출력내용 정리    
df_beta = pd.DataFrame({'Beta': beta})
df_beta.insert(0, 'Name', df_stock['Name'])
df_beta.insert(1, 'Code', df_stock['Code'])
print(df_beta)