import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


# Obtain financial data and store it in the 
def split_data_by_year(ticker='NVDA', train_end_year=2022, test_start_year=2023):
    data = yf.download(ticker, start="2020-01-01", end="2023-01-12")[["Open", "High", "Low", "Close", "Volume"]]
    data = data.fillna(method='ffill')
    data['Year'] = data.index.year
    train_data = data[data['Year'] <= train_end_year].drop(columns='Year')
    test_data = data[data['Year'] >= test_start_year].drop(columns='Year')

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)
    return scaled_train, scaled_test