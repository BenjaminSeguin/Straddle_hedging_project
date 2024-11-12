import pandas as pd
import yfinance as yf

sp = yf.download('^SPX')

sp.index = pd.to_datetime(sp.index)

# Filter the DataFrame
start_date = '2000-01-01'
end_date = '2023-08-31'
filtered_sp = sp.loc[start_date:end_date]

# Remove the 'Ticker' level from the MultiIndex
filtered_sp.columns = filtered_sp.columns.droplevel(1)
filtered_sp = pd.DataFrame(filtered_sp['Close'], index=filtered_sp.index)
filtered_sp.index = pd.to_datetime(filtered_sp.index).tz_localize(None)
filtered_sp

# Read option data from WRDS
data_options = pd.read_csv("data_option_SP.csv")

data_options['date'] = pd.to_datetime(data_options['date'])
data_options['exdate'] = pd.to_datetime(data_options['exdate'])

# Create days to expiration (calendar)
data_options['D to Expiration'] = data_options['exdate'] - data_options['date']
data_options['D to Expiration'] = data_options['D to Expiration'].dt.days

# Drop unnecessary columns 
data_options.drop(columns=['last_date', 'issuer', 'exercise_style', 'forward_price', 'index_flag'], inplace=True)
data_options.set_index('date', inplace=True)
data_options['strike_price'] = data_options['strike_price'] / 1000

# Merge both dataframes to have the underlying price
full_data = data_options.merge(filtered_sp, left_index=True, right_index=True)

# Compute moneyness, if >1, Put ITM and call OTM
full_data['Moneyness'] = full_data['strike_price'] / full_data['Close']

# Filter options to have max 40 days to expiration (a bit more than 1 month) and moneyness between 0.9 and 1.1 ~ATM
mask_money = (full_data['Moneyness'] > 0.9) & (full_data['Moneyness'] < 1.1)
mask_time = (full_data['D to Expiration'] < 40)
data_atm = full_data[mask_money & mask_time]

# data_atm.to_csv("cleaned_data.csv", index=True)

