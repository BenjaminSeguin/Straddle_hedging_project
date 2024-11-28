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
data_options = pd.read_csv("data_option_SP500.csv")

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

def filter_data(df):
    result = pd.DataFrame()
    selected_optionids = set()
    
    for date, group in df.groupby(df.index):
        # Separate calls and puts
        calls = group[group['cp_flag'] == 'C']
        puts = group[group['cp_flag'] == 'P']
        
        # Define maturity groups
        maturity_groups = {
            '1_day': (0, 3),
            '1_week': (4, 14),
            '1_month': (23, 37)
        }
        
        # Find the closest moneyness to 1 for each maturity group
        for group_name, (min_days, max_days) in maturity_groups.items():
            # Filter options within the maturity group
            available_calls = calls[(calls['D to Expiration'] >= min_days) & (calls['D to Expiration'] <= max_days)]
            available_puts = puts[(puts['D to Expiration'] >= min_days) & (puts['D to Expiration'] <= max_days)]
            
            if not available_calls.empty:
                # Select the call with moneyness closest to 1
                closest_call = available_calls.iloc[(available_calls['Moneyness'] - 1).abs().argsort()[:1]]
                selected_optionids.add(closest_call['optionid'].values[0])
            
            if not available_puts.empty:
                # Select the put with moneyness closest to 1
                closest_put = available_puts.iloc[(available_puts['Moneyness'] - 1).abs().argsort()[:1]]
                selected_optionids.add(closest_put['optionid'].values[0])
    
    # Include all rows related to the selected optionids
    result = df[df['optionid'].isin(selected_optionids)]
    
    return result

# Apply the filter function
filtered_data = filter_data(data_atm)

filtered_data.to_csv('filtered_data.csv')

