import pandas as pd
import numpy as np

def delta_hedging(straddles_df, option_df, market_df, transaction_cost=0):
    results = []

    # Precompute deltas for calls and puts
    call_deltas = option_df[option_df['cp_flag'] == 'C'].set_index('optionid')['delta']
    put_deltas = option_df[option_df['cp_flag'] == 'P'].set_index('optionid')['delta']

    # Align risk-free rates and prices with the full trading calendar
    full_calendar = pd.date_range(market_df.index.min(), market_df.index.max(), freq='B')
    sp_prices = market_df['Close'].reindex(full_calendar).fillna(method='ffill')
    rf_rates = market_df['RF'].reindex(full_calendar).fillna(method='ffill')

    for index, row in straddles_df.iterrows():
        strike_price = row['Strike_Price']
        call_id = row['Call_Optionid']
        put_id = row['Put_Optionid']

        # Extract delta series
        call_delta_series = call_deltas.loc[call_id]
        put_delta_series = put_deltas.loc[put_id]

        # Initial values
        initial_proceeds = row['Initial Proceeds']
        underlying_price = row['Underlying Price']
        delta0 = -(row['Call_Delta'] + row['Put_Delta'])
        initial_cash = initial_proceeds - delta0 * underlying_price

        # Initialize intermediate values
        prev_delta = delta0
        prev_prev_delta = 0  # No delta exists two periods before the start
        cash = initial_cash
        hedged_positions = []
        portfolio_values = []

        for i in range(int(row['D to Expiration'])):
            current_date = index + pd.offsets.BusinessDay(i)
            previous_date = index + pd.offsets.BusinessDay(i - 1) if i > 0 else None

            # Retrieve prices and deltas
            current_price = sp_prices.loc[current_date]  # Track the price for the current day
            rf = rf_rates.loc[current_date]
            current_call_delta = call_delta_series.get(current_date, 0)
            current_put_delta = put_delta_series.get(current_date, 0)
            curr_delta = -(current_call_delta + current_put_delta)

            # Update cash with the modified formula
            cash = (
                cash * np.exp(rf / 252) +
                (prev_delta - curr_delta) * current_price -
                transaction_cost * abs(prev_delta - prev_prev_delta) * current_price
            )
            hedged_position = curr_delta * current_price
            portfolio_value = cash + hedged_position

            # Store results
            hedged_positions.append(hedged_position)
            portfolio_values.append(portfolio_value)

            # Update deltas for the next iteration
            prev_prev_delta = prev_delta
            prev_delta = curr_delta

        # Calculate payoff for the short straddle at expiration
        payoff = max(current_price - strike_price, strike_price - current_price)

        # Adjust the final value to account for the payoff
        final_value = portfolio_values[-1] - payoff

        hedging_error = final_value**2 

        results.append({
            'Date': index,
            'Initial Value': initial_proceeds,
            'Final Cash Value': portfolio_values[-1],
            'Payoff': payoff,
            'Hedging Error': hedging_error,
            'P&L': final_value,
        })

    return pd.DataFrame(results)

def calendar_to_business_days(start_date, end_date, calendar_days):

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    # Ensure start_date is in string format
    start_date_str = start_date.strftime('%Y-%m-%d')
    # Calculate the end date
    end_date_str = end_date.strftime('%Y-%m-%d')
    # Calculate business days
    business_days = np.busday_count(start_date_str, end_date_str)
    return business_days

def create_straddles(groups):
    """
    Create a DataFrame of straddles from grouped option data.

    Parameters:
    - groups_train: Iterable of grouped data, where each group is identified by
      a tuple (date, strike_price, day_to_exp) and contains option data.

    Returns:
    - pd.DataFrame containing straddle information.
    """
    straddles = []
    
    for (date, strike_price, day_to_exp), group in groups:
        if len(group) == 2:
            call = group[group['cp_flag'] == 'C']
            put = group[group['cp_flag'] == 'P']
            if not call.empty and not put.empty:
                straddle = {
                    'Date': pd.to_datetime(date),
                    'Strike_Price': strike_price,
                    'D to Expiration': int(day_to_exp),
                    'Call_Midprice': call['Midprice'].values[0],
                    'Put_Midprice': put['Midprice'].values[0],
                    'Contract Size': call['contract_size'].values[0],
                    'Initial Proceeds': call['Midprice'].values[0] + put['Midprice'].values[0],
                    'Underlying Price': call['Close'].values[0],
                    'Call_Delta': call['delta'].values[0],
                    'Put_Delta': put['delta'].values[0],
                    'Call_Optionid': call['optionid'].values[0],
                    'Put_Optionid': put['optionid'].values[0],
                    'Moneyness': call['Moneyness'].values[0]
                }
                straddles.append(straddle)
    
    # Convert the list of straddles to a DataFrame
    straddles_df = pd.DataFrame(straddles)
    
    straddles_df['Date'] = pd.to_datetime(straddles_df['Date'])
    
    # Set 'Date' as the index
    straddles_df.set_index('Date', inplace=True)

    return straddles_df

def process_straddles(straddles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the straddles DataFrame to select the first straddle per month
    closest to 1 moneyness.

    Parameters:
    - straddles_df (pd.DataFrame): The input DataFrame with straddles.
    - dataset_name (str): Identifier for the dataset (e.g., 'train', 'test').

    Returns:
    - pd.DataFrame: The processed DataFrame with selected straddles.
    """
    # Reset the index to convert 'Date' from index to column
    straddles_reset = straddles_df.reset_index()

    # Create 'YearMonth' column for grouping
    straddles_reset['YearMonth'] = straddles_reset['Date'].dt.to_period('M')

    # Compute absolute moneyness differences
    straddles_reset['Moneyness_Diff'] = abs(straddles_reset['Moneyness'] - 1)

    # Sort by YearMonth, then by Moneyness_Diff, then by Date
    straddles_reset.sort_values(
        by=['YearMonth', 'Moneyness_Diff', 'Date'],
        inplace=True
    )

    # Select the first straddle (closest to 1 moneyness) for each month
    first_straddles_daily = straddles_reset.drop_duplicates(
        subset='YearMonth',
        keep='first'
    )

    # Set 'Date' as the index and drop unnecessary columns
    first_straddles_daily.set_index('Date', inplace=True)
    first_straddles_daily.drop(columns=['YearMonth', 'Moneyness_Diff'], inplace=True)

    return first_straddles_daily