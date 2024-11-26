import pandas as pd
import numpy as np
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy.optimize import minimize

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
                transaction_cost * abs(prev_delta - curr_delta) * current_price
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
                    'Moneyness': call['Moneyness'].values[0],
                    'Expiration Date': call['exdate'].values[0]
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

def dp_hedging_collect_training_data(straddles_monthly, option_df, real_market_df, transaction_cost=0):
    """
    Perform dynamic programming hedging on the monthly straddles and collect training data.

    Parameters:
    - first_straddles_monthly: DataFrame containing the monthly straddles to hedge.
    - option_df: DataFrame containing daily option data.
    - real_market_df: DataFrame containing daily underlying price and risk-free rate data.
    - transaction_cost: Transaction cost per unit traded.

    Returns:
    - training_data_df: DataFrame with state variables and optimal phi for each day.
    """
    training_data = []

    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    for index, row in straddles_monthly.iterrows():
        # Extract straddle information
        strike_price = row['Strike_Price']
        call_optionid = row['Call_Optionid']
        put_optionid = row['Put_Optionid']
        initial_proceeds = row['Initial Proceeds']
        business_days_to_expiration = int(row['D to Expiration'])
        start_date = pd.to_datetime(index)
        expiration_date = pd.to_datetime(row['Expiration Date'])

        # Get business dates between start_date and expiration_date (inclusive)
        business_dates = pd.date_range(start=start_date, end=expiration_date, freq=us_bd)

        # Initialize variables
        cash = initial_proceeds
        prev_phi = 0

        for t in range(len(business_dates) - 1):
            current_date = pd.to_datetime(business_dates[t])
            next_date = pd.to_datetime(business_dates[t + 1])

            # Access market data directly
            try:
                S_t = real_market_df.loc[current_date, 'Close']
                S_t1 = real_market_df.loc[next_date, 'Close']
                rf = real_market_df.loc[current_date, 'RF']
            except KeyError:
                continue  # Skip if market data is missing

            # Retrieve option data for current_date
            call_data = option_df[(option_df['optionid'] == call_optionid) & (option_df.index == current_date)]
            put_data = option_df[(option_df['optionid'] == put_optionid) & (option_df.index == current_date)]

            if call_data.empty or put_data.empty:
                continue  # Skip if option data is missing

            # Extract option prices
            call_price = call_data['Midprice'].values[0]
            put_price = put_data['Midprice'].values[0]
            straddle_price_t = call_price + put_price

            # Retrieve implied volatilities
            call_iv = call_data['impl_volatility'].values[0]
            put_iv = put_data['impl_volatility'].values[0]
            straddle_volatility = (call_iv + put_iv) / 2

            # Retrieve deltas and gammas
            delta_call = call_data['delta'].values[0]
            gamma_call = call_data['gamma'].values[0]
            delta_put = put_data['delta'].values[0]
            gamma_put = put_data['gamma'].values[0]
            straddle_delta = -(delta_call + delta_put)
            straddle_gamma = -(gamma_call + gamma_put)

            # Calculate time to expiration in business days
            time_to_expiration = np.busday_count(current_date.date(), expiration_date.date())

            # Ensure time_to_expiration is at least 1
            time_to_expiration = max(time_to_expiration, 1)

            # Retrieve moneyness
            moneyness = call_data['Moneyness'].values[0]

            # Define state variables
            state_variables = {
                'Date': current_date,
                'Moneyness': moneyness,
                'TimeToExpiration': time_to_expiration,
                'Volatility': straddle_volatility,
                'StraddleDelta': straddle_delta,
                'StraddleGamma': straddle_gamma,
                'CallDelta': delta_call,
                'CallGamma': gamma_call,
                'PutDelta': delta_put,
                'PutGamma': gamma_put,
                'StrikePrice': strike_price,
                'UnderlyingPrice': S_t,
                'RiskFreeRate': rf,
                'CallVega': call_data['vega'].values[0],
                'PutVega': put_data['vega'].values[0],
                'CallTheta': call_data['theta'].values[0],
                'PutTheta': put_data['theta'].values[0],
                'StraddleTheta': -(call_data['theta'].values[0] + put_data['theta'].values[0]),
                'StraddleVega': -(call_data['vega'].values[0] + put_data['vega'].values[0]),
            }

            # Define objective function for dynamic programming
            def objective(phi_t):
                # Update cash with hedge change
                phi_change = phi_t - prev_phi
                cash_t1 = cash * np.exp(rf / 252) + phi_change * S_t - transaction_cost * abs(phi_change) * S_t
                # Portfolio value at next time step
                portfolio_value_t1 = cash_t1 + phi_t * S_t1

                # Retrieve option data for next_date
                call_data_next = option_df[(option_df['optionid'] == call_optionid) & (option_df.index == next_date)]
                put_data_next = option_df[(option_df['optionid'] == put_optionid) & (option_df.index == next_date)]

                if call_data_next.empty or put_data_next.empty:
                    # Estimate straddle price change using Taylor expansion
                    dS = S_t1 - S_t
                    straddle_price_t1 = straddle_price_t + straddle_delta * dS + 0.5 * straddle_gamma * dS ** 2
                else:
                    call_price_next = call_data_next['Midprice'].values[0]
                    put_price_next = put_data_next['Midprice'].values[0]
                    straddle_price_t1 = call_price_next + put_price_next

                # Hedging error (squared difference)
                error = (portfolio_value_t1 - straddle_price_t1) ** 2
                return error

            # Solve for optimal phi_t
            res = minimize(objective, x0=prev_phi, bounds=[(-10, 10)])
            phi_t_opt = res.x[0]

            # Store state variables and optimal phi
            training_data.append({
                'Phi': phi_t_opt,
                **state_variables
            })

            # Update cash and prev_phi for next iteration
            phi_change = phi_t_opt - prev_phi
            cash = cash * np.exp(rf / 252) + phi_change * S_t - transaction_cost * abs(phi_change) * S_t
            prev_phi = phi_t_opt

    # Convert training data to DataFrame
    training_data_df = pd.DataFrame(training_data)
    return training_data_df

def train_random_forest_model(training_data_df):
    # Define features and target
    X = training_data_df.drop(columns=['Phi'])
    y = training_data_df['Phi']

    # Initialize the Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    # Get the best model
    best_rf = grid_search.best_estimator_

    print(f"Best parameters: {grid_search.best_params_}")
    return best_rf

def apply_hedging_model(first_straddles_monthly, option_df, real_market_df, model, transaction_cost=0):
    """
    Apply the trained hedging model to the test set and calculate P&L for each straddle.

    Parameters:
    - first_straddles_monthly: DataFrame containing the monthly straddles to hedge.
    - option_df: DataFrame containing daily option data.
    - real_market_df: DataFrame containing daily underlying price and risk-free rate data.
    - model: Trained regression model that predicts phi from state variables.
    - transaction_cost: Transaction cost per unit traded.

    Returns:
    - results_df: DataFrame with hedging results for each straddle.
    """
    results = []

    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    for index, row in first_straddles_monthly.iterrows():
        # Convert start_date and expiration_date to datetime
        start_date = pd.to_datetime(index)
        expiration_date = pd.to_datetime(row['Expiration Date'])

        # Extract straddle information
        strike_price = row['Strike_Price']
        call_optionid = row['Call_Optionid']
        put_optionid = row['Put_Optionid']
        initial_proceeds = row['Initial Proceeds']

        # Get business dates between start_date and expiration_date (inclusive)
        business_dates = pd.date_range(start=start_date, end=expiration_date, freq=us_bd)

        # Initialize variables
        cash = initial_proceeds  # Initial proceeds from selling the straddle
        prev_phi = 0

        # Initialize a list to store daily results
        daily_results = []

        for t in range(len(business_dates) - 1):
            current_date = business_dates[t]
            next_date = business_dates[t + 1]

            # Ensure current_date is a datetime object
            current_date = pd.to_datetime(current_date)

            # Access market data directly
            try:
                S_t = real_market_df.loc[current_date, 'Close']
                S_t1 = real_market_df.loc[next_date, 'Close']
                rf = real_market_df.loc[current_date, 'RF']
            except KeyError:
                continue  # Skip if market data is missing

            # Retrieve option data for current_date
            call_data = option_df[(option_df['optionid'] == call_optionid) & (option_df.index == current_date)]
            put_data = option_df[(option_df['optionid'] == put_optionid) & (option_df.index == current_date)]

            if call_data.empty or put_data.empty:
                continue  # Skip if option data is missing

            # Extract option prices
            call_price = call_data['Midprice'].values[0]
            put_price = put_data['Midprice'].values[0]
            straddle_price_t = call_price + put_price

            # Retrieve implied volatilities
            call_iv = call_data['impl_volatility'].values[0]
            put_iv = put_data['impl_volatility'].values[0]
            straddle_volatility = (call_iv + put_iv) / 2

            # Retrieve deltas and gammas
            delta_call = call_data['delta'].values[0]
            gamma_call = call_data['gamma'].values[0]
            delta_put = put_data['delta'].values[0]
            gamma_put = put_data['gamma'].values[0]
            straddle_delta = delta_call + delta_put
            straddle_gamma = gamma_call + gamma_put

            # Calculate time to expiration in business days
            time_to_expiration = np.busday_count(current_date, expiration_date)
            time_to_expiration = max(time_to_expiration, 1)

            # Calculate moneyness
            moneyness = S_t / strike_price

            # Define state variables
            state_variables = {
                'Moneyness': moneyness,
                'TimeToExpiration': time_to_expiration,
                'Volatility': straddle_volatility,
                'StraddleDelta': straddle_delta,
                'StraddleGamma': straddle_gamma,
                'CallDelta': delta_call,
                'CallGamma': gamma_call,
                'PutDelta': delta_put,
                'PutGamma': gamma_put,
                'StrikePrice': strike_price,
                'UnderlyingPrice': S_t,
                'RiskFreeRate': rf,
            }

            # Convert state variables to DataFrame
            state_df = pd.DataFrame([state_variables])

            # Predict phi_t using the model
            phi_t = model.predict(state_df)[0]

            # Update cash with hedge change
            phi_change = phi_t - prev_phi
            cash = cash * np.exp(rf / 252) + phi_change * S_t - transaction_cost * abs(phi_change) * S_t

            # Store daily results
            daily_results.append({
                'Date': current_date,
                'Cash': cash,
                'Phi': phi_t,
                'UnderlyingPrice': S_t,
                'StraddlePrice': straddle_price_t,
                **state_variables
            })

            # Update prev_phi for next iteration
            prev_phi = phi_t

        # At expiration
        final_date = business_dates[-1]
        try:
            S_T = real_market_df.loc[final_date, 'Close']
        except KeyError:
            continue  # Skip if market data is missing

        # Calculate straddle payoff at expiration (short straddle)
        straddle_payoff = - (max(S_T - strike_price, 0) + max(strike_price - S_T, 0))  # Negative because it's a short position

        # Final portfolio value
        portfolio_value_T = cash * np.exp(rf / 252) + prev_phi * S_T  # Update cash one last time

        # P&L calculation
        pnl = portfolio_value_T + straddle_payoff  # Total P&L from hedging and straddle payoff

        # Hedging error
        hedging_error = (portfolio_value_T + straddle_payoff) ** 2

        # Store results
        results.append({
            'StartDate': start_date,
            'ExpirationDate': expiration_date,
            'Initial Proceeds': initial_proceeds,
            'Final Portfolio Value': portfolio_value_T,
            'Straddle Payoff': straddle_payoff,
            'Hedging Error': hedging_error,
            'P&L': pnl
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    return results_df

def select_atm_options(df, maturity_days):
    """
    Select ATM options (moneyness closest to 1) for the specified maturity.
    """
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Filter options with the specified maturity
    filtered = df[df['D to Expiration'] == maturity_days].copy()
    
    if filtered.empty:
        return pd.DataFrame()
        
    # Compute absolute moneyness difference
    filtered['Moneyness_Diff'] = abs(filtered['Moneyness'] - 1)
    
    # Sort by moneyness difference and group by Date and cp_flag
    atm_options = (
        filtered.sort_values(by=['Date', 'cp_flag', 'Moneyness_Diff'])
        .groupby(['Date', 'cp_flag'])
        .first()
        .reset_index()
    )
    
    # Add expiration date
    atm_options['Expiration_Date'] = atm_options['Date'] + pd.to_timedelta(atm_options['D to Expiration'], unit='D')
    
    return atm_options[['Date', 'cp_flag', 'optionid', 'strike_price', 'D to Expiration', 'Expiration_Date']]

def create_straddle_tracking_dataset(df):
    """
    Create a dataset to track selected ATM options at initiation.
    """
    # Reset index to make Date a column and create a copy
    df = df.reset_index().copy()
    
    # Sort the DataFrame
    df = df.sort_values(by=['Date', 'cp_flag', 'Moneyness'])
    
    # Select ATM options for different maturities
    atm_1d = select_atm_options(df, maturity_days=2)
    atm_1w = select_atm_options(df, maturity_days=7)
    atm_1m = select_atm_options(df, maturity_days=30)
    
    # Combine all selected options
    atm_options = pd.concat([atm_1d, atm_1w, atm_1m], ignore_index=True)
    if atm_options.empty:
        print("No ATM options found.")
        return pd.DataFrame()
        
    atm_options['Maturity'] = atm_options['D to Expiration'].map({2: '1D', 7: '1W', 30: '1M'})
    
    # Return the selected ATM options at initiation
    return atm_options

def compute_straddle_returns(atm_options, data):
    """
    Compute delta-hedged straddle returns using the raw data for price history.

    Parameters:
    - atm_options (pd.DataFrame): DataFrame containing selected ATM options at initiation.
    - data (pd.DataFrame): Raw unfiltered dataset containing options data.

    Returns:
    - pd.DataFrame: A DataFrame containing straddle returns.
    """
    # Ensure the raw data has Date as a column
    data = data.reset_index()
    
    # Ensure Date columns are of datetime type
    atm_options['Date'] = pd.to_datetime(atm_options['Date'])
    atm_options['Expiration_Date'] = pd.to_datetime(atm_options['Expiration_Date'])
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Initialize a list to store computed straddle returns
    straddle_returns = []
    
    # Group the ATM options by initiation Date and Maturity
    grouped = atm_options.groupby(['Date', 'Maturity'])
    
    for (init_date, maturity), group in grouped:
        print(f"Processing initiation Date: {init_date.date()}, Maturity: {maturity}")
        
        # Get call and put option IDs
        call_optionid = group[group['cp_flag'] == 'C']['optionid'].values
        put_optionid = group[group['cp_flag'] == 'P']['optionid'].values
        
        if len(call_optionid) == 0 or len(put_optionid) == 0:
            print(f"Missing call or put for Date: {init_date.date()}, Maturity: {maturity}. Skipping.")
            continue
        
        call_optionid = call_optionid[0]
        put_optionid = put_optionid[0]
        
        expiration_date = group['Expiration_Date'].iloc[0]
        
        # Get initial data for call and put from the raw data
        call_init = data[(data['Date'] == init_date) & (data['optionid'] == call_optionid)]
        put_init = data[(data['Date'] == init_date) & (data['optionid'] == put_optionid)]
        
        if call_init.empty or put_init.empty:
            print(f"Initial data missing for call or put on Date: {init_date.date()}, Maturity: {maturity}. Skipping.")
            continue
        
        call_init = call_init.iloc[0]
        put_init = put_init.iloc[0]
        
        # Use initial delta to compute hedge ratio
        try:
            hedge_ratio = call_init['delta'] / (1 - call_init['delta'])
        except ZeroDivisionError:
            print(f"ZeroDivisionError for Date: {init_date.date()}, Maturity: {maturity}. Skipping.")
            continue
        
        # Get price history for the call and put options from raw data
        call_data = data[(data['optionid'] == call_optionid) & 
                         (data['Date'] >= init_date) & 
                         (data['Date'] <= expiration_date)]
        put_data = data[(data['optionid'] == put_optionid) & 
                        (data['Date'] >= init_date) & 
                        (data['Date'] <= expiration_date)]
        
        if call_data.empty or put_data.empty:
            print(f"No price history found for call or put options from Date: {init_date.date()} to {expiration_date.date()}. Skipping.")
            continue
        
        # Get the last available date before expiration with data
        last_date = min(call_data['Date'].max(), put_data['Date'].max())
        if pd.isnull(last_date) or last_date == init_date:
            print(f"No data available after initiation for Date: {init_date.date()}, Maturity: {maturity}. Skipping.")
            continue
        
        # Get prices at initiation and at last_date
        call_price_init = call_init['Close']
        put_price_init = put_init['Close']
        
        call_price_close = call_data[call_data['Date'] == last_date]['Close'].values[0]
        put_price_close = put_data[put_data['Date'] == last_date]['Close'].values[0]
        
        # Calculate straddle values at initiation and close
        straddle_value_init = call_price_init + hedge_ratio * put_price_init
        straddle_value_close = call_price_close + hedge_ratio * put_price_close
        
        # Compute return
        ret = (straddle_value_close - straddle_value_init) / straddle_value_init
        
        # Append result
        straddle_returns.append({
            'Initiation_Date': init_date,
            'Maturity': maturity,
            'Expiration_Date': expiration_date,
            'Return': ret
        })
    
    # Create a DataFrame from the calculated returns
    result_df = pd.DataFrame(straddle_returns)
    print("Final straddle returns DataFrame:")
    print(result_df.head())
    
    return result_df
