import pandas as pd
import numpy as np

# Load the full batch predictions CSV
results_df = pd.read_csv('batch_5day_predictions.csv')

# Load coinswitch coin IDs
coinswitch_mapping = pd.read_csv('coinswitch_coin_mapping.csv')
coinswitch_coin_ids = set(coinswitch_mapping['coin_id'].dropna().str.lower())

# Filter for coinswitch coins
filtered_df = results_df[results_df['Coin'].str.lower().isin(coinswitch_coin_ids)]

# Convert percent change to float and filter for positive change
filtered_df['5-Day % Change'] = filtered_df['5-Day % Change'].str.replace('%','').astype(float)
filtered_df = filtered_df[filtered_df['5-Day % Change'] > 0]

# Sort and keep top 5
filtered_df = filtered_df.sort_values('5-Day % Change', ascending=False).head(5)

# Load last 5 days actual vs predicted
try:
    last5_df = pd.read_csv('batch_last5_actual_vs_predicted.csv')
except FileNotFoundError:
    last5_df = pd.DataFrame()

# Prepare new summary columns with date in the heading
summary_cols = []
for i in range(5):
    # For each day, get the date for the first coin (if available)
    date_heading = 'NA'
    for idx, row in filtered_df.iterrows():
        coin = row['Coin']
        if not last5_df.empty and not last5_df[last5_df['Coin'].str.lower() == coin.lower()].empty:
            coin_rows = last5_df[last5_df['Coin'].str.lower() == coin.lower()]
            if len(coin_rows) > i:
                date_heading = coin_rows.iloc[i]['Date'] if 'Date' in coin_rows.columns else 'NA'
                break
    summary_col = f'Day {i+1} Summary ({date_heading})'
    summary = []
    for idx, row in filtered_df.iterrows():
        coin = row['Coin']
        if last5_df.empty or last5_df[last5_df['Coin'].str.lower() == coin.lower()].empty:
            summary.append('NA')
        else:
            coin_rows = last5_df[last5_df['Coin'].str.lower() == coin.lower()]
            if len(coin_rows) > i:
                actual = coin_rows.iloc[i]['Actual']
                pred = coin_rows.iloc[i]['Predicted']
                pct = coin_rows.iloc[i]['pct_diff_pred_vs_actual'] if 'pct_diff_pred_vs_actual' in coin_rows.columns else 'NA'
                summary.append(f"A:{actual} | P:{pred} | %:{pct}")
            else:
                summary.append('NA')
    filtered_df[summary_col] = summary
    summary_cols.append(summary_col)

# Keep only relevant columns for output
output_cols = ['Coin', '5-Day Predicted Prices', '5-Day % Change'] + summary_cols
filtered_df[output_cols].to_csv('top5_coinswitch_5day_predictions.csv', index=False)
print('Filtered CSV written to top5_coinswitch_5day_predictions.csv with summary columns for each day (date in heading).')
