import pandas as pd
import numpy as np

# Load the top 5 green coinswitch predictions
pred_df = pd.read_csv('top5_coinswitch_5day_predictions.csv')

# Load last 5 days actual vs predicted
try:
    last5_df = pd.read_csv('batch_last5_actual_vs_predicted.csv')
except FileNotFoundError:
    last5_df = pd.DataFrame()

# Prepare new columns
actual_cols = [f'Actual Day {i+1}' for i in range(5)]
pred_cols = [f'Predicted Day {i+1}' for i in range(5)]
pct_diff_cols = [f'Pct Diff Day {i+1}' for i in range(5)]

# Helper to get actuals and predicted for a coin
def get_last5_for_coin(coin):
    if last5_df.empty:
        return ['NA']*5, ['NA']*5, ['NA']*5
    coin_rows = last5_df[last5_df['Coin'].str.lower() == coin.lower()]
    if coin_rows.empty:
        return ['NA']*5, ['NA']*5, ['NA']*5
    actuals = coin_rows['Actual'].tolist()[:5]
    preds = coin_rows['Predicted'].tolist()[:5]
    # Pad with NA if less than 5
    actuals += ['NA']*(5-len(actuals))
    preds += ['NA']*(5-len(preds))
    pct_diff = []
    for a, p in zip(actuals, preds):
        try:
            if a == 'NA' or p == 'NA':
                pct_diff.append('NA')
            elif float(a) == 0:
                pct_diff.append('NA')
            else:
                pct_diff.append(round((float(p)-float(a))/float(a)*100, 4))
        except Exception:
            pct_diff.append('NA')
    return actuals, preds, pct_diff

# Add columns to DataFrame
for idx, row in pred_df.iterrows():
    coin = row['Coin']
    actuals, preds, pct_diff = get_last5_for_coin(coin)
    for i in range(5):
        pred_df.at[idx, actual_cols[i]] = actuals[i]
        pred_df.at[idx, pred_cols[i]] = preds[i]
        pred_df.at[idx, pct_diff_cols[i]] = pct_diff[i]

# Save updated CSV
pred_df.to_csv('top5_coinswitch_5day_predictions.csv', index=False)
print('Updated CSV with actuals, predictions, and percent difference for last 5 days.')
