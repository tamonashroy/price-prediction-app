import pandas as pd

# Load the top 5 green coinswitch predictions
pred_df = pd.read_csv('top5_coinswitch_5day_predictions.csv')

# Combine actual, predicted, and pct diff for each day into a single summary column
summary_cols = []
for i in range(5):
    actual_col = f'Actual Day {i+1}'
    pred_col = f'Predicted Day {i+1}'
    pct_col = f'Pct Diff Day {i+1}'
    summary = []
    for idx, row in pred_df.iterrows():
        actual = row[actual_col]
        pred = row[pred_col]
        pct = row[pct_col]
        summary.append(f"A:{actual} | P:{pred} | %:{pct}")
    pred_df[f'Day {i+1} Summary'] = summary
    summary_cols.append(f'Day {i+1} Summary')

# Keep only relevant columns for output
output_cols = ['Coin', '5-Day Predicted Prices', '5-Day % Change'] + summary_cols
out_df = pred_df[output_cols]
out_df.to_csv('top5_coinswitch_5day_predictions_summary.csv', index=False)
print('CSV with combined actual, predicted, and percent diff per day written to top5_coinswitch_5day_predictions_summary.csv')
