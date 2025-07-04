# Merton_model
Gluonts.pyを実行するとMertonModelで予測した結果をEvaluatorが評価してくれる。
dataset_name = "exchange_rate_nips"
を変更すると違うデータセットで予測してほしい(未確認)


{'MSE': 0.0005130783260129117, 
'abs_error': 16.44174441602081,
'abs_target_sum': 975.9766580164433,
'abs_target_mean': 0.8133138816803693,
'seasonal_error': nan,
'MASE': nan,
'MAPE': 0.027041835654526947,
'sMAPE': 0.025954026480515795,
'MSIS': nan,
'num_masked_target_values': 0.0,
'mean_absolute_QuantileLoss': 14.994998814682994, 
'mean_wQuantileLoss': 0.0153640957409356, 
'MAE_Coverage': 0.4758333333333334, 
'OWA': nan}

mean_wQuantileLossがCRPSと同じ評価指標
3回の結果 {0.0153640957409356, 0.01552779726216514, 0.01553300482575982}
mean 0.01547496594, std 0.00007842589
