## 実験データの処理

各種実験（`piControl`，`abrupt-2xCO2`，`abrupt-4xCO2`，`1pctCO2`，`historical`，`ssp119`，`ssp245`，`ssp370`，`ssp460`，`ssp585`）のそれぞれについて，
必要な変数（`tas`，`rsdt`，`rsut`，`rlut`，`areacella`）のデータ（例えばMIROC6であれば，`data_raw/CMIP6/required.txt`に挙げたもの）を[ESGF](https://esgf.llnl.gov/)からダウンロードておく．
これを全て`data_raw/CMIP6`に置く．
面積データ（`areacella*.nc`）は必ずしも必要でない．

全球年平均の時系列データ（csvファイル）の作成：
```
python process_cmip_data.py [--model_id MIROC6]
```
引数の`model_id`はダウンロードしたデータに応じて適宜変更（デフォルトは`MIROC6`）．

## 実験データのプロット

コントロール実験，倍増実験，漸増実験：
```
python plot_experiment_tas.py [--model_id MIROC6]
```
![気候モデル（MIROC6）の実験データ](output/fig_plot_experiment_tas.svg)

歴史実験：

```
python plot_historical_tas.py [--model_id MIROC6]
```
![気候モデルによる歴史実験](output/fig_plot_historical_tas.svg)

シナリオ実験：
```
python plot_scenario_tas.py [--model_id MIROC6]
```
![シナリオ実験](output/fig_plot_scenario_tas.svg)

## エミュレータのカリブレーション

2層のボックスモデルのカリブレーション（`abrupt-4xCO2`に基づく）：
```
Rscript calibrate_emulator.r [MIROC6]
```
推定結果は`output`フォルダに保存される．

## エミュレータの性能評価

カリブレイトされたエミュレータの内的妥当性（`abrupt-4xCO2`）と
外的妥当性（`historical`，`ssp119`，`ssp245`，`ssp370`，`ssp460`，`ssp585`）を評価：
```
python evaluate_emulator.py [--model_id MIROC6]
```
![エミュレータのカリブレーションと性能評価](output/fig_evaluate_emulator.svg)

## データソース

- Osborn, T.J., Jones, P.D., Lister, D.H., Morice, C.P., Simpson, I.R., Winn, J.P., Hogan, E., and Harris, I.C., (2021). Land surface air temperature variations across the globe updated to 2019: the CRUTEM5 dataset. Journal of Geophysical Research: Atmospheres. 126, e2019JD032352, https://doi.org/10.1029/2019JD032352
- Tatebe, Hiroaki; Watanabe, Masahiro (2018). MIROC MIROC6 model output prepared for CMIP6 CMIP.Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.881
- Meinshausen, M., Nicholls, Z. R. J., Lewis, J., Gidden, M. J., Vogel, E., Freund, M., Beyerle, U., Gessner, C., Nauels, A., Bauer, N., Canadell, J. G., Daniel, J. S., John, A., Krummel, P. B., Luderer, G., Meinshausen, N., Montzka, S. A., Rayner, P. J., Reimann, S., Smith, S. J., van den Berg, M., Velders, G. J. M., Vollmer, M. K., and Wang, R. H. J. (2020). The shared socio-economic pathway (SSP) greenhouse gas concentrations and their extensions to 2500, Geoscientific Model Development, 13, 3571–3605, https://doi.org/10.5194/gmd-13-3571-2020
- Smith, Christopher J. (2019, October 21). Effective Radiative Forcing from Shared Socioeconomic Pathways (Version v0.3.1). Zenodo. http://doi.org/10.5281/zenodo.3515339
- Bauer, M. D., & Rudebusch, G. D. (2023). The rising cost of climate change: Evidence from the bond market. The Review of Economics and Statistics, 105(5), 1255–1270. https://doi.org/10.1162/rest_a_01109
- Rennert, K., Errickson, F., Prest, B. C., Rennels, L., Newell, R. G., Pizer, W., Kingdon, C., Wingenroth, J., Cooke, R., Parthum, B., Smith, D., Cromar, K., Diaz, D., Moore, F. C., Müller, U. K., Plevin, R. J., Raftery, A. E., Ševčíková, H., Sheets, H., … Anthoff, D. (2022). Comprehensive evidence implies a higher social cost of CO2. Nature, 610(7933), 687–692. https://doi.org/10.1038/s41586-022-05224-9
- Joos, F., Roth, R., Fuglestvedt, J. S., Peters, G. P., Enting, I. G., Bloh, W. von, Brovkin, V., Burke, E. J., Eby, M., Edwards, N. R., & Others. (2013). Carbon dioxide and climate impulse response functions for the computation of greenhouse gas metrics: a multi-model analysis. Atmospheric Chemistry and Physics, 13(5), 2793–2825. https://acp.copernicus.org/articles/13/2793/2013/
- MPD version 2023: Bolt, Jutta and Jan Luiten van Zanden (2024). Maddison style estimates of the evolution of the world economy: A new 2023 update, Journal of Economic Surveys, 1–41. http://doi.org/10.1111/joes.12618
- World Bank. Gross savings (% of GDP), World Development Indicators, The World Bank Group, https://data.worldbank.org/indicator/NY.GNS.ICTR.ZS
- [Noto Font](https://fonts.google.com/noto) is licensed under the SIL Open Font License, Version 1.1. Copyright 2012 Google Inc. All Rights Reserved. This license is available at: http://scripts.sil.org/OFL
