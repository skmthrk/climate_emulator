# Climate emulator and its applications to economic analysis

## Date processing

Download CMIP raw data (`tas`，`rsdt`，`rsut`，`rlut`，`areacella`) for relevant experiments (`piControl`，`abrupt-2xCO2`，`abrupt-4xCO2`，`1pctCO2`，`historical`，`ssp119`，`ssp245`，`ssp370`，`ssp460`，`ssp585`)
from [ESGF](https://esgf.llnl.gov/) and store them at `./data_raw/CMIP6`.

Aggregate temperature and radiation data and generate global-mean time series (.csv files):
```
python process_cmip_data.py [--model_id MIROC6]
```
where `model_id` is defaulted at `MIROC6`.

Retrieve raw CMIP data for surface air temperature (`tas`), incoming shortwave radiation (`rsdt`), outgoing shortwave radiation (`rsut`),
outgoing longwave radiation (`rlut`), and grid cell area (`areacella`) from the Earth System Grid Federation ([https://esgf.llnl.gov/](https://esgf.llnl.gov/)).
Download data associated with the following experiments: `piControl`, `abrupt-2xCO2`, `abrupt-4xCO2`, `1pctCO2`, `historical`, `ssp119`, `ssp245`, `ssp370`, `ssp460`, and `ssp585`.
Save the downloaded files to the `./data_raw/CMIP6` directory.

Next, process the temperature and radiation data to generate global-mean time series, outputting the results as .csv files. This can be done using the provided Python script:
```
python process_cmip_data.py [--model_id MIROC6]
```
The `--model_id` argument allows for specifying a particular climate model, with `MIROC6` as the default.

## Plotting experiment data

After preprocessing the CMIP data, visualizations can be generated to examine the results. For example:

```python
plot_experiment_tas.py [--model_id MIROC6]
```

![piControl, abrupt-4xCO2, abrupt-2xCO2, 1pctCO2](output/fig_plot_experiment_tas.svg)

```python
plot_historical_tas.py [--model_id MIROC6]
```

![historical](output/fig_plot_historical_tas.svg)

```python
plot_scenario_tas.py [--model_id MIROC6]
```

![ssp scenarios](output/fig_plot_scenario_tas.svg)


## Climate emulator

Calibrate a two-layer energy balance model based on `abrupt-4xCO2` experiment:
```
Rscript calibrate_emulator.r [MIROC6]
```
推定結果は`output`フォルダに保存される．

カリブレイトされたエミュレータの内的妥当性（`abrupt-4xCO2`）と
外的妥当性（`historical`，`ssp119`，`ssp245`，`ssp370`，`ssp460`，`ssp585`）を評価：
```
python evaluate_emulator.py [--model_id MIROC6]
```
![エミュレータのカリブレーションと性能評価](output/fig_evaluate_emulator.svg)

## 物質循環

Joos et al. (2013)の実験結果（`PI100`，`PD100`，`PI5000`）を用いて線形炭素循環モデルのカリブレーション：
```
python calibrate_co2_cycle_linear.py
```
![線形炭素循環モデルのカリブレーション](output/fig_co2_cycle_linear.svg)

線形モデルのパラメタをベースラインとして，
フィードバック効果を考慮した非線形モデルに拡張する．
非線形性の効果を識別できるだけの十分な実験データがないので，
RCMIPで用意されたSSPシナリオのデータにフィットさせる．
```
python calibrate_co2_cycle_nonlinear.py
```
![非線形炭素循環モデルのカリブレーション](output/fig_co2_cycle.svg)

メタンと亜酸化窒素については線形モデルを用いる：
```
python calibrate_gas_cycle.py
```
![メタン・亜酸化窒素循環モデルのカリブレーション](output/fig_gas_cycle.svg)

## 放射強制力

二酸化炭素，メタン，亜酸化窒素の放射強制力モデルをカリブレーション：
```
python calibrate_forcing.py
```
![放射強制力モデルのカリブレーション](output/fig_forcing.svg)

## RFF将来予測

[RFF社会経済予測](https://zenodo.org/records/6016583)をダウンロードし，
`pop_income`と`emissions`を`data_raw/RFF`に置く．

GDPと人口について，
過去データと将来予測を結合したcsvファイルをサンプルごとに作成．
```
python process_gdp_pop_data.py
```

排出量についても，
同様に過去データと将来予測を結合したcsvファイルをサンプルごとに作成．
```
python process_emission_data.py
```

作成したデータをプロット．
```
python plot_rff_projections.py
```
![RFF社会経済予測](output/fig_rff_projections.svg)

## 貯蓄率

[World Bank](https://data.worldbank.org/indicator/NY.GNS.ICTR.ZS)から貯蓄率データをダウンロードし，
`data_raw/gross_savings.csv`として保存．

```
python calibrate_savings_rate.py
```
![貯蓄率](output/fig_savings_rate.svg)

## 選好パラメタ

[Michael Bauerの個人サイト](https://www.michaeldbauer.com/)から
[Bauer and Rudebusch (2023, REStat)](https://doi.org/10.1162/rest_a_01109)のレプリケーションコードをダウンロードし，
利子率モデルのパラメタを推定する（`estimate_uc.R`）．
結果を`data_processed/uc_estimates_y10.RData`として保存．

利子率の将来予測：

```
Rscript simulate_interest_rate.r
```
![利子率データ](output/fig_interest_rate.svg)


GDP成長率の将来予測と突き合わせて選好パラメタを推定：
```
python calibrate_rho_eta.py
```
![選好パラメタのカリブレーション](output/fig_rho_eta_g.svg)

## 損害関数

[Barrage and Nordhaus (2024)](https://doi.org/10.1073/pnas.2312030121)のAppendixから
気候変動の損害推定の離散データを`data_raw/Barrage2024/dice2023.csv`として保存．

```
python calibrate_damage_function.py
```
![損害関数](output/fig_damage_dice2023.svg)

## 大気中濃度と強制力の将来予測

RFFの排出予測と物質循環モデルとを組み合わせて
大気中濃度と強制力の将来予測を作成する．

```
python calculate_scc_1_emis_conc_forc.py
```
![大気中濃度の将来予測（CO2）](output/fig_co2_samples.png)

![大気中濃度の将来予測（CH4）](output/fig_ch4_samples.png)

![大気中濃度の将来予測（N2O）](output/fig_n2o_samples.png)

## 温室効果ガスの社会的費用の推定

パルスありとパルスなしのそれぞれについて，各サンプルパスの気温を計算．
```
python calculate_scc_2_forc_temp.py
```

気温の差分を計算（社会的費用の推定には不要）．
```
python calculate_scc_3_delta_tas.py
```

各サンプルパスについて損害（当期価値）を計算．
```
python calculate_scc_4_damage.py
```

割り引いて現在価値を計算
```
python calculate_scc_5_present_value.py
```
```
social cost of co2 (MIROC6): 90.85139519515872 (USD/tCO2)
social cost of ch4 (MIROC6): 463.6005273819292 (USD/tCH4)
social cost of n2o (MIROC6): 15724.936000175458 (USD/tN2O)
```

## データソース

- Barrage, L., & Nordhaus, W. (2024). Policies, projections, and the social cost of carbon: Results from the DICE-2023 model. Proceedings of the National Academy of Sciences of the United States of America, 121(13), e2312030121. https://doi.org/10.1073/pnas.2312030121
- Bauer, M. D., & Rudebusch, G. D. (2023). The rising cost of climate change: Evidence from the bond market. The Review of Economics and Statistics, 105(5), 1255–1270. https://doi.org/10.1162/rest_a_01109
- Joos, F., Roth, R., Fuglestvedt, J. S., Peters, G. P., Enting, I. G., Bloh, W. von, Brovkin, V., Burke, E. J., Eby, M., Edwards, N. R., & Others. (2013). Carbon dioxide and climate impulse response functions for the computation of greenhouse gas metrics: a multi-model analysis. Atmospheric Chemistry and Physics, 13(5), 2793–2825. https://acp.copernicus.org/articles/13/2793/2013/
- Meinshausen, M., Nicholls, Z. R. J., Lewis, J., Gidden, M. J., Vogel, E., Freund, M., Beyerle, U., Gessner, C., Nauels, A., Bauer, N., Canadell, J. G., Daniel, J. S., John, A., Krummel, P. B., Luderer, G., Meinshausen, N., Montzka, S. A., Rayner, P. J., Reimann, S., Smith, S. J., van den Berg, M., Velders, G. J. M., Vollmer, M. K., and Wang, R. H. J. (2020). The shared socio-economic pathway (SSP) greenhouse gas concentrations and their extensions to 2500, Geoscientific Model Development, 13, 3571–3605, https://doi.org/10.5194/gmd-13-3571-2020
- MPD version 2023: Bolt, Jutta and Jan Luiten van Zanden (2024). Maddison style estimates of the evolution of the world economy: A new 2023 update, Journal of Economic Surveys, 1–41. http://doi.org/10.1111/joes.12618
- [Noto Font](https://fonts.google.com/noto) is licensed under the SIL Open Font License, Version 1.1. Copyright 2012 Google Inc. All Rights Reserved. This license is available at: http://scripts.sil.org/OFL
- Osborn, T.J., Jones, P.D., Lister, D.H., Morice, C.P., Simpson, I.R., Winn, J.P., Hogan, E., and Harris, I.C., (2021). Land surface air temperature variations across the globe updated to 2019: the CRUTEM5 dataset. Journal of Geophysical Research: Atmospheres. 126, e2019JD032352, https://doi.org/10.1029/2019JD032352
- Kevin Rennert, Brian C. Prest, William A. Pizer, Richard G. Newell, David Anthoff, Cora Kingdon, Lisa Rennels, Roger Cooke, Adrian E. Raftery, Hana Ševčíková, & Frank Errickson. (2022). Resources for the Future Socioeconomic Projections (RFF-SPs) (Version 5) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6016583
- Smith, Christopher J. (2019, October 21). Effective Radiative Forcing from Shared Socioeconomic Pathways (Version v0.3.1). Zenodo. http://doi.org/10.5281/zenodo.3515339
- Tatebe, Hiroaki; Watanabe, Masahiro (2018). MIROC MIROC6 model output prepared for CMIP6 CMIP.Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.881
- World Bank. Gross savings (% of GDP), World Development Indicators, The World Bank Group, https://data.worldbank.org/indicator/NY.GNS.ICTR.ZS
- Zebedee Nicholls, & Jared Lewis. (2021). Reduced Complexity Model Intercomparison Project (RCMIP) protocol (v5.1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4589756
