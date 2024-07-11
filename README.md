## 実験データの処理

まず，各種実験（`piControl`，`abrupt-2xCO2`，`abrupt-4xCO2`，`1pctCO2`，`historical`，`ssp119`，`ssp245`，`ssp370`，`ssp460`，`ssp585`）のそれぞれについて，
必要な変数（`tas`，`rsdt`，`rsut`，`rlut`，`areacella`）のデータを[ESGF](https://esgf.llnl.gov/)からダウンロードします．
MIROC6であれば，`data_raw/required.txt`に挙げたデータになります．
これを全て`data_raw/CMIP6`に置きます．面積データ（`areacella*.nc`）はなくても動きます．

その上で
```
python process_cmip_data.py [--model_id MIROC6]
```
とすれば`data_processed`に全球年平均の時系列データ（csvファイル）が作成されます．
`model_id`はダウンロードしたデータに応じて適宜変更（デフォルトは`MIROC6`）．

## 実験データのプロット

```
python plot_experiment_tas.py [--model_id MIROC6]
python plot_historical_tas.py [--model_id MIROC6]
python plot_scenario_tas.py [--model_id MIROC6]
```
図は全て`output`フォルダに作成されます．

## エミュレータのカリブレーション

```
Rscript calibrate_emulator.r [MIROC6]
```
推定結果が`output`フォルダに保存されます．

## エミュレータの性能評価

```
python evaluate_emulator.py [--model_id MIROC6]
```
図が`output`フォルダに保存されます．

## データソース

- Osborn, T.J., Jones, P.D., Lister, D.H., Morice, C.P., Simpson, I.R., Winn, J.P., Hogan, E., and Harris, I.C., (2021). Land surface air temperature variations across the globe updated to 2019: the CRUTEM5 dataset. Journal of Geophysical Research: Atmospheres. 126, e2019JD032352, https://doi.org/10.1029/2019JD032352
- Tatebe, Hiroaki; Watanabe, Masahiro (2018). MIROC MIROC6 model output prepared for CMIP6 CMIP.Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.881
- Meinshausen, M., Nicholls, Z. R. J., Lewis, J., Gidden, M. J., Vogel, E., Freund, M., Beyerle, U., Gessner, C., Nauels, A., Bauer, N., Canadell, J. G., Daniel, J. S., John, A., Krummel, P. B., Luderer, G., Meinshausen, N., Montzka, S. A., Rayner, P. J., Reimann, S., Smith, S. J., van den Berg, M., Velders, G. J. M., Vollmer, M. K., and Wang, R. H. J. (2020). The shared socio-economic pathway (SSP) greenhouse gas concentrations and their extensions to 2500, Geoscientific Model Development, 13, 3571–3605, https://doi.org/10.5194/gmd-13-3571-2020
- Smith, Christopher J. (2019, October 21). Effective Radiative Forcing from Shared Socioeconomic Pathways (Version v0.3.1). Zenodo. http://doi.org/10.5281/zenodo.3515339
