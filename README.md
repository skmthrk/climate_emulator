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

