
# Climate emulator and its applications to economic analysis

1.  [Data processing](#org89e0a9d)
2.  [Plotting experimental data](#org8a2bb4e)
3.  [Climate emulator](#org98b387c)
4.  [References](#org2f7887e)


<a id="org89e0a9d"></a>

## Data processing

Retrieve raw CMIP data for surface air temperature (`tas`), incoming shortwave radiation (`rsdt`), outgoing shortwave radiation (`rsut`),
outgoing longwave radiation (`rlut`), and grid cell area (`areacella`) from the [Earth System Grid Federation](https://esgf.llnl.gov/).
Download data associated with the following experiments: `piControl`, \`abrupt-2xCO2\`, \`abrupt-4xCO2\`, `1pctCO2`, `historical`, `ssp119`, `ssp245`, `ssp370`, `ssp460`, and `ssp585`.
Save the downloaded files to the `./data_raw/CMIP6` directory.

Next, process the temperature and radiation data to generate global-mean time series, outputting the results as .csv files. This can be done using the provided Python script:

    python process_cmip_data.py [--model_id MIROC6]

The `--model_id` argument allows for specifying a particular climate model, with `MIROC6` as the default.


<a id="org8a2bb4e"></a>

## Plotting experimental data

After preprocessing the CMIP data, visualizations can be generated to examine the results. For example:

    python plot_experiment_tas.py [--model_id MIROC6]

![img](./output/fig_plot_experiment_tas.svg)

    python plot_historical_tas.py [--model_id MIROC6]

![img](./output/fig_plot_historical_tas.svg)

    python plot_scenario_tas.py [--model_id MIROC6]

![img](./output/fig_plot_scenario_tas.svg)

The generated figures will be stored in `./output` directory.


<a id="org98b387c"></a>

## Climate emulator

Calibrate a two-layer energy balance model a la [Cummins, Stephenson, Stott (2020)](https://doi.org/10.1175/JCLI-D-19-0589.1) based on `abrupt-4xCO2` experiment:

    python calibrate_emulator.py [MIROC6]

The estimated parameter values of the model will be stored in `./output` directory.

Evaluate the internal validity (against `abrupt-4xCO2`)
and the external validity (against `historical`, `ssp119`, `ssp245`, `ssp370`, `ssp460`, `ssp585`)
of the calibrated model:

    python evaluate_emulator.py [--model_id MIROC6]

![img](./output/fig_evaluate_emulator.svg)


<a id="org2f7887e"></a>

## References

-   Barrage, L., & Nordhaus, W. (2024). [Policies, projections, and the social cost of carbon: Results from the DICE-2023 model](https://doi.org/10.1073/pnas.2312030121). Proceedings of the National Academy of Sciences of the United States of America, 121(13), e2312030121.
-   Bauer, M. D., & Rudebusch, G. D. (2023). [The rising cost of climate change: Evidence from the bond market](https://doi.org/10.1162/rest_a_01109). The Review of Economics and Statistics, 105(5), 1255–1270.
-   Cummins, D. P., Stephenson, D. B., & Stott, P. A. (2020). [Optimal Estimation of Stochastic Energy Balance Model Parameters](https://doi.org/10.1175/JCLI-D-19-0589.1). Journal of Climate, 33(18), 7909–7926.
-   Joos, F., Roth, R., Fuglestvedt, J. S., Peters, G. P., Enting, I. G., Bloh, W. von, Brovkin, V., Burke, E. J., Eby, M., Edwards, N. R., & Others. (2013). [Carbon dioxide and climate impulse response functions for the computation of greenhouse gas metrics: a multi-model analysis](https://acp.copernicus.org/articles/13/2793/2013/). Atmospheric Chemistry and Physics, 13(5), 2793–2825.
-   Meinshausen, M., Nicholls, Z. R. J., Lewis, J., Gidden, M. J., Vogel, E., Freund, M., Beyerle, U., Gessner, C., Nauels, A., Bauer, N., Canadell, J. G., Daniel, J. S., John, A., Krummel, P. B., Luderer, G., Meinshausen, N., Montzka, S. A., Rayner, P. J., Reimann, S., Smith, S. J., van den Berg, M., Velders, G. J. M., Vollmer, M. K., and Wang, R. H. J. (2020). [The shared socio-economic pathway (SSP) greenhouse gas concentrations and their extensions to 2500](https://doi.org/10.5194/gmd-13-3571-2020). Geoscientific Model Development, 13, 3571–3605.
-   MPD version 2023: Bolt, Jutta and Jan Luiten van Zanden (2024). [Maddison style estimates of the evolution of the world economy: A new 2023 update](http://doi.org/10.1111/joes.12618). Journal of Economic Surveys, 1–41.
-   Osborn, T.J., Jones, P.D., Lister, D.H., Morice, C.P., Simpson, I.R., Winn, J.P., Hogan, E., and Harris, I.C., (2021). [Land surface air temperature variations across the globe updated to 2019: the CRUTEM5 dataset](https://doi.org/10.1029/2019JD032352). Journal of Geophysical Research: Atmospheres. 126, e2019JD032352,
-   Kevin Rennert, Brian C. Prest, William A. Pizer, Richard G. Newell, David Anthoff, Cora Kingdon, Lisa Rennels, Roger Cooke, Adrian E. Raftery, Hana Ševčíková, & Frank Errickson. (2022). [Resources for the Future Socioeconomic Projections (RFF-SPs) (Version 5) [Data set]​](https://doi.org/10.5281/zenodo.6016583). Zenodo.
-   Smith, Christopher J. (2019, October 21). [Effective Radiative Forcing from Shared Socioeconomic Pathways (Version v0.3.1)](http://doi.org/10.5281/zenodo.3515339). Zenodo.
-   Tatebe, Hiroaki; Watanabe, Masahiro (2018). [MIROC6 model output prepared for CMIP6](https://doi.org/10.22033/ESGF/CMIP6.881). Earth System Grid Federation.
-   World Bank. [Gross savings (% of GDP) World Development Indicators](https://data.worldbank.org/indicator/NY.GNS.ICTR.ZS). The World Bank Group.
-   Zebedee Nicholls, & Jared Lewis. (2021). [Reduced Complexity Model Intercomparison Project (RCMIP) protocol (v5.1.0) [Data set]​](https://doi.org/10.5281/zenodo.4589756). Zenodo.

