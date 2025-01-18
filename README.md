# Table of Contents

1.  [Climate emulator and its applications to economic analysis](#org10a727b)
    1.  [Data processing](#org79d9a70)
    2.  [Plotting experimental data](#orgd757f45)
    3.  [Climate emulator](#org8f62890)
    4.  [References](#org4c4b032)


<a id="org10a727b"></a>

# Climate emulator and its applications to economic analysis


<a id="org79d9a70"></a>

## Data processing

Retrieve raw CMIP data for surface air temperature (`tas`), incoming shortwave radiation (`rsdt`), outgoing shortwave radiation (`rsut`),
outgoing longwave radiation (`rlut`), and grid cell area (`areacella`) from the [Earth System Grid Federation](https://esgf.llnl.gov/).
Download data associated with the following experiments: `piControl`, \`abrupt-2xCO2\`, \`abrupt-4xCO2\`, `1pctCO2`, `historical`, `ssp119`, `ssp245`, `ssp370`, `ssp460`, and `ssp585`.
Save the downloaded files to the \`./data<sub>raw</sub>/CMIP6\` directory.

Next, process the temperature and radiation data to generate global-mean time series, outputting the results as .csv files. This can be done using the provided Python script:

    python process_cmip_data.py [--model_id MIROC6]

The \`&#x2013;model<sub>id</sub>\` argument allows for specifying a particular climate model, with `MIROC6` as the default.


<a id="orgd757f45"></a>

## Plotting experimental data

After preprocessing the CMIP data, visualizations can be generated to examine the results. For example:

    python plot_experiment_tas.py [--model_id MIROC6]

![img](./output/fig_plot_experiment_tas.svg)

    python plot_historical_tas.py [--model_id MIROC6]

![img](./output/fig_plot_historical_tas.svg)

    python plot_scenario_tas.py [--model_id MIROC6]

![img](./output/fig_plot_scenario_tas.svg)

The generated figures will be stored in `./output` directory.


<a id="org8f62890"></a>

## Climate emulator

Calibrate a two-layer energy balance model a la [Cummins, Stephenson, Stott (2020)](https://doi.org/10.1175/JCLI-D-19-0589.1) based on `abrupt-4xCO2` experiment:

    python calibrate_emulator.py [MIROC6]

The estimated parameter values of the model will be stored in `./output` directory.

Evaluate the internal validity (against `abrupt-4xCO2`)
and the external validity (against `historical`, `ssp119`, `ssp245`, `ssp370`, `ssp460`, `ssp585`)
of the calibrated model:

    python evaluate_emulator.py [--model_id MIROC6]

![img](./output/fig_evaluate_emulator.svg)


<a id="org4c4b032"></a>

## References

-   Barrage, L., & Nordhaus, W. (2024). Policies, projections, and the social cost of carbon: Results from the DICE-2023 model. Proceedings of the National Academy of Sciences of the United States of America, 121(13), e2312030121. <https://doi.org/10.1073/pnas.2312030121>
-   Bauer, M. D., & Rudebusch, G. D. (2023). The rising cost of climate change: Evidence from the bond market. The Review of Economics and Statistics, 105(5), 1255–1270. <https://doi.org/10.1162/rest_a_01109>
-   Cummins, D. P., Stephenson, D. B., & Stott, P. A. (2020). Optimal Estimation of Stochastic Energy Balance Model Parameters. Journal of Climate, 33(18), 7909–7926. <https://doi.org/10.1175/JCLI-D-19-0589.1>
-   Joos, F., Roth, R., Fuglestvedt, J. S., Peters, G. P., Enting, I. G., Bloh, W. von, Brovkin, V., Burke, E. J., Eby, M., Edwards, N. R., & Others. (2013). Carbon dioxide and climate impulse response functions for the computation of greenhouse gas metrics: a multi-model analysis. Atmospheric Chemistry and Physics, 13(5), 2793–2825. <https://acp.copernicus.org/articles/13/2793/2013/>
-   Meinshausen, M., Nicholls, Z. R. J., Lewis, J., Gidden, M. J., Vogel, E., Freund, M., Beyerle, U., Gessner, C., Nauels, A., Bauer, N., Canadell, J. G., Daniel, J. S., John, A., Krummel, P. B., Luderer, G., Meinshausen, N., Montzka, S. A., Rayner, P. J., Reimann, S., Smith, S. J., van den Berg, M., Velders, G. J. M., Vollmer, M. K., and Wang, R. H. J. (2020). The shared socio-economic pathway (SSP) greenhouse gas concentrations and their extensions to 2500, Geoscientific Model Development, 13, 3571–3605, <https://doi.org/10.5194/gmd-13-3571-2020>
-   MPD version 2023: Bolt, Jutta and Jan Luiten van Zanden (2024). Maddison style estimates of the evolution of the world economy: A new 2023 update, Journal of Economic Surveys, 1–41. <http://doi.org/10.1111/joes.12618>
-   Osborn, T.J., Jones, P.D., Lister, D.H., Morice, C.P., Simpson, I.R., Winn, J.P., Hogan, E., and Harris, I.C., (2021). Land surface air temperature variations across the globe updated to 2019: the CRUTEM5 dataset. Journal of Geophysical Research: Atmospheres. 126, e2019JD032352, <https://doi.org/10.1029/2019JD032352>
-   Kevin Rennert, Brian C. Prest, William A. Pizer, Richard G. Newell, David Anthoff, Cora Kingdon, Lisa Rennels, Roger Cooke, Adrian E. Raftery, Hana Ševčíková, & Frank Errickson. (2022). Resources for the Future Socioeconomic Projections (RFF-SPs) (Version 5) [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.6016583>
-   Smith, Christopher J. (2019, October 21). Effective Radiative Forcing from Shared Socioeconomic Pathways (Version v0.3.1). Zenodo. <http://doi.org/10.5281/zenodo.3515339>
-   Tatebe, Hiroaki; Watanabe, Masahiro (2018). MIROC MIROC6 model output prepared for CMIP6 CMIP.Earth System Grid Federation. <https://doi.org/10.22033/ESGF/CMIP6.881>
-   World Bank. Gross savings (% of GDP), World Development Indicators, The World Bank Group, <https://data.worldbank.org/indicator/NY.GNS.ICTR.ZS>
-   Zebedee Nicholls, & Jared Lewis. (2021). Reduced Complexity Model Intercomparison Project (RCMIP) protocol (v5.1.0) [Data set]. Zenodo. <https://doi.org/10.5281/zenodo.4589756>
