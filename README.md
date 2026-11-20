# Commodity Futures Cross-Hedging

Which of 13 December-2025 commodity futures can hedge each other, how many contracts are needed,
and whether the hedges keep working on data they were not estimated on.

## Key results

- **Ranking:** all 78 pairs are ranked by R². The crude grades are almost interchangeable
  (Brent/WTI R² 0.995), refined products follow crude closely (RBOB 0.94, ULSD 0.87), and natural
  gas, coffee and copper have no useful cross-hedge (best R² 0.15 or less).
- **Sizing:** contracts = β × value of one target contract ÷ value of one hedge contract. For
  gasoline against crude, β = 0.87 becomes 1.14 contracts: a desk long 10 RBOB sells 11 WTI.
- **Out of sample:** with the ratio estimated on the first 70% of days, WTI with Brent removes 99% of
  the variance on the last 30%, and RBOB with WTI 87%. Gold/silver (69%) and corn/wheat (40%) are
  partial hedges.
- **Stability:** rolling windows, sub-periods and a block bootstrap show the crude hedges are stable
  all year, while the weaker pairs change a lot (gold/silver R² 0.31 to 0.70).
- **Sample design:** a pairwise-overlap design gives almost the same ranking (Spearman 0.964). The
  small fall in R² is statistically significant (paired t-test, Wilcoxon) and comes from ULSD close
  to expiry (Fisher z-test), which supports ending the main sample on 31 October 2025.

## Structure

- `src/hedging_tools.py`: data loading, contract values, regressions, hedge ratios, stability and
  out-of-sample tests
- `notebooks/hedging_analysis.ipynb`: the full analysis, charts, cleaning log and recommendation

## How to run

```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The data is not included: it is a licensed Bloomberg export and cannot be redistributed. The
notebook expects `data/cmdy_futures.csv` with one row per date and contract, including the columns
`price_date` (dd/mm/yyyy), `bbg_symbol`, `last` and `volume`. The 13 tickers used are listed in
`SPECS` in the notebook. The notebook keeps its outputs, so all results can be read without the data.

## Limitations

Ten months of data for one contract month. Relationships weaken close to expiry. London and New
York contracts settle at different times, which understates their daily R². Futures trade in whole
contracts, and trading costs are ignored.
