import numpy as np
import pandas as pd
import statsmodels.api as sm

#for reference
# SPECS = {
#     "Brent":  {"ticker": "COZ25 Comdty",  "exchange": "ICE Futures Europe", "quote": "USD/bbl",       "lot": 1000,  "conversion": 1}, # conversion means convert to dollar, so 1 dollar is 1 dollar, 1 cent is 0.01 dollar
#     "WTI":    {"ticker": "CLZ25 Comdty",  "exchange": "NYMEX",              "quote": "USD/bbl",       "lot": 1000,  "conversion": 1},
#     "Dubai":  {"ticker": "DBLZ25 Comdty", "exchange": "Platts-linked (verify)", "quote": "USD/bbl",   "lot": 1000,  "conversion": 1},
#     "ULSD":   {"ticker": "HOZ25 Comdty",  "exchange": "NYMEX",              "quote": "US cents/gal",  "lot": 42000, "conversion": 0.01},
#     "Gasoil": {"ticker": "QSZ25 Comdty",  "exchange": "ICE Futures Europe", "quote": "USD/tonne",     "lot": 100,   "conversion": 1},
#     "RBOB":   {"ticker": "XBZ25 Comdty",  "exchange": "NYMEX",              "quote": "US cents/gal",  "lot": 42000, "conversion": 0.01},
#     "NatGas": {"ticker": "NGZ25 Comdty",  "exchange": "NYMEX",              "quote": "USD/MMBtu",     "lot": 10000, "conversion": 1},
#     "Gold":   {"ticker": "GCZ25 Comdty",  "exchange": "COMEX",              "quote": "USD/oz",        "lot": 100,   "conversion": 1},
#     "Silver": {"ticker": "SIZ25 Comdty",  "exchange": "COMEX",              "quote": "USD/oz",        "lot": 5000,  "conversion": 1},
#     "Copper": {"ticker": "HGZ25 Comdty",  "exchange": "COMEX",              "quote": "US cents/lb",   "lot": 25000, "conversion": 0.01},
#     "Coffee": {"ticker": "KCZ25 Comdty",  "exchange": "ICE Futures US",     "quote": "US cents/lb",   "lot": 37500, "conversion": 0.01},
#     "Corn":   {"ticker": "C Z25 Comdty",  "exchange": "CBOT",               "quote": "US cents/bu",   "lot": 5000,  "conversion": 0.01},
#     "Wheat":  {"ticker": "W Z25 Comdty",  "exchange": "CBOT",               "quote": "US cents/bu",   "lot": 5000,  "conversion": 0.01},
# }


def load_field(path, spec, field = "last"):
    raw = pd.read_csv(path)
    raw['date'] = pd.to_datetime(raw['price_date'], format = '%d/%m/%Y')
    ticker_to_name = {info["ticker"]: name for name, info in spec.items()}
    raw = raw[raw["bbg_symbol"].isin(ticker_to_name)].copy()   #This is to remove all the ones that is not what we consider
    raw['name'] = raw['bbg_symbol'].map(ticker_to_name)
    table = raw.pivot(index = 'date', columns = 'name', values = field)  #we consider the last price as default
    return table[list(spec)]

def contract_value(prices, specs):
    #Here is to help later calculate the total value of each contract
    convert = pd.Series({name : attribute['conversion'] for name, attribute in specs.items()})
    amount = pd.Series({name : attribute['lot'] for name, attribute in specs.items()})
    return prices* convert * amount

def coverage_table(prices, volume):
    rows = {}
    for name in prices.columns:
        p = prices[name].dropna()
        inside = prices[name].loc[p.index[0]:p.index[-1]]
        v = volume[name].reindex(p.index)
        rows[name] = {
            "first date" : p.index[0].date(),
            "last date" : p.index[-1].date(),
            "days with value" : len(p),
            "gaps inside" : int(inside.isna().sum()),
            "unchanged days" : int((p.diff() == 0).sum()),
            "median volume" : v.median(),
        }
    return pd.DataFrame(rows).T

def pair_ols(y,x,maxlags=5):
    paired_data = pd.concat([y,x], axis = 1).dropna()
    X = sm.add_constant(paired_data.iloc[:, 1])
    model = sm.OLS(paired_data.iloc[:,0], X).fit(cov_type= "HAC", cov_kwds= {"maxlags": maxlags})
    return {   #return list of model.params
        "n": int(model.nobs),    
        "corr": paired_data.iloc[:, 0].corr(paired_data.iloc[:, 1]),
        "r2": model.rsquared,
        "beta": model.params.iloc[1],
        "beta_se": model.bse.iloc[1],
        "alpha": model.params.iloc[0],
        "resid_sd": model.resid.std(),
        "dw": sm.stats.durbin_watson(model.resid),
    }
#用 pair_ols 把目标对其他每一个合约回归，按 R² 从大到小排序，返回前 top 个（一张表，每个对冲品一行）。
def rank_hedges(returns, target, top=3):
    rows = {}
    for hedge in list(returns):
        if hedge != target:
            rows[hedge] = pair_ols(returns[target], returns[hedge])
    table = pd.DataFrame(rows).T.sort_values("r2", ascending=False)
    return table.head(top)


def min_var_ratio(target_pnl, hedge_pnl):
    # h = cov(a,b)/var(b)
    return np.cov(target_pnl,hedge_pnl)[0,1]/ np.var(hedge_pnl, ddof=1)

def hedge_effectiveness(target_pnl, hedge_pnl, h):
    #HE = 1 − Var(目标 − h × 对冲品) / Var(目标)
    return 1 - np.var(target_pnl - h * hedge_pnl, ddof = 1) / np.var(target_pnl, ddof = 1)

def rolling_fit(y, x, window):
    r2 = y.rolling(window).corr(x) **2
    beta = y.rolling(window).cov(x) / x.rolling(window).var()
    return pd.DataFrame({
        "r2" : r2,
        "beta" : beta,
    }).dropna()


def oos_test(pnl, target, hedge, train_frac=0.7):
    separate = int(len(pnl) * train_frac)
    train_part = pnl.iloc[:separate]
    test_part = pnl.iloc[separate:]
    train_ratio = min_var_ratio(train_part[target], train_part[hedge])
    train_effectiveness = hedge_effectiveness(train_part[target], train_part[hedge], train_ratio)
    test_effectiveness = hedge_effectiveness(test_part[target], test_part[hedge], train_ratio)
    return  {'h* (train)':train_ratio, 'HE in-sample' : train_effectiveness, 'HE out-sample' : test_effectiveness}




