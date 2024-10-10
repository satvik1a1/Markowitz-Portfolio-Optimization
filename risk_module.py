import numpy as np
import pandas as pd
#scipy is a python library used for scientific and technical computing
from scipy.stats import norm
from scipy.optimize import minimize

#Skewness - Describes the asymmetry of a distribution around its mean, it quantifies the extent to which a distribution deviates from normal distribution
def skewness(r):
    #calculates the skewness of a return series/dataframe
    sigma_r = r.std()
    skew_r = (((r - r.mean())/sigma_r)**3).mean()
    return skew_r

#Kurtosis - Quantifies the tailedness/shape of a distribution relative to normal distribution, provides insights into sharpness of distribution
def kurtosis(r):
    #calculates the kurtosis of a return series/dataframe
    sigma_r = r.std()
    kurt_r = (((r - r.mean())/sigma_r)**4).mean() - 3
    return kurt_r

def compound(r):
    #calculates compound returns of a return series/dataframe
    return np.expm1(np.log1p(r).sum())

def annualized_return(r):
    # Calculates annual returns given a daily return series
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(252/n_periods) - 1

def annualized_vol(r):
    # Calculates annual volatility given a daily return series
    return r.std() * np.sqrt(252)

def sharpe_ratio(r, riskfree_rate=0.03):
    # Calculates the Sharpe ratio given a daily return series and risk free rate as 3 percent
    rf_per_period = (1 + riskfree_rate)**(1/252) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualized_return(excess_ret)
    ann_vol = annualized_vol(r)
    return ann_ex_ret / ann_vol

#Drawdown is a measure used to evaluate the decline in value of an investment portfolio from its peak value before a new peak is achieved, it provides insights into risk and volatility of portfolio
def drawdown(return_series: pd.Series):
    #output: a dataframe having columns which give drawdown, the wealth index and the maximum return value so far for every month
    wealth_index = 1000*(1+return_series).cumprod()  #wealth index is the value of portfolio at time t, calculated for a initial investment of 1000
    previous_peaks = wealth_index.cummax()  #identifies the highest value reached by the wealth index upto each point in time
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})

#Semideviation is measure of downside risk that focuses on volatility of negative returns
def semideviation(r):
    #calculates semi-deviation(deviation for negative returns) for a monthly return series
    if isinstance(r, pd.Series):
        is_negative = r < 0  #a boolean function - True if r < 0, else False
        return r[is_negative].std(ddof=0)  #calculates population std dev(dof = 0) of the return series with negative returns
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

#VaR - Maximum expected loss over a specific time period for a certain level of confidence, eg - 5% var of $1 million over day implies that there is 5% chance that the portfolio will lose more than $1 million in a day
def var_gaussian(r, level=5, modified=False):
    #calculates the Value at risk for a return series and 5% confidence
    #For normal returns, use modified = False
    #For returns which are not normal, use modified = True (Cornish Fisher VaR is used)
    z = norm.ppf(level/100)  #z-score for a given confidence level for normal distribution
    #Adjusted z-score for modified distribution using Cornish Fisher expansion
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))


# Function to calculate portfolio return
def portfolio_return(weights, expected_returns):
    return np.dot(weights, expected_returns)

# Function to calculate portfolio volatility
def portfolio_vol(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def minimize_vol(target_return, er, cov):
    #outputs weights for assets to minimize volatility for a particular return rate of the portfolio
    n = er.shape[0]  #number of assets
    init_guess = np.repeat(1/n, n)  #initial guess for weights - each asset assigned equal weights
    bounds = ((0.0, 1.0),) * n  #weight of each asset is bounded b/w 0 and 1

    weights_sum_to_1 = {'type': 'eq',  #equality constraint - function specified in 'fun' must be = 0
                        'fun': lambda weights: np.sum(weights) - 1
    }  #constraint on sum of weights
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }  #constraint on return of portfolio, should be equal to target return
    #using minimize function from scipy to calculate optimal weights for minimum voltility
    #portfolio_vol is the function to be minimized
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',  #SLSQP - Sequential Least Square Quadratic Programming - gradient based optimization algorithm
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x

def msr(riskfree_rate, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

    def neg_sharpe(weights, riskfree_rate, er, cov):
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol

    result = minimize(neg_sharpe, init_guess, args=(riskfree_rate, er, cov), method='SLSQP', options={'disp': False}, constraints=(weights_sum_to_1,), bounds=bounds)
    return result.x

#Global Minimum Variance Portfolio (GMV) - Portfolio with minimum variance (lowest risk)
def gmv(cov_matrix):
    num_assets = cov_matrix.shape[0]
    init_guess = np.repeat(1 / num_assets, num_assets)
    bounds = ((0.0, 1.0),) * num_assets
    weights_sum_to_1 = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    
    def portfolio_vol(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    result = minimize(portfolio_vol, init_guess,
                      args=(cov_matrix,), 
                      method='SLSQP',
                      options={'disp': False},
                      constraints=(weights_sum_to_1,),
                      bounds=bounds)
    return result.x

def optimal_weights(n_points, expected_returns, cov_matrix):
    num_assets = len(expected_returns)
    return [np.random.dirichlet(np.ones(num_assets)) for _ in range(n_points)]

#Plotting the Efficient frontier
#show_cml - To show the capital market line (Default = False)
#show_ew - To show the equally weighted portfolio (Dafault = False)
#show_gmv - To show the Global Minimum Variance portfolio (Defalut = False)
def plot_ef(n_points, er, cov, riskfree_rate=0.03):
    weights = [msr(riskfree_rate, er, cov) for _ in range(n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({"Returns": rets, "Volatility": vols})
    ax = ef.plot.line(x="Volatility", y="Returns", style='.-')
    return ax

#CPPI - Allows investor to maintain exposure to the upside value of the risky asset while providing a capital guarantee against downside risk
def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    #Runs a backtest of the CPPI strategy, given a set of returns for the risky asset
    #Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    #set up the CPPI parameters
    #riskfree_rate is used if safe_r is not provided
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = account_value
    #if risky_r is a pd.Series, we convert it into a pd.DataFrame
    if isinstance(risky_r, pd.Series): 
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    #if safe_r is None, we use a constant monthly riskfree_rate
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 

    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floorval_history = pd.DataFrame().reindex_like(risky_r)
    peak_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save the histories for analysis and plotting
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value
        floorval_history.iloc[step] = floor_value
        peak_history.iloc[step] = peak
        
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history
    }
    return backtest_result

def summary_stats(r, riskfree_rate=0.03):
    #Outputs a dataframe with basic analysis for a return series or dataframe
    ann_r = annualized_return(r)
    ann_vol = annualized_vol(r)
    ex_ret = [i - riskfree_rate for i in ann_r]
    ann_sr = ex_ret/ann_vol
    skew = skewness(r)
    kurt = kurtosis(r)
    cf_var5 = var_gaussian(r, modified = True)

    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Sharpe Ratio": ann_sr,
        
    }).reset_index()