import math,numpy as np
def norm_cdf(x):return .5*(1+math.erf(x/math.sqrt(2)))
def black_scholes_price(spot,strike,t_years,rate,sigma,option_type="call",dividend_yield=0.0):
    try:S=float(spot);K=float(strike);T=float(t_years);r=float(rate);v=float(sigma);q=float(dividend_yield)
    except Exception:return np.nan
    if not all(np.isfinite([S,K,T,r,v,q])) or S<=0 or K<=0:return np.nan
    if T<=0:return max(S-K,0.0) if option_type=="call" else max(K-S,0.0)
    v=max(.01,min(5.0,v));root=math.sqrt(T)
    d1=(math.log(S/K)+(r-q+.5*v*v)*T)/(v*root);d2=d1-v*root
    if option_type=="call":return S*math.exp(-q*T)*norm_cdf(d1)-K*math.exp(-r*T)*norm_cdf(d2)
    return K*math.exp(-r*T)*norm_cdf(-d2)-S*math.exp(-q*T)*norm_cdf(-d1)
def theoretical_row(spot,strike,dte,rate,sigma,call_actual,put_actual):
    T=max(float(dte),0)/365
    c=black_scholes_price(spot,strike,T,rate,sigma,"call");p=black_scholes_price(spot,strike,T,rate,sigma,"put")
    ci=max(float(spot)-float(strike),0);pi=max(float(strike)-float(spot),0)
    ce=max(c-ci,0) if np.isfinite(c) else np.nan;pe=max(p-pi,0) if np.isfinite(p) else np.nan
    return {"bs_call_price":c,"bs_put_price":p,"bs_call_extrinsic":ce,"bs_put_extrinsic":pe,
            "call_expectation_gap":ce-float(call_actual) if np.isfinite(ce) else np.nan,
            "put_expectation_gap":pe-float(put_actual) if np.isfinite(pe) else np.nan}
