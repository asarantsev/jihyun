import numpy
import pandas
from scipy import stats
from statsmodels.api import OLS
from matplotlib import pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

zeros = pandas.read_excel('treasury.xlsx', sheet_name = 'zeros')
vix = zeros['VIX'].iloc[1:]
zeros = zeros.drop(['Month', 'VIX'], axis = 1)
zeros['slope'] = zeros['10Y'] - zeros['1Y']
zeros['curv'] = 2 * zeros['5Y'] - zeros['10Y'] - zeros['1Y']
plt.plot(zeros['1Y'])
plt.plot(zeros['slope'])
plt.plot(zeros['curv'])
plt.show()

def analysis(res):
    nres = res/vix
    return [stats.skew(res), stats.skew(nres), stats.kurtosis(res), stats.kurtosis(nres)]

def AR(series):
    adf = adfuller(series, maxlag = 15)[1]
    Reg = stats.linregress(series[:-1], numpy.diff(series))
    print(Reg)
    s = Reg.slope
    i = Reg.intercept
    r = Reg.rvalue
    p = Reg.pvalue
    resid = numpy.diff(series) - s * series[:-1] - i * numpy.ones(len(series)-1)
    return [adf, i, s+1, r, p, resid]

def white(resid):
    nresid = resid/vix
    plot_acf(resid)
    plt.title('Original Residuals for PC' + str(n+1))
    plt.show()
    plot_acf(nresid)
    plt.title('Normalized Residuals for PC' + str(n+1))
    plt.show()
    plot_acf(abs(resid))
    plt.title('Original Absolute for PC' + str(n+1))
    plt.show()
    plot_acf(abs(nresid))
    plt.title('Normalized Absolute for PC' + str(n+1))
    plt.show()
    return 0

Results = pandas.DataFrame(columns = zeros.keys())
output = ['ADF p', 'a', 'b', 'R-value', 'Student p', 'OrigSkew', 'NormSkew', 'OrigKurt', 'NormKurt']
Results.insert(0, 'values', output, True)
for maturity in zeros:
    rate = zeros[maturity].values
    resid = AR(rate)[-1]
    Results[maturity] = AR(rate)[:5] + analysis(resid)

factors = ['PC1', 'PC2', 'PC3']
pca = PCA(n_components=3)
pca.fit(zeros)
print(pca.explained_variance_ratio_)
components = pca.transform(zeros)
for n in range(3):
    PC = components[:, n]
    resid = AR(PC)[-1]
    white(noise)
    Results['PC'+str(n+1)] = AR(PC)[:5] + analysis(resid)
    
for n in range(3):
    plt.plot(components[:, n])
plt.show()

print(Results.round(3).to_string(index=False))

model = VAR(zeros[['1Y', '5Y', '10Y']]).fit(1)
print('Free terms = ', model.params.iloc[0])
matrix = model.params.iloc[1:]
print('Matrix = ', matrix)
print('Eigenvalues = ', numpy.linalg.eig(matrix).eigenvalues)
VARresid = numpy.transpose(model.resid)
print(numpy.corrcoef(VARresid))
VARresults = pandas.DataFrame(columns = ['1Y', '5Y', '10Y'])
for maturity in model.params:
    VARresults[maturity] = [numpy.std(model.resid[maturity])] + analysis(model.resid[maturity])
    
VARresults['output'] = ['std', 'OrigSkew', 'NormSkew', 'OrigKurt', 'NormKurt']
print(VARresults.round(3).to_string(index=False))

spreads = pandas.DataFrame(columns = zeros.keys())
spreads = spreads.drop(['1Y', 'slope', 'curv'], axis = 1)
for maturity in spreads:
    spreads[maturity] = zeros[maturity] - zeros['1Y']
    
spreadResults = pandas.DataFrame(columns = spreads.keys())
spreadResults.insert(0, 'values', output, True)
for maturity in spreads:
    rate = spreads[maturity].values
    resid = AR(rate)[-1]
    spreadResults[maturity] = AR(rate)[:5] + analysis(resid)

pca = PCA(n_components=2)
pca.fit(spreads)
print(pca.explained_variance_ratio_)
components = pca.transform(spreads)
for n in range(2):
    PC = components[:, n]
    resid = AR(PC)[-1]
    white(noise)
    spreadResults['PC' + str(n+1)] = AR(PC)[:5] + analysis(resid)
    
for n in range(2):
    plt.plot(components[:, n])
plt.show()

print(spreadResults.round(3).to_string(index=False))