import numpy
import pandas
from scipy import stats
from statsmodels.api import OLS
from matplotlib import pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf

DFclassic = pandas.read_excel('treasury.xlsx', sheet_name = 'classics')
vix = DFclassic['VIX'].iloc[1:]
rates = DFclassic.drop(['Month', 'VIX'], axis = 1)
plt.plot(rates.iloc[0])
plt.plot(rates.iloc[120])
plt.plot(rates.iloc[240])
plt.plot(rates.iloc[-1])
plt.show()
rates['slope'] = rates['30Y'] - rates['3M']
rates['curv'] = 2 * rates['5Y'] - rates['30Y'] - rates['3M']
plt.plot(rates['3M'])
plt.plot(rates['slope'])
plt.plot(rates['curv'])
plt.show()

def analysis(res):
    nres = res/vix
    return [stats.skew(res), stats.skew(nres), stats.kurtosis(res), stats.kurtosis(nres)]

def AR(series):
    adf = adfuller(series, maxlag = 15)[1]
    Reg = stats.linregress(series[:-1], numpy.diff(series))
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

Results = pandas.DataFrame(columns = rates.keys())
output = ['ADF p', 'a', 'b', 'R-value', 'Student p', 'OrigSkew', 'NormSkew', 'OrigKurt', 'NormKurt']
Results.insert(0, 'values', output, True)
for maturity in rates:
    rate = rates[maturity].values
    resid = AR(rate)[-1]
    Results[maturity] = AR(rate)[:5] + analysis(resid)

pca = PCA(n_components=3)
pca.fit(rates)
print(pca.explained_variance_ratio_)
components = pca.transform(rates)
for n in range(3):
    PC = components[:, n]
    resid = AR(PC)[-1]
    white(resid)
    Results['PC' + str(n+1)] = AR(PC)[:5] + analysis(resid)

for n in range(3):
    plt.plot(components[:, n])
plt.show()
print(Results.round(3).to_string(index=False))

model = VAR(rates[['3M', '5Y', '30Y']]).fit(1)
print('Free terms = ', model.params.iloc[0])
matrix = numpy.transpose(model.params.iloc[1:])
print('Matrix = ', matrix)
print('Eigenvalues = ', numpy.linalg.eig(matrix).eigenvalues)
VARresid = numpy.transpose(model.resid)
print(numpy.corrcoef(VARresid))
VARresults = pandas.DataFrame(columns = ['3M', '5Y', '30Y'])
for item in ['3M', '5Y', '30Y']:
    VARresults[item] = [numpy.std(model.resid[item])] + analysis(model.resid[item])
    
VARresults['output'] = ['std', 'OrigSkew', 'NormSkew', 'OrigKurt', 'NormKurt']
print(VARresults.round(3).to_string(index=False))

spreads = pandas.DataFrame(columns = rates.keys())
spreads = spreads.drop(['3M', 'slope', 'curv'], axis = 1)
for maturity in spreads:
    spreads[maturity] = rates[maturity] - rates['3M']
    
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
    white(resid)
    spreadResults['PC' + str(n+1)] = AR(PC)[:5] + analysis(resid)

for n in range(2):
    plt.plot(components[:, n])
plt.show()

print(spreadResults.round(3).to_string(index=False))