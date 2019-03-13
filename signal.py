import pandas as pd
import quandl
from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import f1_score, confusion_matrix, precision_score
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from math import sqrt


NUM_STOCKS = 50 #Keep this many stocks on each side
MONTHLY_CONF_THRESHOLD = 0.4 #Keep this probability 
DAILY_CONF_THRESHOLD = 0.6

def setup_data():
	market_data = pd.read_csv('market_data.csv')
	company_data = pd.read_csv('company_data.csv')
	index_data = pd.read_csv('indice_data.csv')
	fin_data = pd.read_csv('financial_data.csv')

	daily_rf = quandl.get("USTREASURY/YIELD")['3 MO'] / (60.0 * 100)
	daily_rf.name = 'Daily RF'
	monthly_rf = quandl.get("USTREASURY/YIELD", collapse="monthly")['3 MO'] / (3.0 * 100)
	monthly_rf.name = 'Monthly RF'

	daily = company_data.merge(market_data, on='Ticker')

	daily['Date'] = pd.to_datetime(daily['Date'])
	index_data['Date'] = pd.to_datetime(index_data['Date'])
	daily = daily.sort_values(['Date','Ticker'])

	mkt = index_data[index_data['Index'] == 'SPY'].merge(daily_rf, on='Date')
	mkt['Market Return'] = mkt['Return'] - mkt['Daily RF']
	mkt.drop('Return', axis=1, inplace=True)

	monthly_mkt = index_data[index_data['Index'] == 'SPY'].merge(monthly_rf, on='Date')
	monthly_mkt['Market Return'] = monthly_mkt['Return'] - monthly_mkt['Monthly RF']
	monthly_mkt.drop('Return', axis=1, inplace=True)

	daily = daily.merge(mkt, on='Date')
	daily['Daily Return'] = daily['Total Return'] - daily['Daily RF']
	daily.drop('Total Return', axis=1, inplace=True)

	#daily = daily.rename(columns={'Return':'Market Return'})
	#daily = daily.merge(daily_rf, left_on='Date', right_index=True)


	monthly = daily.set_index('Date').groupby('Ticker').resample('1M'
		,how='sum')[['Daily Return', 'Daily Volume', 'Market Return']].reset_index()
	monthly = monthly.rename(columns={'Daily Return': 'Monthly Return',
		'Market Return': 'Monthly Market Return', 'Daily Volume':'Monthly Volume'})
	monthly = monthly.merge(monthly_rf, left_on='Date', right_index=True)
	monthly = monthly.merge(company_data, on='Ticker')

	return daily, monthly
	#daily.to_pickle('daily.df')
	#monthly.to_pickle('monthly.df')
daily, monthly = setup_data()

def load_data():
	return pd.read_pickle('daily.df'), pd.read_pickle('monthly.df')

def get_daily_beta_adj_return(daily):
	'''Computed daily betas and beta adjusted return to remove market effects in prediction'''

	def calc_daily_beta(group):
		group = group[group['Daily Return'].notnull()]
		X = DataFrame(group['Market Return'])
		X['Int'] = 1
		return Series(np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(group['Daily Return']))

	betas = daily.groupby('Ticker').apply(calc_daily_beta)
	betas.columns = ['Beta', 'Intercept']
	daily = daily.merge(betas, left_on='Ticker', right_index=True)
	daily['Beta Adj Return'] = daily['Daily Return'] - daily['Beta']*daily['Market Return']
	return daily

daily = get_daily_beta_adj_return(daily)

def create_features_daily(daily, scaled=False):
	Xy = DataFrame()
	Xy['Ticker'] = daily['Ticker']
	Xy['Date'] = daily['Date']

	gg = daily.groupby('Ticker')
	scale_cols = []
	for lag in range(0,13):
		print('Volume-%d'%lag)
		Xy['Volume-%d'%lag] = gg['Daily Volume'].apply(lambda x: x.shift(-lag))
		scale_cols.append('Volume-%d'%lag)

	for lag in range(0,1):
		print('Ret-%d'%lag)
		Xy['Ret-%d'%lag] = gg['Beta Adj Return'].apply(lambda x: x.shift(-lag))
		scale_cols.append('Volume-%d'%lag)

	Xy['Y'] = gg['Beta Adj Return'].apply(lambda x: x.shift(1)) #Next day return
	Xy = Xy.dropna()
	Xy['VolumeRatio'] = Xy['Volume-0'] / Xy.filter(regex='Volume-').mean(axis=1) - 1
	Xy['SignedVolRatio'] = Xy['VolumeRatio'] * (Xy['Ret-0'] / Xy['Ret-0'].abs())
	Xy['Sign'] = (Xy['Y'] / Xy['Y'].abs()).astype(int)

	if scaled:
		print("Scaling")
		Xy = scale_xy(Xy, scale_cols)
	
	return Xy

xy = create_features_daily(daily, scaled=True)


def scale_xy(Xy, toscale):
	Xy[toscale] = Xy.groupby('Date')[toscale].apply(zscore)
	for lag in range(0,1):
		X_cols.append('RetVol-%d'%lag)
		Xy['RetVol-%d'%lag] = Xy['Ret-%d'%lag] * Xy['Volume-%d'%lag]
	Xy['RetVol-0'] = Xy.groupby('Date')['RetVol-0'].apply(zscore)
	return Xy

daily_cols = ['Volume-0', 'Ret-0', 'VolumeRatio', 'SignedVolRatio']

def run_daily_classifier(xy, insample_date='20160101'):
	clf = LogisticRegression()
	xy = xy[xy['Date'] <= insample_date]
	clf.fit(xy[daily_cols], xy['Sign'])
	return clf

def gen_daily_pf(clf, xy, insample_date='20160101'):
	dfs = []

	xy['Prediction'] = clf.predict(xy[daily_cols])
	probs = DataFrame(clf.predict_proba(xy[daily_cols]))
	xy['PWinner'] = probs[1]
	xy['PLoser'] = probs[0]
	import pdb
	pdb.set_trace()
	for date in xy['Date'].unique():
		data = xy[xy['Date'] == date].set_index('Ticker')

		#Select likely winners
		top_winners = data.sort_values('PWinner', ascending=False).head(NUM_STOCKS)['PWinner']
		winners = top_winners[top_winners > DAILY_CONF_THRESHOLD]
		top_losers = data.sort_values('PLoser', ascending=False).head(NUM_STOCKS)['PLoser']
		losers = top_losers[top_losers > DAILY_CONF_THRESHOLD]

		#assert(len(winners) > 0)
		#assert(len(losers) > 0)

		winners_norm = (winners / winners.sum()).reset_index() #Scale long pf to 1
		losers_norm = (-1*losers / losers.sum()).reset_index() #Scale short pf to 1 

		winners_norm = winners_norm.rename(columns={'PWinner' :'Weight'})
		losers_norm = losers_norm.rename(columns={'PLoser' :'Weight'})		
		#return winners, losers, winners_norm, losers_norm
		df = pd.concat([winners_norm, losers_norm], ignore_index=1)
		df['Date'] = date
		if (len(winners) > 0 and len(losers) > 0):
			dfs.append(df)
			print(pd.to_datetime(date).strftime('%Y%m%d') + " Winners: %d Losers: %d" %(len(winners),len(losers)))
		else:
			pass 
			print("No good winners/losers for %s" % pd.to_datetime(date).strftime('%Y%m%d'))
			#print(pred_probs.describe())
	return pd.concat(dfs).reset_index(drop=True)


def analyze_daily_pf(pf, daily, insample_date = pd.to_datetime('20160101')):
	comb = pf.merge(daily[['Date','Ticker', 'Y']], on = ['Date','Ticker'], how='left')
	comb['RetCon'] = comb['Weight'] * comb['Y']
	
	dailyRet = comb.groupby('Date')['RetCon'].sum()

	insample = dailyRet[dailyRet.index <= insample_date]

	outofsample = dailyRet[dailyRet.index > insample_date]
	print ("In Sample Sharpe: " + str(insample.mean() / insample.std() * sqrt(252)))
	print ("Out of Sample Sharpe: " + str(outofsample.mean() / outofsample.std() * sqrt(252)))
	return comb	

def run_all_daily(xy):
	clf = run_daily_classifier(xy)
	pf = gen_daily_pf(clf,xy)
	ret = analyze_daily_pf(pf, xy)
	return (clf, pf, ret)

daily_results = run_all_daily(xy)

def fill_fin_data(monthly):
	'''Forward fill financial data so every company has most recent data on every date'''
	daily, monthly = load_data()

	tickers = monthly.groupby('Ticker').first()
	indSize = tickers.groupby('Major SIC')['Date'].transform('count')

	MIN_IND_SIZE = 3
	fin_data['Date'] = pd.to_datetime(fin_data['Fiscal Period End Date'])
	fin_data_filled = fin_data.set_index('Date').groupby('Ticker').apply(lambda x: x.reindex(pd.date_range(x.index.min(), x.index.max(), freq='M'))
		.fillna(method='ffill')).reset_index(1).reset_index(drop=True)
	fin_data_filled = fin_data_filled.rename(columns={'level_1': 'Date'}) 
	fin_data_filled = fin_data_filled.sort_values(['Date', 'Ticker']).set_index('Date')

	monthly = monthly_return.merge(fin_data_filled, on=['Ticker','Date'])

#monthly = fill_fin_data(monthly)

def get_beta_adj_return(monthly):
	'''Computed monthly betas and beta adjusted return to remove market effects in prediction'''

	def calc_monthly_beta(group):
		group = group[group['Monthly Return'].notnull()]
		X = DataFrame(group['Monthly Market Return'])
		X['Int'] = 1
		return Series(np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(group['Monthly Return']))

	betas = monthly.groupby('Ticker').apply(calc_monthly_beta)
	betas.columns = ['Beta', 'Intercept']

	monthly = monthly.merge(betas, left_on='Ticker', right_index=True)
	monthly['Beta Adj Return'] = monthly['Monthly Return'] - monthly['Beta']*monthly['Monthly Market Return']

monthly = get_beta_adj_return(monthly)

#Decisions
	#Calculate beta using monthly return

def zscore(group):
	return ((group - group.mean())/group.std()).fillna(0)


'''
dex(['Ticker', 'Date', 'Monthly Return', 'Monthly Volume',
       'Monthly Market Return', 'Fiscal Period End Date', 'Quarterly Sales',
       'Cost of Goods Sold', 'Selling, General, and Administrative',
       'Net Income', 'Operating Cash Flow', 'Assets', 'Debt', 'Common Equity',
       'Division SIC', 'Major SIC', 'N', 'Beta', 'Intercept',
       'Beta Adj Return']
'''

yoys = ['Quarterly Sales', 'Cost of Goods Sold', 'Selling, General, and Administrative', 
'Operating Cash Flow', 'Net Income', 'D/E', 'Current Ratio', 'ROE']

tsz = ['Quarterly Sales' ,'Cost of Goods Sold', 'Selling, General, and Administrative', 
'Operating Cash Flow', 'Net Income', 'D/E', 'Current Ratio', 'ROE']

xsz = ['Quarterly Sales' ,'Cost of Goods Sold', 'Selling, General, and Administrative', 
'Operating Cash Flow', 'Net Income', 'D/E', 'Current Ratio', 'ROE']

def create_features(monthly):

	monthly['D/E'] = monthly['Debt'] / monthly['Common Equity'] 
	monthly['Current Ratio'] = monthly['Assets'] / monthly['Debt']
	monthly['ROE'] = monthly['Net Income'] / monthly['Common Equity']


	def get_trailing(val):
		return (val - val.shift(12))/ abs(val.shift(12)) 
	monthly[['YoY ' + x for x in yoys]] = monthly.groupby('Ticker')[yoys].apply(get_trailing)

	monthly[['XSZ ' + x for x in xsz]] = monthly.groupby(['Date', 'Division SIC'])[xsz].apply(zscore)

	return monthly.loc[:,~monthly.columns.duplicated()]

monthly = create_features(monthly)



def get_return_bins(monthly):
	def bin(group):
		return pd.qcut(group, 5, labels=[1,2,3,4,5])

	ret = monthly.groupby(['Date'])['Beta Adj Return'].apply(lambda group:group.shift(-1))
	monthly['Prev Beta Adj Return'] = ret
	monthly['Return Bin'] = monthly.groupby('Date')['Prev Beta Adj Return'].apply(bin)
	return monthly

monthly = get_return_bins(monthly)

X_cols = ['Quarterly Sales', 'Cost of Goods Sold', 'Selling, General, and Administrative', 'Net Income', 'Operating Cash Flow', 'Assets','Debt', 'Common Equity'] + ['D/E','Current Ratio', 'ROE'] + ['YoY ' + x for x in yoys] + ['XSZ ' + x for x in xsz]


def getTraining(monthly, insample_date='20160101'):
	Xy = monthly[monthly['Date'] <= insample_date][X_cols + ['Return Bin']].dropna() #Remove NaNs and focus insample
	Xy = Xy[Xy.abs().max(axis=1) != np.inf] #

	X = Xy[X_cols]
	y = Xy['Return Bin']
	import copy
	yex = copy.deepcopy(y)
	yex[(yex != 5) & (yex != 1)] = 3
	return X, yex

def myscore(est, X_test, y_test):
  		f = precision_score(y_test, est.predict(X_test), average=None)
  		return 1/2. * (f[0] + f[2])
	

def run_classifier(monthly, insample_date = pd.to_datetime('20160101')):
	X,y = getTraining(monthly, insample_date)
	
	# Number of trees in random forest
	n_estimators = [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)]
	# Maximum number of levels in tree
	#max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
	max_depth = [5, 10, 15, 20]
	# Minimum number of samples required to split a node
	#min_samples_split = [2, 5, 10]
	min_samples_split = [0.005, 0.05]
	# Minimum number of samples required at each leaf node
	min_samples_leaf = [1, 2, 4]
	# Method of selecting samples for training each tree
	bootstrap = [True, False]

	random_grid = {#'n_estimators': n_estimators,
		           #'max_features': max_features,
		           #'min_samples_split': min_samples_split,
		           #'min_samples_leaf': min_samples_leaf,
		           'min_impurity_decrease': [1e-7, 1e-5, 1e-3],
		           'min_samples_split': [0.0005, 0.005, 0.05],
		           'max_depth': [10, 20, 50]
		           ##'max_depth': [10, 25, 50]
		           #'bootstrap': bootstrap}
		          	}
	
	clf = RandomForestClassifier(100, max_depth=20, bootstrap=True, class_weight='balanced', max_features='sqrt', min_samples_split=0.0005, min_impurity_decrease=1e-5)

		#clf = RandomForestClassifier(50, bootstrap=True, class_weight='balanced', max_features='sqrt', min_samples_split=0.0005, oob_score=True, verbose=0, random_state=99)
	ts = TimeSeriesSplit(n_splits=4)
	cv = ts.split(X)

	
	grid = GridSearchCV(estimator=clf, cv=cv, param_grid = random_grid, verbose=3, n_jobs=10, scoring=myscore, refit=True)
	#grid.fit(X,y)
	#print(grid.best_params_)
	#clf = grid.best_estimator_
	clf.fit(X,y)

	#print(f1_score(clf.predict(X), yex, average=None))
	cm = DataFrame(confusion_matrix(y, clf.predict(X)), index=[1,3,5], columns=[1,3,5])
	print(cm)
	
	return clf


def gen_pf(clf, monthly, CONF_THRESHOLD=MONTHLY_CONF_THRESHOLD):
	#pf = DataFrame(columns=['Date','Ticker','Weight'])
	dfs = []
	for date in monthly['Date'].unique():

		X = monthly[monthly['Date'] == date][X_cols + ['Ticker']].dropna()
		if len(X) == 0:
			continue

		X = X.set_index('Ticker')
		X = X[X.abs().max(axis=1) != np.inf]

		preds = clf.predict(X)
		pred_probs = DataFrame(clf.predict_proba(X), index=X.index)

		#if pd.to_datetime(date) >= pd.to_datetime('20160131'):
		#	pdb.set_trace()
		#Get top longs and short
		X[['PLoser', 'PFlat', 'PWinner']] = pred_probs
		top_winners = X.sort_values('PWinner', ascending=False).head(NUM_STOCKS)['PWinner']
		winners = top_winners[top_winners > MONTHLY_CONF_THRESHOLD]
		top_losers = X.sort_values('PLoser', ascending=False).head(NUM_STOCKS)['PLoser']
		losers = top_losers[top_losers > MONTHLY_CONF_THRESHOLD]

		#common = set(winners.index).intersection(set(losers.index))
		#winners = winners.drop(list(common))
		#losers = losers.drop(list(common))

		winners_norm = (winners / winners.sum()).reset_index() #Scale long pf to 1
		losers_norm = (-1*losers / losers.sum()).reset_index() #Scale short pf to 1 

		winners_norm = winners_norm.rename(columns={'PWinner' :'Weight'})
		losers_norm = losers_norm.rename(columns={'PLoser' :'Weight'})		
		#return winners, losers, winners_norm, losers_norm
		df = pd.concat([winners_norm, losers_norm], ignore_index=1)
		df['Date'] = date
		if (len(winners) > 0 and len(losers) > 0):
			dfs.append(df)
			print(pd.to_datetime(date).strftime('%Y%m%d') + " Winners: %d Losers: %d" %(len(winners),len(losers)))
		else: 
			print("No good winners/losers for %s" % pd.to_datetime(date).strftime('%Y%m%d'))
			print(pred_probs.describe())
		#import pdb
		#pdb.set_trace()
	return pd.concat(dfs).reset_index(drop=True)


def analyze_pf(pf, monthly, insample_date = pd.to_datetime('20160101')):
	comb = pf.merge(monthly[['Date','Ticker', 'Prev Beta Adj Return']], on = ['Date','Ticker'], how='left')
	comb['RetCon'] = comb['Weight'] * comb['Prev Beta Adj Return']
	
	monthlyRet = comb.groupby('Date')['RetCon'].sum()

	insample = monthlyRet[monthlyRet.index <= insample_date]

	outofsample = monthlyRet[monthlyRet.index > insample_date]
	print ("In Sample Sharpe: " + str(insample.mean() / insample.std() * sqrt(12)))
	print ("Out of Sample Sharpe: " + str(outofsample.mean() / outofsample.std() * sqrt(12)))
	return comb	

def run_all(monthly):
	clf = run_classifier(monthly)
	pf = gen_pf(clf, monthly)
	ret = analyze_pf(pf, monthly)
	return clf, pf, ret

monthly_results = run_all(monthly)

def combine_pfs(daily_results, monthly_results):
	dailypf = daily_results[1]
	monthlypf = monthly_results[1]
	mo = monthlypf.drop_duplicates(['Date','Ticker'], keep=False) #drop accidental short and long
	mo = monthly.groupby('Ticker').apply(lambda g: g.set_index('Date').resample('B').ffill()) #Monthly pf to daily pf
	mo.sort_values(['Date','Ticker'], inplace=True)
	mo = mo.reset_index(level=0, drop=True).reset_index()

	def process(group): #rescale to 1
	    group.loc[group['Weight'] > 0,'Weight'] = group[group['Weight'] > 0]['Weight'] * 1 / (group[group['Weight'] > 0]['Weight'].sum())
	    group.loc[group['Weight'] < 0,'Weight'] = group[group['Weight'] < 0]['Weight'] * 1 / abs(group[group['Weight'] < 0]['Weight'].sum())
	    return group
	mo2 = mo.groupby('Date').apply(process)
	mo2['Weight'] = mo2['Weight'] * 0.6
	dailypf['Weight'] = dailypf['Weight'] * 0.4
	both = pd.concat([mo2[['Ticker','Date','Weight']],dailypf])
	both.dropna(inplace=True)
	both.sort_values('Date',inplace=True)
	both.to_csv('pf.csv')

combine_pfs(daily_results, monthly_results)

