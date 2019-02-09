import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import norm, skew
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
 
train = pd.read_csv('G:/hoip_data/HOIP_cif/input/Train14Features1012_bg.csv')
test = pd.read_csv('G:/hoip_data/HOIP_cif/input/Search14Features1012.csv')
 
corrmat = train.corr()
ax = sns.heatmap(corrmat, vmax=.8, square=True, cmap = 'Blues', annot_kws={'size': 10})
plt.tight_layout()
plt.savefig('corrmat.png')
plt.show()
 
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.Bandgap_GGA.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['Bandgap_GGA'], axis=1, inplace=True)
all_data.to_csv('all_data.csv', header=None, index=None, sep=' ', mode='a')
 
train = all_data[:ntrain]
test = all_data[ntrain:]
 
train.to_csv('train.csv', header=None, index=None, sep=' ', mode='a')
non_features = ['Bandgap_GGA']
features = [col for col in list(train) if col not in non_features]
 
X = train[features].values
y_E = y_train
 
from sklearn.model_selection import GridSearchCV
num_trees = [200, 400, 600]
max_feat = ['auto', 'sqrt']
max_levels = [4, 5, 6, 7, 8]
min_split = [2, 5, 10]
min_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': num_trees,
               'max_features': max_feat,
               'max_depth': max_levels,
               'min_samples_split': min_split,
               'min_samples_leaf': min_leaf,
               'bootstrap': bootstrap}
print("Random grid")
 
rf = RandomForestRegressor(random_state = 42)
rf_grid_search = GridSearchCV(rf, random_grid, cv=5, 
                              scoring="neg_mean_squared_error", verbose = 1)
print(train.values)
rf_grid_search.fit( train.values, y_train )  
 
with open('random_forest_parameters.txt', 'a') as the_file:
    the_file.write(str(rf_grid_search.best_params_))
 
test_size = 0.2
rstate = 42
X_train_E, X_test_E, y_train_E, y_test_E = train_test_split(X, y_E, 
                                                            test_size=test_size, random_state=rstate)
 
# gridsearch parameters
max_feat = 'sqrt'
bootstrap = True
min_leaf = 1
min_split = 5
num_trees = 400
max_levels = 8
 
rf_E = RandomForestRegressor(num_trees=num_trees, max_levels=max_levels, random_state=rstate)
rf_E.fit(X_train_E, y_train_E)
def rmsle(actual, predicted):
    return np.sqrt(np.mean(np.power(np.log1p(actual) - np.log1p(predicted), 2)))
def get_r2_value(act, pred):
    ybar = np.sum(act)/len(act)          
    ssreg = np.sum((act-pred)**2)  
    sstot = np.sum((act - ybar)**2)  
    r2_value = 1 - ssreg / sstot
    return r2_value
def plot_actual_pred(train_pred, train_actual, test_pred, test_actual, target):
    s = 75
    lw = 0
    alpha = 0.6
    train_color = 'orange'
    train_marker = 's'
    test_color = 'red'
    test_marker = '^'
    axis_width = 1.5
    maj_tick_len = 6
    fontsize = 16
    label = '__nolegend__'
    ax = plt.scatter(train_pred, train_actual,
                     marker=train_marker, color=train_color, s=s,
                     lw=lw, alpha=alpha, label='train')
    ax = plt.scatter(test_pred, test_actual,
                     marker=test_marker, color=test_color, s=s,
                     lw=lw, alpha=alpha, label='test')
    ax = plt.legend(frameon=False, fontsize=fontsize, handletextpad=0.4)
    all_vals = list(train_pred) + list(train_actual) + list(test_pred) + list(test_actual)
    full_range = abs(np.max(all_vals) - np.min(all_vals))
    cushion = 0.1
    xmin = np.min(all_vals) - cushion * full_range
    xmax = np.max(all_vals) + cushion * full_range
    ymin = xmin
    ymax = xmax
    ax = plt.xlim([xmin, xmax])
    ax = plt.ylim([ymin, ymax])
    ax = plt.plot([xmin, xmax], [ymin, ymax],
                  lw=axis_width, color='black', ls='--',
                  label='__nolegend__')
    ax = plt.xlabel('Actual bandgap $E^g_{HOIP}$ (eV)', fontsize=fontsize)
    ax = plt.ylabel('Predicted bandgap $E^g_{HOIP}$ (eV)', fontsize=fontsize)
    ax = plt.xticks(fontsize=fontsize)
    ax = plt.yticks(fontsize=fontsize)
    ax = plt.tick_params('both', length=maj_tick_len, width=axis_width,
                         which='major', right=True, top=True)
    return ax
 
y_train_E_pred = rf_E.predict(X_train_E)
y_test_E_pred = rf_E.predict(X_test_E)
 
rf_E_test_pred = rf_E.predict(test.values)
 
sub = pd.DataFrame()
sub['Bandgap_GGA'] = rf_E_test_pred
sub.to_csv('predict_bg_1013.csv',index=False)
 
target_E = 'band gap (eV)'
print('RMSLE for band gap = %.3f eV (training) and %.3f eV (test)'
      % (rmsle(y_train_E, y_train_E_pred), rmsle(y_test_E, y_test_E_pred)))
 
print(y_train_E[0:int(len(y_train_E)/2)])
plt.subplot(2, 1, 1)
ax1 = plot_actual_pred(y_train_E, y_train_E_pred,
                       y_test_E, y_test_E_pred,
                       target_E)
                       
print('R^2 for band gap = %.3f (training) and %.3f (test)'
      % (get_r2_value(y_train_E, y_train_E_pred), get_r2_value(y_test_E, y_test_E_pred)))
 
plt.tight_layout()
plt.savefig('bgrun1013.png')
plt.show()
plt.close()

predictions = pd.DataFrame()
sub['Bandgap'] = y_test_E_pred
sub.to_csv('output_band_gap.csv',index=False)
