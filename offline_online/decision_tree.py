
#%% 計算相關性
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
# https://stackoverflow.com/questions/46498455/categorical-features-correlation/46498792#46498792

import scipy.stats as ss
import numpy as np

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(df[x], df[y]) # 製作列聯表 (contingency table)
    chi2 = ss.chi2_contingency(confusion_matrix)[0] #Chi-square test of independence of variables in a contingency table
    # https://blog.pulipuli.info/2017/10/correlations-with-categorical-variables.html
    # [0] 為卡方統計量
    # [2] 為 自由度
    # [3] 為 列連表期望個數
    n = confusion_matrix.sum().sum() #列聯表行與列加總
    r,k = confusion_matrix.shape
    return np.sqrt(chi2/(n*min(r-1, k-1)))
    '''
    phi2 = chi2/n #計算 Cramer 係數(部分，還要除以 min(列聯表行數, 列聯表列數) 再開根號)
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1)) # (k-1)*(r-1) 自由度
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
    '''


# 連續數值欄位
df_nums = ['v05_1', 'v05_2', 'v06_1', 'v07_1', 'v08_1', 'v09_1', 'v10_1', 
          'v11_1', 'v11_2', 'v12_1','v13_1', 'v13_2', 'v14_1', 'v14_2', 
          'v15_1', 'v15_2', 'v16_1', 'v16_2', 'v18', 'v18_1', 'v21', 
          'v23', 'v23_1', 'v24', 'v28', 'v29']

# 類別欄位
df_cats = list(set(df.columns) - set(df_nums))

# 類別欄位與 target 的相關度
cat_corr = pd.DataFrame()

for col in df_cats:
    corr_d = []
    corr = cramers_v(col, 'target')
    corr_d.append(corr)
    
    cat_corr = pd.concat([cat_corr, pd.DataFrame(corr_d, columns = [col])], axis = 1)

cat_corr = cat_corr.rename(index = {0 : 'corr'}).T
cat_corr = cat_corr.drop('target')
print(f'相關係數最高為 {max(abs(cat_corr["corr"])):.3f}，為 {cat_corr[cat_corr["corr"] == max(cat_corr["corr"])].index.tolist()[0]}')
cat_corr = cat_corr.sort_values(by = 'corr', key = abs, ascending = False)
cat_candi = list(cat_corr[:int(len(cat_corr)/2)].index)

# 數值欄位
num_corr = df[df.columns.intersection(df_nums)]
num_corr = pd.concat([num_corr, df['target']], axis = 1).rename(columns = {'target' : 'corr'})
num_corr = num_corr.corr()['corr'].to_frame()
num_corr = num_corr.drop('corr')
print(f'相關係數最高為 {max(abs(num_corr["corr"])):.3f}，為 {num_corr[num_corr["corr"] == max(num_corr["corr"])].index.tolist()[0]}')
num_corr = num_corr.sort_values(by = 'corr', key = abs, ascending = False)
num_candi = list(num_corr[:int(len(num_corr)/2)].index)

#%% 實驗資料集

# 利用相關係數挑選特徵子集
#train = pd.concat([df[cat_candi], df[num_candi], df['target']], axis = 1)
#x = train.iloc[:, :-1]
#y = train.target

# 不挑選特徵
x = df.iloc[:, :-1]
y = df.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 5553)
# x_train.shape # (4,473, 200)
# x_test.shape # (397, 200)
# y_train.groupby(y_train).count()/len(y_train) # 1,367 (0.30) / 1,678 (0.38) / 1,428 (0.32)
# y_test.groupby(y_test).count()/len(y_test) # 146 (0.29) / 198 (0.40) / 153 (0.31)

#%% decision tree
file_path = 'best_decision_tree.joblib'

def build_tree(d, s, best_acc, x_train, y_train, x_test, y_test):
    clf = DecisionTreeClassifier(max_depth = d, min_samples_leaf = s)
    clf = clf.fit(x_train, y_train)
    
    y_pred = clf.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    
    # 儲存最佳模型
    if acc > best_acc:
        joblib.dump(clf, file_path)
        print(f'更新最佳模型：深度 {d}、最小樣本數 {l}、ACC: {acc:.3f}')
        
    return clf, y_pred, acc

# 深度、min_samples
depth = [3, 4, 5, 6, 7, 8, 9, 10]
min_samples = [1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40 , 45, 50, 55, 60]  

acc_dic = defaultdict(lambda : defaultdict(float))
best_acc = 0.607645875251509
best_d = 0
best_l = 0

for d in depth:
    for l in min_samples:
        clf, _, acc = build_tree(d, l, best_acc, x_train, y_train, x_test, y_test)
        acc_dic[d][l] = acc
        if acc > best_acc:
            best_acc = acc
            best_d = d
            best_l = l

print(f'最佳模型：深度 {best_d}、最小樣本數 {best_l}、ACC: {best_acc:.3f}')

loaded_clf = joblib.load(file_path)
loaded_pred = loaded_clf.predict(x_test)
print('acc: ', metrics.accuracy_score(y_test, loaded_pred))
print(metrics.confusion_matrix(y_test, loaded_pred))
#最佳模型：深度 6、最小樣本數 20、ACC: 0.610

#%% 繪圖
features = list(x.columns)
labels = ['traditionalists', 'techys', 'omnivores']
    
dot_data = tree.export_graphviz(loaded_clf,
                                out_file = None,
                                feature_names = features,
                                class_names = labels,
                                filled = True)

graph = graphviz.Source(dot_data, format = 'png')
graph

#%% 特徵重要性
feature_importances = loaded_clf.feature_importances_

not_zero = 0
for i in range(len(feature_importances)):
    if feature_importances[i] != 0:
        not_zero += 1
print(f'共有 {not_zero} 個非 0 特徵')

for i in range(5):
    max_importances =  max(feature_importances)
    max_feature = np.where(feature_importances == max_importances)
    feature_importances = np.delete(feature_importances, max_feature)
    max_feature = x_train.columns[max_feature[0].tolist()].tolist()[0]
    print(f'影響第{i + 1}高的特徵是{max_feature}，影響力為{max_importances:.3f}')
    

#%% decision path
feature = loaded_clf.tree_.feature # 用於分割節點的特徵
threshold = loaded_clf.tree_.threshold # 節點分隔的判斷準則
node_indicator = loaded_clf.decision_path(x_test) # 每個樣本的決策路徑；csr_matrix
leaf_id = loaded_clf.apply(x_test) # 每個樣本最後落於哪片葉子

sample_id = 3
# 取出 sample_id 的決策路徑
node_idx = node_indicator.indices[node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]] # indptr: get row(s) in a csr_matrix

for node_id in node_idx:
    if leaf_id[sample_id] == node_id: # sample_id 落於哪個葉子
        continue
    
    if x_test.iloc[sample_id][feature[node_id]] <= threshold[node_id]:
        threshold_sign = '<='
    else: threshold_sign = '>='
    
    print(
         "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
         "{inequality} {threshold})".format(
             node=node_id,
             sample=sample_id,
             feature=feature[node_id],
             value=x_test.iloc[sample_id][feature[node_id]],
             inequality=threshold_sign,
             threshold=threshold[node_id],
             )
         )

# v11_1 是 feature[14]
x_test[x_test['v11_1'] > 0]
x_test.index.get_loc(1036)
x_test.loc[1036]['v11_1']
list(x.columns)[25]

#%% 雙北模型
taipei = df[(df['v01_1.0'] == 1) | (df['v01_2.0'] == 1)]
x = taipei.iloc[:, :-1]
y = taipei.target
taipei_x_train, taipei_x_test, taipei_y_train, taipei_y_test = train_test_split(x, y, test_size = 0.1, random_state = 5553)

file_path = 'taipei_decision_tree.joblib'

acc_dic = defaultdict(lambda : defaultdict(float))
best_acc = 0.6229508196721312
best_d = 0
best_l = 0


for d in depth:
    for l in min_samples:
        clf, _, acc = build_tree(d, l, best_acc, taipei_x_train, taipei_y_train, taipei_x_test, taipei_y_test)
        acc_dic[d][l] = acc
        if acc > best_acc:
            best_acc = acc
            best_d = d
            best_l = l

print(f'最佳模型：深度 {best_d}、最小樣本數 {best_l}、ACC: {best_acc:.3f}')

loaded_clf = joblib.load(file_path)
loaded_pred = loaded_clf.predict(taipei_x_test)
print('acc: ', metrics.accuracy_score(taipei_y_test, loaded_pred))
print(metrics.confusion_matrix(taipei_y_test, loaded_pred))
# 最佳模型：深度 10、最小樣本數 15、ACC: 0.623

#%% random forest
file_path = 'best_random_forest.joblib'

def build_forest(d, s, best_acc):
    clf = RandomForestClassifier(max_depth = d, min_samples_leaf = s)
    clf = clf.fit(x_train, y_train)
    
    y_pred = clf.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    
    # 儲存最佳模型
    if acc > best_acc:
        joblib.dump(clf, file_path)
        print(f'更新最佳模型：深度 {d}、最小樣本數 {l}、ACC: {acc:.3f}')
        
    return clf, y_pred, acc

# 深度、min_samples
depth = [3, 4, 5]#, 6, 7, 8, 9, 10]
min_samples = [3, 4, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40 , 45, 50, 55, 60]  

acc_dic = defaultdict(lambda : defaultdict(float))
best_acc = 0.6277665995975855
best_d = 0
best_l = 0

for d in depth:
    for l in min_samples:
        clf, _, acc = build_forest(d, l, best_acc)
        acc_dic[d][l] = acc
        if acc > best_acc:
            best_acc = acc
            best_d = d
            best_l = l           

print(f'最佳模型：深度 {best_d}、最小樣本數 {best_l}、ACC: {best_acc:.3f}')

loaded_clf = joblib.load(file_path)
loaded_pred = loaded_clf.predict(x_test)
print('acc: ', metrics.accuracy_score(y_test, loaded_pred))
print(metrics.confusion_matrix(y_test, loaded_pred))
#最佳模型：深度 8、最小樣本數 1、ACC: 0.628

#%% 特徵重要性
feature_importances = loaded_clf.feature_importances_

not_zero = 0
for i in range(len(feature_importances)):
    if feature_importances[i] != 0:
        not_zero += 1
print(f'共有 {not_zero} 個非 0 特徵')

for i in range(6):
    max_importances =  max(feature_importances)
    max_feature = np.where(feature_importances == max_importances)
    feature_importances = np.delete(feature_importances, max_feature)
    max_feature = x_train.columns[max_feature[0].tolist()].tolist()[0]
    print(f'影響第{i + 1}高的特徵是{max_feature}，影響力為{max_importances:.3f}')

#%% decision path
feature = loaded_clf.tree_.feature # 用於分割節點的特徵
threshold = loaded_clf.tree_.threshold # 節點分隔的判斷準則
node_indicator = loaded_clf.decision_path(x_test) # 每個樣本的決策路徑；csr_matrix
leaf_id = loaded_clf.apply(x_test) # 每個樣本最後落於哪片葉子

sample_id = 0
# 取出 sample_id 的決策路徑
node_idx = node_indicator.indices[node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]] # indptr: get row(s) in a csr_matrix

for node_id in node_idx:
    if leaf_id[sample_id] == node_id: # sample_id 落於哪個葉子
        continue
    
    if x_test.iloc[sample_id][feature[node_id]] <= threshold[node_id]:
        threshold_sign = '<='
    else: threshold_sign = '>='
    
    print(
         "decision node {node} : (X_test[{sample}, {feature}] = {value}) "
         "{inequality} {threshold})".format(
             node=node_id,
             sample=sample_id,
             feature=feature[node_id],
             value=x_test.iloc[sample_id][feature[node_id]],
             inequality=threshold_sign,
             threshold=threshold[node_id],
             )
         )
    
# v11_1 是 feature[14]
x_test[x_test['v11_1'] > 0]
x_test.index.get_loc(1036)
x_test.loc[1036]['v11_1']
list(x.columns)[25]    

(node_indicator, _) = loaded_clf.decision_path(x_test)
print(node_indicator)
