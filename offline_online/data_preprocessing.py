import pandas as pd

df = pd.read_csv('2018_2019_raw.csv')
# 會產生一列 nan，先刪除
# 刪除 no1（流水號）和 year （全部是2018）
df.drop(df.tail(1).index, inplace = True)
df = df.drop(['no1', 'year'], axis = 1)

df.head()

# 資料整理
# 資料使用998、999等數值代表「不知道」、「不回答」、「跳答」，以 0 取代
# v33, v35 用 98、99
cols_999 = ['v05_1', 'v05_2', 'v06_1', 'v07_1', 'v08_1',
            'v09_1', 'v10_1', 'v11_1', 'v11_2', 'v12_1',
            'v13_1', 'v13_2', 'v14_1', 'v14_2', 'v15_1',
            'v15_2', 'v16_1', 'v16_2', 'v18', 'v18_1',
            'v21', 'v23', 'v23_1', 'v24', 'v28', 'v29']

cols_99 =['v33', 'v35', 'v36', 'v37', 'v38']

for col in cols_999:
    df[col] = df[col].apply(lambda x : 0 if x == 998 or x == 999 else x)
    
for col in cols_99:
    df[col] = df[col].apply(lambda x : 0 if x == 98 or x == 99 else x)

# 類別型資料處理
# 如資料原本有編碼為 90 的「其他」類別，將90改為接續編號
# 如資料類別沒有 90 「其他」，則將類別代碼全部減1

cols_90 = ['v36', 'v37', 'v38']
  
for col in cols_90:
    l = list(set(df[col]))
    l.remove(max(l))
    cat = max(l) + 1
    df[col] = df[col].apply(lambda x : cat if x == 90 else x)

# v39 性別調整為 男1 女0
df['v39'] = df['v39'].apply(lambda x : 0 if x == 2 else 1)

# 類別欄位 one hot encoding
one_hot_cols = ['v01', 'v33', 'v34', 'v35', 'v36', 'v37', 'v38']

# 移除 df 中的類別欄位、加上 one_hot 欄位
for col in one_hot_cols:
    one_hot = pd.get_dummies(df[col], prefix = col, dtype = int)
    df = pd.concat([df, one_hot], axis = 1)
    df = df.drop([col], axis = 1)

print(df.head())

# 產生 target：看現場節目 vs 看線上節目（不管付不付費）
# v13：現場古典&傳統音樂
# v14：現場戲劇
# v15：現場傳統戲曲
# v16：現場舞蹈
# v26_10：線上觀看藝術表演節目（含 古典與傳統音樂、現代戲劇、傳統戲曲、舞蹈）
# v27_10：（付費）線上觀看藝術表演節目（回答是的，26_10 都為 是）

def had_offline(classical, theater, traditional, dance):
    if classical + theater + dance >= 1:
        return 1
    else: return 0

def had_online(free, paid):
    if free + paid >= 1:
        return 1
    else: return 0

def off_on_group(offline, online):
    if offline == 1 and online == 0:
        return 0
    elif offline == 0 and online == 1:
        return 1
    elif offline == 1 and online == 1:
        return 2
    else: return 3

# 若曾參與或欣賞任一表演藝術：「古典與傳統音樂類」、「現代戲劇類」、「傳統戲曲類」、「舞蹈類」，則為1；都沒參與或欣賞的為0。
df['offline'] = df.apply(lambda x : had_offline(x.v13, x.v14, x.v15, x.v16), axis = 1)
df['online'] = df.apply(lambda x : had_online(x.v26_10, x.v27_10), axis = 1)

# 若 offline 為 1，target = 0 （1,513 筆 / 0.149）
# 若 online 為 1，target = 1 （1,876 筆 / 0.185）（不含 3 的話，佔 0.37746478873239436)
# 若 offline & online 皆為 1，target = 2 （1,581 筆 / 0.156）
# 都沒有看，target = 3 （5,176 筆 / 0.510）
df['target'] = df.apply(lambda x : off_on_group(x.offline, x.online), axis = 1)
#print(df['target'].groupby(df['target']).count())
#print(df['target'].groupby(df['target']).count()/len(df['target']))

# 刪除 target = 0 的人
# 主要是想知道看線下、線上、兩個都看的人的差別
df = df[df['target'] != 3]

# 刪除不需要放到訓練集的欄位
drop_cols = ['offline', 'online', 'w1', 'w', 
             # 現場演出相關的欄位
             'v13', 'v13_1', 'v13_2', 'v14', 'v14_1', 'v14_2', 'v15', 'v15_1', 'v15_2', 'v16', 'v16_1', 'v16_2',
             # 線上演出相關欄位
             'v26_10', 'v26_13', 'v27_10', 'v27_14',
             # 刪除 v25_2 去過下列哪種文化機構或使用過下列哪種文化藝術場所：表演藝術中心及專業表演館所　。因為去場館可能代表去看演出。
             'v25_2', 'v25_11']

df = df.drop(drop_cols, axis = 1)
