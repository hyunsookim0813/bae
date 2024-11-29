# 지표에 xxx_error 회귀문제
import pandas as pd
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train = train.drop('target',axis=1)
y_train = train['target'] 
X_train = pd.get_dummies(X_train)
test = pd.get_dummies(test)

from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_tr.shape, X_val.shape, y_tr.shape, y_val.shape

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
model = RandomForestRegressor()
model = RandomForestClassifier()
model.fit(X_tr,y_tr)
pred = model.predict(X_val) #roc-auc 만 predict_proba
pred[:10] 
from sklearn.metrics import mean_squared_error
mae = mean_squared_error(y_val, pred)
print(mae)
from sklearn.metrics import f1_score
f1_score(y_val,pred,average='macro')

pred = model.predict(test)
pred[:10]

submit = pd.DataFrame({'pred':pred}) #pred[:,1]
submit.to_csv('result.csv',index=False)
pd.read_csv('result.csv')
---------------------------
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit_transform(df[['target']])

pd.to_datetime(df['time'],format=%Y:%m:%d)
--------------------------
print(dir(scipy.stats))
#단일표본
from scipy import stats
stats.ttest_1samp(df['target'],기준)
stats.ttest_1samp(df['target'],기준,alternative='greater') #대립가설
stats.ttest_1samp(df['target'],기준,alternative='less') #기준보다
#shapiro - 정규성 확인
stats.shapiro(df['target'])
#wilcoxon 비모수
stats.wilcoxon(df['target'],기준,alternative='less')

#대응표본(전, 후)
stats.ttest_rel(df['before']-df['after'],alternative='less') #앞값기준
#shapiro - 정규성 확인
stats.shapiro(df['before']-df['after'])'
#wilcoxon 비모수
stats.wilcoxon(df['before']-df['after'],alternative='less')

#독립표본(그룹간 차이)
from scipy import stats
stats.ttest_ind(a,b) #처리,대조
stats.ttest_ind(a,b,equal_val=True,alternative='less')
#shapiro - 정규성 확인
stats.shapiro(a),stats.shapiro(b)
#mannwhitneyu 비모수
stats.mannwhitneyu(a,b,alternative='less') 

#적합도검정
from scipy import stats
stats.chisquare(관찰,기대)
#독립성검정
df=[[80,20],[90,10]]
stats.chi2_contingency(df)
from statsmodels.formula.api import logit
logit(‘종속 ~ 독립’, data=df).fit()
np.exp(model.params[‘변수’]*5) 5 만큼 증가할 때
df=pd.crosstab(df['a'],df['b'])
stats.chi2_contingency(df)

#상관
df.corr(numeric_only=True)
#단순선형회귀
import statsmodels.formula.api as ols
model = ols('종속~독립',data=df).fit()
model.summary()
#다중선형회귀
model = ols('종속~독립+독립2',data=df).fit()
model.summary()

#일원분산분석
stats.f_oneway(df[a],df[b],df[C],df[d])
#anova table
from statsmodel.formula.api import ols
from statsmodel.anova import anova_lm
model = ols('value~variable'data=df).fit()
anova_lm(model)
#이원분석
from statsmodel.formula.api import ols
from statsmodel.anova import anova_lm
model = ols('value~var+var2+var:var2'data=df).fit()
anova_lm(model)
