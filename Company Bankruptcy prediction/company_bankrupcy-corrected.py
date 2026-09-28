
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold
df=pd.read_csv('C:\Users\Norhan El-Feky\Data-Analysis-and-Forecasting-for-Oil-companies-insurance-s-risk-management-\Company Bankruptcy prediction\american_bankruptcy.csv')
m={"X1":"CA","X2":"COGS","X3":"DA","X4":"EBITDA","X5":"Inv","X6":"NI","X7":"Rec","X8":"MV","X9":"NS","X10":"TA","X11":"LTD","X12":"EBIT","X13":"GP","X14":"CL","X15":"RE","X16":"Rev","X17":"TL","X18":"OpEx"}
df=df.rename(columns=m)
eq=df.TA-df.TL
df["Return on Assets (%)"]=df.NI/df.TA*100
df["Return on Equity (%)"]=df.NI/eq*100
df["Gross Profit Margin (%)"]=df.GP/df.Rev*100
df["Operating Margin (%)"]=df.EBIT/df.Rev*100
df["Current Ratio"]=df.CA/df.CL
df["Debt to Equity Ratio"]=df.TL/eq
df["Leverage_Ratio"]=df.TL/eq
df["Receivables_to_Sales"]=df.Rec/df.NS
df["OpEx_to_EBITDA"]=df.OpEx/df.EBITDA.replace(0,np.nan)
df["High_Insolvency_Risk"]=(df.Leverage_Ratio>2)&(df["Current Ratio"]<1)
df["Trade_Credit_Warning"]=df.Receivables_to_Sales>0.30
F=["Return on Assets (%)","Return on Equity (%)","Gross Profit Margin (%)","Operating Margin (%)","Current Ratio","Debt to Equity Ratio","Leverage_Ratio","Receivables_to_Sales","OpEx_to_EBITDA","High_Insolvency_Risk","Trade_Credit_Warning"]
df[F]=df[F].replace([np.inf,-np.inf],np.nan)
df=df.dropna(subset=F).copy()
df['y_orig']=(df.status_label=='failed').astype(int)
# new label: failed within next 1-2 years = last two observed years of a failed firm
last=df.groupby('company_name').year.transform('max')
df['y_new']=((df.status_label=='failed')&(df.year>=last-1)).astype(int)
print('rows',len(df),'firms',df.company_name.nunique(),'orig pos',df.y_orig.sum(),round(df.y_orig.mean(),4),'new pos',df.y_new.sum(),round(df.y_new.mean(),4))
def rf(): return RandomForestClassifier(n_estimators=200,max_depth=10,class_weight='balanced_subsample',random_state=42,n_jobs=-1)
def ev(yt,p,thr):
    yp=(p>=thr).astype(int)
    return dict(auc=roc_auc_score(yt,p),prauc=average_precision_score(yt,p),rec=recall_score(yt,yp),prec=precision_score(yt,yp,zero_division=0),f1=f1_score(yt,yp),acc=accuracy_score(yt,yp),flag_healthy=yp[yt==0].mean(),n=len(yt),pos=int(yt.sum()))
X=df[F].astype(float); g=df.company_name
res={}
# A original
Xtr,Xte,ytr,yte=train_test_split(X,df.y_orig,test_size=.2,random_state=42,stratify=df.y_orig)
mA=rf().fit(Xtr,ytr); res['A_row_split_orig_label']=ev(yte,mA.predict_proba(Xte)[:,1],0.40)
# B firm split, orig label
gss=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42); tr,te=next(gss.split(X,df.y_orig,g))
mB=rf().fit(X.iloc[tr],df.y_orig.iloc[tr]); res['B_firm_split_orig_label']=ev(df.y_orig.iloc[te],mB.predict_proba(X.iloc[te])[:,1],0.40)
# C firm split, new label; threshold chosen on training via OOF to reach 70% recall
yN=df.y_new
mC=rf().fit(X.iloc[tr],yN.iloc[tr]); pC=mC.predict_proba(X.iloc[te])[:,1]
sgk=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
oof=np.zeros(len(tr)); Xt=X.iloc[tr]; yt=yN.iloc[tr]; gt=g.iloc[tr]
for a,b in sgk.split(Xt,yt,gt):
    oof[b]=rf().fit(Xt.iloc[a],yt.iloc[a]).predict_proba(Xt.iloc[b])[:,1]
thr=np.quantile(oof[yt.values==1],0.30)  # 70% recall on training OOF
print('threshold chosen on train OOF for 70% recall:',round(thr,3))
res['C_firm_split_new_label_thr040']=ev(yN.iloc[te],pC,0.40)
res['C_firm_split_new_label_thrR70']=ev(yN.iloc[te],pC,thr)
# D 5-fold group CV new label
aucs=[];pr=[]
for a,b in StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(X,yN,g):
    p=rf().fit(X.iloc[a],yN.iloc[a]).predict_proba(X.iloc[b])[:,1]; aucs.append(roc_auc_score(yN.iloc[b],p)); pr.append(average_precision_score(yN.iloc[b],p))
print('5-fold group CV new label AUC mean %.3f sd %.3f; PR-AUC mean %.3f sd %.3f'%(np.mean(aucs),np.std(aucs),np.mean(pr),np.std(pr)))
# E temporal: train <=2011, test >=2012? firms may overlap; use firm+time: test = firms in test group and years>=2012
out=pd.DataFrame(res).T; pd.set_option('display.width',200); print(out.round(3))
imp=pd.Series(mC.feature_importances_,index=F).sort_values(ascending=False); print(imp.round(3))
# medians new label
print(df.groupby('y_new')[['Return on Assets (%)','Current Ratio','Debt to Equity Ratio']].median().round(2))
print(df.groupby('y_new')[['High_Insolvency_Risk','Trade_Credit_Warning']].mean().round(3))
# tier table with new label
r=df['Return on Assets (%)']; de=df['Debt to Equity Ratio']
neg=r<0; hl=(de>1)|(de<0)
t=np.where(neg&hl,'C',np.where(neg|hl,'B','A'))
tb=df.groupby(t).agg(n=('y_new','size'),rate=('y_new','mean'),fails=('y_new','sum')); tb['share']=tb.n/len(df); tb['fail_share']=tb.fails/tb.fails.sum(); print(tb.round(3))
out.to_csv('bankruptcy_rerun_results.csv')
te_df=df.iloc[te]; print('test firms',te_df.company_name.nunique(),'train firms',df.iloc[tr].company_name.nunique())
yv=yN.iloc[te].values; o=np.argsort(-pC)
for q in [0.05,0.10,0.20]:
    k=int(len(pC)*q); print(q, 'captures %.3f of failures, rate %.3f'%(yv[o[:k]].sum()/yv.sum(), yv[o[:k]].mean()))
print('test base rate',yv.mean())