import pandas as pd, numpy as np, re, json, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import *
df = pd.read_csv("sms.csv", sep="\t", header=None, names=["label","text"], quoting=csv.QUOTE_NONE).drop_duplicates()
raw_n = 5574
df["y"] = (df.label=="spam").astype(int)
df["len"] = df.text.str.len()
print(len(df), df.y.value_counts().to_dict())
stats = {"raw":raw_n,"clean":len(df),"ham":int((df.y==0).sum()),"spam":int((df.y==1).sum()),
 "len_ham":df[df.y==0].len.mean(),"len_spam":df[df.y==1].len.mean()}
# EDA plots
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
df.label.value_counts().plot.bar(ax=ax[0],color=["#4d4d4d","#b3b3b3"]);ax[0].set_title("Class distribution");ax[0].set_xlabel("");ax[0].tick_params(axis='x',rotation=0)
for l,c in [("ham","#4d4d4d"),("spam","#b3b3b3")]:
    ax[1].hist(df[df.label==l].len,bins=40,alpha=.8,label=l,color=c,edgecolor="black",linewidth=.3)
ax[1].set_title("Message length (characters)");ax[1].legend()
plt.tight_layout();plt.savefig("eda.png",dpi=160);plt.close()
# top words
from sklearn.feature_extraction.text import CountVectorizer
def top(lbl):
    cv=CountVectorizer(stop_words="english",max_features=3000);X=cv.fit_transform(df[df.label==lbl].text)
    s=np.asarray(X.sum(0)).ravel();i=s.argsort()[::-1][:12];return [(cv.get_feature_names_out()[j],int(s[j])) for j in i]
ts,th=top("spam"),top("ham");stats["top_spam"]=ts;stats["top_ham"]=th
fig,ax=plt.subplots(1,2,figsize=(9,3.6))
for a,t,c,n in [(ax[0],ts,"#b3b3b3","Top words in spam"),(ax[1],th,"#4d4d4d","Top words in ham")]:
    a.barh([w for w,_ in t][::-1],[v for _,v in t][::-1],color=c);a.set_title(n)
plt.tight_layout();plt.savefig("words.png",dpi=160);plt.close()
Xtr,Xte,ytr,yte=train_test_split(df.text,df.y,test_size=.2,stratify=df.y,random_state=42)
stats["train"]=len(Xtr);stats["test"]=len(Xte)
tf=lambda:TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1,2),min_df=2,sublinear_tf=True)
models={"Multinomial Naive Bayes":MultinomialNB(alpha=0.1),
"Logistic Regression":LogisticRegression(C=10,max_iter=1000,class_weight="balanced"),
"Linear SVM":LinearSVC(C=1,class_weight="balanced"),
"Random Forest":RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1)}
res=[];fitted={}
for n,m in models.items():
    p=make_pipeline(tf(),m);p.fit(Xtr,ytr);pr=p.predict(Xte);fitted[n]=(p,pr)
    cv=cross_val_score(make_pipeline(tf(),m),Xtr,ytr,cv=5,scoring="f1").mean()
    res.append(dict(Model=n,Accuracy=accuracy_score(yte,pr),Precision=precision_score(yte,pr),Recall=recall_score(yte,pr),F1=f1_score(yte,pr),CV_F1=cv))
R=pd.DataFrame(res).round(4);print(R.to_string())
R.to_csv("results.csv",index=False)
best=R.sort_values("F1",ascending=False).iloc[0].Model;stats["best"]=best
p,pr=fitted[best];cm=confusion_matrix(yte,pr);stats["cm"]=cm.tolist()
ConfusionMatrixDisplay(cm,display_labels=["ham","spam"]).plot(cmap="Greys",colorbar=False);plt.title(f"Confusion matrix - {best}");plt.tight_layout();plt.savefig("cm.png",dpi=160);plt.close()
print(classification_report(yte,pr,target_names=["ham","spam"]))
stats["report"]=classification_report(yte,pr,target_names=["ham","spam"],output_dict=True)
STY={"Multinomial Naive Bayes":("-","black"),"Logistic Regression":("--","#555555"),"Linear SVM":("-.","#888888"),"Random Forest":(":","black")}
fig,ax=plt.subplots(figsize=(4.8,3.8))
for n,(pp,_) in fitted.items():
    s=pp.predict_proba(Xte)[:,1] if hasattr(pp,"predict_proba") else pp.decision_function(Xte)
    f,t,_=roc_curve(yte,s);ls,cl=STY[n];ax.plot(f,t,ls,color=cl,lw=1.4,label=f"{n} (AUC={auc(f,t):.3f})")
ax.plot([0,1],[0,1],"k--",lw=.6);ax.set_xlabel("False positive rate");ax.set_ylabel("True positive rate");ax.set_title("ROC curves");ax.legend(fontsize=6.5,loc="lower right")
plt.tight_layout();plt.savefig("roc.png",dpi=160);plt.close()
# misclassified
mis=Xte[pr!=yte];stats["mis"]=[(t[:110],"spam" if yte[i]==1 else "ham") for i,t in mis.items()][:6]
# feature weights from LR
lr=make_pipeline(tf(),LogisticRegression(C=10,max_iter=1000,class_weight="balanced")).fit(Xtr,ytr)
names=lr[0].get_feature_names_out();w=lr[1].coef_[0];stats["lr_top"]=[names[i] for i in w.argsort()[::-1][:10]]
# demo
demo=["Congratulations! You have won a free prize. Call 09061701461 now to claim","Hey, are we still meeting for lunch at 1?","URGENT! Your mobile number has been selected for a 2000 cash award, txt WIN to 80086","Can you send me the notes from today's lecture?"]
stats["demo"]=[(d,"spam" if p.predict([d])[0] else "ham") for d in demo]
import joblib;joblib.dump(p,"spam_model.joblib")
json.dump(stats,open("stats.json","w"),default=float,indent=1)
print(stats["mis"],stats["lr_top"],stats["demo"])