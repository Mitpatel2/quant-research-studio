"""Train-only scaling and clustering. Never fit to the evaluation observations."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES=['return4','volatility16','range14','trend16','volume20']


def session_days(index):
    """CME-style Chicago 17:00 opening boundary; label by following date."""
    wall=index.tz_convert('America/Chicago').tz_localize(None)
    return (wall+pd.Timedelta(hours=7)).normalize()


def features(df):
    ret=np.log(df.close).diff()
    volume_mean=df.volume.rolling(20).mean()
    volume_std=df.volume.rolling(20).std()
    result=pd.DataFrame(dict(return4=np.log(df.close/df.close.shift(4)),
        volatility16=ret.rolling(16).std(),
        range14=((df.high-df.low)/df.close).rolling(14).mean(),
        trend16=df.close/df.close.rolling(16).mean()-1,
        volume20=(df.volume-volume_mean)/volume_std.replace(0,np.nan)),index=df.index)
    # Gaps break feature continuity. Do not fill missing source candles.
    gap=df.index.to_series().diff().ne(pd.Timedelta(minutes=15)).astype(int).cumsum()
    uninterrupted=gap.groupby(gap).cumcount()>=20
    return result.where(uninterrupted,axis=0).replace([np.inf,-np.inf],np.nan).dropna()


@dataclass
class RegimeModel:
    scaler: object
    cluster: object
    labels: dict
    training_end: pd.Timestamp

    def predict(self,frame):
        if (frame.index<=self.training_end).any():
            raise ValueError('Prediction frame must follow training_end')
        raw=self.cluster.predict(self.scaler.transform(frame[FEATURES]))
        return pd.Series([self.labels[int(i)] for i in raw],index=frame.index,name='regime')


def fit(training):
    if len(training)<100:raise ValueError('Need at least 100 complete training feature rows')
    if not np.isfinite(training[FEATURES]).all().all():raise ValueError('Training features must be finite')
    if len(training[FEATURES].drop_duplicates())<3:raise ValueError('Need at least three distinct training observations')
    scaler=StandardScaler().fit(training[FEATURES])
    cluster=KMeans(n_clusters=3,random_state=42,n_init=20).fit(scaler.transform(training[FEATURES]))
    centers=scaler.inverse_transform(cluster.cluster_centers_)
    order=np.argsort(centers[:,1])
    labels={int(raw):name for raw,name in zip(order,['Lower volatility','Middle volatility','Higher volatility'])}
    return RegimeModel(scaler,cluster,labels,training.index.max())


def walk_forward(df,train_days=60,test_days=40):
    feat=features(df)
    local=session_days(feat.index)
    days=local.unique();records=[];folds=[]
    for pos in range(train_days,len(days),test_days):
        stop=min(pos+test_days,len(days))
        if stop-pos<5:break
        train=feat.loc[(local>=days[pos-train_days])&(local<days[pos])]
        test=feat.loc[(local>=days[pos])&(local<=days[stop-1])]
        model=fit(train);labels=model.predict(test)
        rows=test.join(labels)
        rows['fold']=len(folds)+1;rows['training_end']=str(model.training_end)
        rows['close']=df.close.reindex(rows.index)
        # Outcome column is diagnostic only and is never passed to fit/predict.
        rows['next_return']=np.log(df.close.shift(-1)/df.close).reindex(rows.index)
        contiguous=(df.index.to_series().shift(-1)-df.index.to_series()).eq(pd.Timedelta(minutes=15))
        rows.loc[~contiguous.reindex(rows.index),'next_return']=np.nan
        # Never use the first observation of a later fold as this fold's outcome.
        rows.loc[rows.index.max(),'next_return']=np.nan
        records.append(rows)
        folds.append(dict(fold=len(folds)+1,training_end=str(model.training_end),
            test_start=str(test.index.min()),test_end=str(test.index.max()),
            training_rows=len(train),test_rows=len(test),inertia=float(model.cluster.inertia_),
            scaler_mean=model.scaler.mean_.tolist(),scaler_scale=model.scaler.scale_.tolist(),
            centers_standardized=model.cluster.cluster_centers_.tolist(),labels=model.labels))
    if not records:raise ValueError('Need more history for the requested train/test windows')
    return pd.concat(records),folds
