from app import app
app.app_context().push()
import services.model_loader as ml
ml.auto_load_all_models()
from services.feature_engineering import get_scaled_feature_sequence, get_station_dataframe_cached
from services.model_loader import directional_models, directional_scalers
from sklearn.metrics import r2_score
import numpy as np
stations=['North Ave','Cubao','Ortigas','Ayala Ave','Taft']
directions=['Northbound','Southbound']
print('{:<15} {:<12} {:>8} {:>8} {:>6}'.format('Station','Dir','R2_raw','R2_bias','bias'))
print('-'*60)
for s in stations:
    for d in directions:
        mk=s+'_'+d
        model=directional_models.get(mk)
        ts=directional_scalers.get(mk+'_target')
        if model is None or ts is None: continue
        hourly=get_station_dataframe_cached(s,d)
        if hourly is None or len(hourly)<50: continue
        test=hourly.iloc[-300:]
        a=[]; p=[]
        for t,row in test.iterrows():
            feats=get_scaled_feature_sequence(s,d,t)
            if feats is None: continue
            raw=model.predict(feats.reshape(1,24,-1),verbose=0)[0][0]
            a.append(float(row['TotalPassenger']))
            p.append(max(0.0,float(ts.inverse_transform([[raw]])[0][0])))
        a=np.array(a); p=np.array(p)
        bias=a.mean()/p.mean()
        r2r=r2_score(a,p)
        r2b=r2_score(a,p*bias)
        print('{:<15} {:<12} {:>8.3f} {:>8.3f} {:>6.3f}'.format(s,d,r2r,r2b,bias))
