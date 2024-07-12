import pandas as pd
import NUMPY as np
import matplotlib . PY plot as
%matplotlib inline
from matplotlib . PY lab import r c Params
r c Params ['figure . fig size']=20,10
from KERAS .models import Sequential
from KERAS .layers import LSTM, Dropout ,Dense
from SK learn .preprocessing import Min Max Scaler
df=pd. Read _csv("NSE-TATA.csv")
df. head()
df["Date"]=pd .to _datetime(df .Date ,format="%Y-%m-%d")
df .index=df['Date']
PLT .figure(fig size=(16,8))
PLT .plot(df["Close"],label='Close Price history
data=df.sort_index(ascending=True,axis=0)new_dataset=pd.DataFrame(index=r
ange(0,len(df)),columns=['Date','Close'])
for I in range(0,len(data)):
new _ dataset["Date"][I]=data['Date'][I]
new _dataset["Close"][I]=data["Close"][I]
scaler=Min Max Scaler(feature _range=(0,1))
final _dataset=new _dataset .values
 train _data=final _dataset[0:987,:]
valid _data=final _dataset[987:,:]
new _dataset .index=new _dataset .Date
new _dataset .drop("Date" ,axis=1,inplace=True)
scaler=Min Max Scaler(feature _range=(0,1))
scaled _data=scaler .fit _transform (final _dataset)
x _train _data ,y _train _data=[],[]
for I in range(60,len(train _data)):
x _train _data .append(scaled _data[i-60:i,0])
y_train_data.append(scaled_data[i,0])x_train_data,y_train_data=np.array(x_trai
n_data),np.array(y_train_data)x_train_data=np.reshape(x_train_data,(x_train_d
ata.shape[0],x_train_data.shape[1],1)lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_
data.shape[1],1)))lstm_model.add(LSTM(units=50))
LSTM _model .add(Dense(1))
inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs _data=inputs _data .reshape(-1,1)
inputs _data=scaler .transform(inputs _data)
LSTM _model .compile(loss='mean _squared _error ',optimizer='a dam')
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)
X _test=[]
for I in range(60,inputs_data.shape[0]):
X _test .append(inputs _data[i-60:i,0])
 X _test=np .array(X _test)
X _test=np .reshape(X _test,(X _test .shape[0],X _test .shape[1],1))
Predicted _closing _price=LSTM _model .predict(X _test
predicted_closing_price=scaler.inverse_transform(predicted_closing_price
LSTM _model .save("saved_model.h5")
Train _data=new _dataset[:987]
Valid _data=new _dataset[987:]
Valid _data['Predictions']=predicted _closing _price
#plt.plot(train_data["Close"])
