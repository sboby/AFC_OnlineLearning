from OnlineLSTMObject import OnlineLSTM
import pandas as pd 
import numpy as np 
import pdb as pdb
import matplotlib.pyplot as plt 
import seaborn as sns 

sns.set(style='darkgrid')

# -----LOADING THE DATASET --------------------

data = pd.read_csv("springMassDamperData.csv")
data.columns = ["control input", "output"]

u = np.array(data['control input'])
y = np.array(data['output'])

#-----SIMUALTION PARAMETERS---------------------
time_steps = 5
for_periods = 1  # recursive prediction 
dt = 0.01
duration = 5 
time_horizon = 100


# ---------NETWORK INITIALIZATION----------------------------------------
onlineLSTM = OnlineLSTM(time_steps=time_steps, for_period=for_periods, dt=dt, duration=duration, horizon=time_horizon)
onlineLSTM.initializeModel(hidden_neurons=64)

N = len(onlineLSTM.t)-for_periods-time_horizon
onlineLSTM.createBuffer(buffer_size=100, n_features=2)
n_epochs = 5

#----- STORING TRAINING LOSSES --------------- 

training_losses = np.zeros((N))

#-----------TRYING TO PLOT SHIT ------------------------

plt.ion() 
fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(2, 1, 1) 
ax2 = fig.add_subplot(2,1,2)
loss1_, = ax2.plot([], [], label=' epoch 1')
loss2_, = ax2.plot([], [], label=' epoch 2')
loss3_, = ax2.plot([], [], label=' epoch 3')
loss4_, = ax2.plot([], [], label=' epoch 4')
loss5_, = ax2.plot([], [], label=' epoch 5')


ax1.legend()
# ax2.legend()
plt.show(block=False)
plt.legend()
# -------------


for i in range(N): 
    X_train, Y_train = onlineLSTM.createTrainingBatch()

    if i>=onlineLSTM.buffer_size:
        onlineLSTM.trainStep(X_train, Y_train, n_epochs=n_epochs)
        
    

    new_input = onlineLSTM.initializeForecastArray()

    for k in range(1, onlineLSTM.forecast.shape[0]): 
        u_new = u[i+k]
        new_input, next_pred = onlineLSTM.predictAndForecast(u_new, new_input, k=k)
    
    onlineLSTM.storeForecast()

    loss = onlineLSTM.forecastLoss(onlineLSTM.forecast, y, i)
    
    training_losses[i] = loss

    # ------------ plotting -----------------
    forecast = onlineLSTM.forecast
    

    forecast_coords = np.array([[onlineLSTM.t[i+j], forecast[j]] for j in range(len(forecast))])
    forecast_t = forecast_coords[:, 0]
    forecast_y = forecast_coords[:, 1]

    ax1.clear() 
    ax1.plot()
    ax1.plot(forecast_t, forecast_y, label='Network Forecast')
    ax1.plot(onlineLSTM.t[:i], y[:i], label='Actual System Behavior')
    ax1.set_title(' Online Rescusive LSTM Forecasting Performance')
    
    loss1_.set_data(onlineLSTM.t[:i], training_losses[:i])
    
    ax2.set_ylim(-0.05, 0.2)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_title('Network Learning Loss')

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('System Output')
    
    plt.legend()
    plt.draw() 
    plt.pause(0.1)


    
    onlineLSTM.updateBuffer(u[i+1], y[i+1])

plt.savefig('LSTM_Results.png')
plt.ioff()



