from OnlineRNNObject import OnlineRNNUpdate
import pandas as pd 
import numpy as np 
import pdb as pdb
import matplotlib.pyplot as plt 
import seaborn as sns 

sns.set(style='darkgrid')

# -----LOADING THE DATASET --------------------

data = pd.read_csv("spring_mass_damper_data.csv")
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
onlineRNN = OnlineRNNUpdate(time_steps=time_steps, for_period=for_periods, dt=dt, duration=duration, horizon=time_horizon)
onlineRNN.initializeModel(hidden_neurons=64)

N = len(onlineRNN.t)-for_periods-time_horizon
onlineRNN.createBuffer(buffer_size=100, n_features=2)

#-----------TRYING TO PLOT SHIT ------------------------

plt.ion() 
fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(2, 1, 1) 
ax2 = fig.add_subplot(2,1,2)
loss_, = ax2.plot([], [], label=' MSE')


ax1.legend()
# ax2.legend()
plt.show(block=False)
# -------------


for i in range(N): 
    X_train, Y_train = onlineRNN.createTrainingBatch()

    if i>=onlineRNN.buffer_size:
        onlineRNN.trainStep(X_train, Y_train, n_epochs=5)
        
    

    new_input = onlineRNN.initializeForecastArray()

    for k in range(1, onlineRNN.forecast.shape[0]): 
        u_new = u[i+k]
        new_input, next_pred = onlineRNN.predictAndForecast(u_new, new_input, k=k)
    
    onlineRNN.storeForecast()

    loss = onlineRNN.forecastLoss(onlineRNN.forecast, y, i)
    # onlineRNN.losses.append((loss))

    # ------------ plotting -----------------
    forecast = onlineRNN.forecast
    print('This is my', i, 'th forecast')
    # print(forecast)

    forecast_coords = np.array([[onlineRNN.t[i+j], forecast[j]] for j in range(len(forecast))])
    forecast_t = forecast_coords[:, 0]
    forecast_y = forecast_coords[:, 1]

    ax1.clear() 
    ax1.plot()
    ax1.plot(forecast_t, forecast_y, label='forecast')
    ax1.plot(onlineRNN.t[:i], y[:i], label='ground truth')
   

    loss_.set_data(onlineRNN.t[:i], np.array(onlineRNN.losses[:i]))
    # ax2.axis('auto')
    

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('System Output')
    
    plt.legend()
    plt.draw() 
    plt.pause(0.1)


    
    onlineRNN.updateBuffer(u[i+1], y[i+1])


plt.ioff()



