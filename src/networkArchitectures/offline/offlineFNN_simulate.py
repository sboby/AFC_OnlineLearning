import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import math as m 
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM , Dense, Flatten, Input
from keras.optimizers import Adam 
from ClassicFNNObject import ClassicalFNN
from io import BytesIO
from PIL import Image 
import math as m 
import seaborn as sns 
import pdb as pdb 
from tensorflow import keras 

sns.set(style='darkgrid')

#-------------DATA DESCRIPTION -----------------
n_controlInputs  = 2
n_controlOutputs = 1
total_features   = n_controlInputs + n_controlOutputs


#-----------LOADING THE DATASET------------------
df1  = pd.read_csv('OfflineDatasets/Class_2.02_StepResponse/stepResponse_2.21.txt', delimiter=' ')

df2  = pd.read_csv('OfflineDatasets/Class_2.02_StepResponse/stepResponse_2.22.txt', delimiter=' ')

df3  = pd.read_csv('OfflineDatasets/Class_2.02_StepResponse/stepResponse_2.23.txt', delimiter=' ')

df4  = pd.read_csv('OfflineDatasets/Class_2.02_StepResponse/stepResponse_2.24.txt', delimiter=' ')


#--------- filtering and scaling data data ------ 
skip = 20 

dataset = [df1, df2, df3, df4]


#-------------------------------------------------
#------------ SIMULATION PARAMETERS -----------
# NOTE: YOU REMOVED THE SKIP THING 
for_periods     = 1  # number of timesteps ahead the network should predict 
dt              = 0.00115547 
duration        = 50 
time_horizon    = 50 # how much into the future to forecast 
horizonFactor   = 5
time_steps      = 5 # number of past data points used as input  
t               = np.linspace(0, duration, 43302)
buffer_size     = 500
# ------------------------------------------------


fnn = ClassicalFNN(
    n_layers= 2, 
    time_steps=time_steps, 
    for_period=for_periods, 
    dt=dt, 
    duration=duration, 
    horizon=time_horizon, 
    n_controlInputs=n_controlInputs, 
    n_controlOutputs=n_controlOutputs
)

# --- Initialize neural network model 

# test = df = pd.read_csv('tentacleDataset_2.01.txt', delimiter=' ')

df = dataset[0]
batch = fnn.normalizeData(df)



# 1. Normalize the Dataset --> find another way to do this 
# df_scaled = 2*(batch - batch.min()) / (batch.max() - batch.min()) - 1 

# 2. Scaled Features [q1, q2, rt]
q1 = np.array(batch['q1'])
q2 = np.array(batch['q2'])
rt = np.array(batch['concentration'])

# Defining X and Y data for the network 
X_data = np.zeros((len(t), n_controlInputs))
X_data[:, 0:n_controlInputs] = np.array([q1,q2]).T

Y_data = rt 

#----------------------------------------------
base_model = fnn.initializeBaseModel()

# pdb.set_trace()


test_size = 0.2 

n_trainingSamples = int(len(X_data)*(1-test_size))


X_train_raw, X_test_raw = X_data[:n_trainingSamples, :], X_data[n_trainingSamples:, :]
Y_train_raw, Y_test_raw = Y_data[:n_trainingSamples], Y_data[n_trainingSamples:]


max_samples = X_train_raw.shape[0] - fnn.time_steps - fnn.time_horizon


# ------- creating training samples for one batch -----------


X_train, Y_train, control_sequences = fnn.trainTestSplit(X_data=X_data,
                   Y_data=Y_data, 
                   test_size=0.2)



fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(1, 1, 1) 


ax1.legend()
plt.legend()


N = q1.shape[0] - time_horizon*2

losses = []
times = []


n_epochs = 1000 
batch_size = 32 

for epoch in range(n_epochs):
            
            
    print(f"\nEpoch {epoch + 1}/{n_epochs}")
            # treating X_train data in batches 

    for step in range(0, X_train.shape[0], batch_size): 
        

        x_batch = X_train[step: step + batch_size]
        y_batch = Y_train[step: step + batch_size]

        batch_losses = []
        batch_forecasts = []

        for sample in range(0, x_batch.shape[0]): 
            control_sequence = control_sequences[sample]
            input_data = x_batch[sample].reshape((1,fnn.time_steps, 
                                                    fnn.n_InputFeatures))

            ground_truth = tf.convert_to_tensor(y_batch[sample])

            loss, gradients,forecast = fnn.predictForecast(ground_truth, 
                                                            control_sequence, 
                                                            input_data)


            batch_forecasts.append(forecast)
            batch_losses.append(loss)

            fnn.backprop(gradients)
        
            print(f"step {step + 1}: loss = {loss:.4f}")

fnn.model.save('test_model_01.h5')

        


# pdb.set_trace()



    # # pdb.set_trace()
    # #----------- plotting -------
    
    # forecast = forecast.reshape(50,)

    # forecast_coords = np.array([[t[i+j], forecast[j]] for j in range(len(forecast))])
    # forecast_t = forecast_coords[:, 0]
    # forecast_y = forecast_coords[:, 1]




    # ax1.cla() 
    # ax1.plot()
    # ax1.plot(t[i:i+time_horizon], rt[i:i+time_horizon], label='True Forecast')
    # ax1.plot(forecast_t, forecast_y, label='Network Forecast')
    # ax1.plot(t[:i], rt[:i], label='Actual System Behavior')
    # ax1.set_title(' FNN Forecasting Performance: ' + str(i))
    # # ax1.set_xlim(t[0], 10)
    # ax1.set_ylim(-1.2, 1.2)
    # ax1.set_xlim(5, 50)
    # ax1.set_xlabel('Time [s]')
    # ax1.set_ylabel('System Output')
    # ax1.legend()


    # with open("verification_01.txt", 'a') as file: 
    #     file.write(np.array2string(forecast) + '\n')

    # # with open("verification_losses_01.txt", 'a') as file: 
    # #     file.write(np.array2string(loss) + '\n')
    
    # # print("Loss:",loss.item())

    # plt.pause(0.1)







