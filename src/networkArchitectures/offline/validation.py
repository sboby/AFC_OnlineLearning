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

#--------- filtering and scaling data data ------ 
skip = 20 


#-------------------------------------------------
#------------ SIMULATION PARAMETERS -----------
for_periods     = 1  # number of timesteps ahead the network should predict 
dt              = 0.00115547 * skip
duration        = 50 
time_horizon    = 50 # how much into the future to forecast 
horizonFactor   = 5
time_steps      = 5 # number of past data points used as input  
t               = np.linspace(0, duration, 43302)[5498::skip]
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

fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(1, 1, 1) 


ax1.legend()
plt.legend()


# --- Initialize neural network model 

test = df = pd.read_csv('tentacleDataset_2.01.txt', delimiter=' ')

batch = fnn.normalizeData(test)



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


X_train, Y_train, control_sequences, X_test, Y_test, control_sequences_test = fnn.trainTestSplit(X_data=X_data,
                   Y_data=Y_data, 
                   test_size=0.2)



# pdb.set_trace()



loaded_model = tf.keras.models.load_model('test_model.h5')
loaded_model.summary()

for i in range(len(X_test)):

    control_sequence = control_sequences_test[i]
    input_data = X_test[i].reshape((1, fnn.time_steps, fnn.n_InputFeatures))
    ground_truth = tf.convert_to_tensor(Y_test[i])

    predictions = []

    current_input = input_data

   
    for j in range(fnn.time_horizon): 

            
        #--- changed model.predict to just model 
        prediction = loaded_model(current_input, control_sequence)

        # pdb.set_trace()

        predictions.append(prediction)

        u_next = control_sequence[j]

        current_input[:-1]   = current_input[1:]

        current_input[0][-1][:fnn.n_controlInputs] = u_next 
        current_input[0][-1,-1] = prediction 

    
    loss = fnn.forcastMSE(ground_truth, predictions)


    forecast = np.array(predictions)

    forecast_coords = np.array([[t[i+j], forecast[j]] for j in range(len(forecast))])
    forecast_t = forecast_coords[:, 0]
    forecast_y = forecast_coords[:, 1]




    ax1.cla() 
    ax1.plot()
    ax1.plot(t[i:i+time_horizon], Y_test[i], label='True Forecast')
    ax1.plot(forecast_t, forecast_y, label='Network Forecast')
    # ax1.plot(t[:i], rt[:i], label='Actual System Behavior')
    ax1.set_title(' FNN Forecasting Performance: ' + str(i))
    # ax1.set_xlim(t[0], 10)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_xlim(5, 50)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('System Output')
    ax1.legend()


    with open("verification_01.txt", 'a') as file: 
        file.write(np.array2string(forecast) + '\n')

    # with open("verification_losses_01.txt", 'a') as file: 
    #     file.write(np.array2string(loss) + '\n')
    
    # print("Loss:",loss.item())

    plt.pause(0.1)


