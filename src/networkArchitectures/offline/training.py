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

#--------DATA DESCRIPTION ------------------- 
n_controlInputs     = 2 
n_controlOutputs    = 1 
total_features      = n_controlInputs + n_controlOutputs

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
# --------------TRAINING PARAMETERS----------------------------------
n_epochs = 50
batch_size = 32 



# 1. Load dataset --> 4 cases 
#-----------LOADING THE DATASET------------------
df1  = pd.read_csv('OfflineDatasets/Class_2.02_StepResponse/stepResponse_2.21.txt', delimiter=' ')

df2  = pd.read_csv('OfflineDatasets/Class_2.02_StepResponse/stepResponse_2.22.txt', delimiter=' ')

df3  = pd.read_csv('OfflineDatasets/Class_2.02_StepResponse/stepResponse_2.23.txt', delimiter=' ')

df4  = pd.read_csv('OfflineDatasets/Class_2.02_StepResponse/stepResponse_2.24.txt', delimiter=' ')


# 2. Select cases for training --> 3/4 cases used for training 
#   Remainder case becomes validation/test set 
training_cases = [df1, df2, df3]
validation_case = df4


# 3. Instatitate classical fnn object 
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
base_model = fnn.initializeBaseModel()


# 4. For each of the 3 cases:
#   4.0  normalize all the data 
#   4.1  create batches 
#   4.2  train regime for n epochs (weight updates)


fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(1,1,1)



for idx, case in enumerate(training_cases):

    # 4.0 scale the case data 
    case = fnn.normalizeData(case) 

    q1 = np.array(case['q1'])
    q2 = np.array(case['q2'])
    rt = np.array(case['concentration'])

    X_data = np.zeros((len(t), n_controlInputs))
    X_data[:, 0:n_controlInputs] = np.array([q1,q2]).T

    Y_data = rt 

    X_train, Y_train, control_sequences= fnn.trainTestSplit(X_data=X_data,
                   Y_data=Y_data, 
                   test_size=0.0)
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

                ax1.cla()
                ax1.plot()
                ax1.plot(batch_losses)

                plt.pause(0.1)



fnn.model.save('test_model_02.h5')



# 5. save model weights  