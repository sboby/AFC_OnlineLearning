import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import math as m 
import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM , Dense, Flatten, Input
from keras.optimizers import Adam 
from tensorflow import keras 
import pdb as pdb 


class ClassicalFNN: 

    def __init__(self, n_layers, time_steps, for_period, dt, duration, horizon, 
                 n_controlInputs, n_controlOutputs): 
        
        self.n_layer            = n_layers
        self.time_steps         = time_steps
        self.for_period         = for_period
        self.dt                 = dt 
        self.duration           = duration
        self.time_horizon       = horizon 

        self.n_controlInputs    = n_controlInputs
        self.n_controlOutputs   = n_controlOutputs
        self.n_InputFeatures    = n_controlInputs + n_controlOutputs

        self.t                  = np.linspace(0, duration, 1000)
        self.model              = None 


    def normalizeData(self, df): 

        zero_ = (df ==0).all() 

        df_scaled = df.copy() 

        for column in df.columns: 
            if not zero_[column]:
                df_scaled[column] = 2 * (df[column] - df[column].min()) / (df[column].max() - df[column].min()) - 1
            else:
                df_scaled[column] = 0  # or any other value or handling you'd like for all-zero columns

        return df_scaled  # You probably want to return the scaled dataframe

    def preprocessData(self):
        pass 


    def initializeBaseModel(self): 
        self.model = Sequential() 
        self.model.add(Flatten(input_shape=(self.time_steps, 3)))
        self.model.add(Dense(64, activation='tanh'))
        self.model.add(Dense(64, activation='tanh'))
        self.model.add(Dense(1))

        self.optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
        self.model.compile(optimizer=self.optimizer, loss='mse')

        # -----------------------------------

        # # multi input network 
        # # one head needs to have the future control sequences 
        # # one head takes the input that has the forward pass conducted on it 

        # # 1. definte the input layers 
        # input_x = tf.keras.layers.Input(shape=(5,3), name='x_input') 
        # input_z = tf.keras.layers.Input(shape=(self.time_horizon. self.n_controlInputs), 
        #                                 name='z_input')
        # # 2. Build the model 
        # combined = tf.layers.Concatenate() 

        # # inputs = Input(shape=(5,3))
        # # control_sequences = Input(shape=(self.time_horizon, self.n_controlInputs))
        # # x = Dense(64, activation='tanh')(inputs)
        # # prediction = Dense(1, activation='softmax')(x)

        # # self.model = keras.models.Model(inputs=[inputs, control_sequences], 
        # #                                 outputs=prediction)
        # # self.optimizer = Adam(learning_rate=0.001, clipvalue=0.5)

        # # self.model.compile(optimizer='adam', 
        # #                    loss=self.forcastMSE)

        return self.model


    def trainTestSplit(self, X_data, Y_data,  test_size): 
        '''
        n_batches : how many seperate cases are in the complete dataset 
                    e.g 3 step response cases are considered for the step response 
                    dataset class (2.02)

        X_data  : control input + corresponding system output [ q1(t), q2(t), rt(t) ]
        Y_data  : corresponding system output r(t)
        test_size : percentage of data for testing 
        
        '''
        n_trainingSamples = int(len(X_data)*(1-test_size))


        X_train_raw, X_test_raw = X_data[:n_trainingSamples, :], X_data[n_trainingSamples:, :]
        Y_train_raw, Y_test_raw = Y_data[:n_trainingSamples], Y_data[n_trainingSamples:]


        max_samples_train = X_train_raw.shape[0] - self.time_steps - self.time_horizon

        # ------------------------- creating training -------------------
        X_train     = np.zeros((max_samples_train, self.time_steps, self.n_InputFeatures))
    
        # (future control inputs so that network can build prediction horizon )
        control_sequences  = np.zeros((max_samples_train, self.time_horizon, self.n_controlInputs))
        
        # so that we calculate the loss on the horizon 
        # ground truth for prediction horizons 
        Y_train     = np.zeros((max_samples_train, self.time_horizon)) 


        for i in range(self.time_steps, max_samples_train):

            X_train[i, :, 0:self.n_controlInputs] = X_train_raw[i - self.time_steps : i]
            X_train[i, :, -1]                     = Y_train_raw[i - self.time_steps : i]

            control_sequences[i, :, 0:self.n_controlInputs] = X_train_raw[i : i + self.time_horizon]

            Y_train[i, : ] = Y_train_raw[i  : i + self.time_horizon] # ground truth of time horizon 


        # ---------------------------creating test data -----------------------
        # max_samples_test = X_test_raw.shape[0] - self.time_steps - self.time_horizon
        
        
        # X_test                  = np.zeros((max_samples_test, self.time_steps, self.n_InputFeatures))
        # control_sequences_test  = np.zeros((max_samples_test, self.time_horizon, self.n_controlInputs))
        # Y_test                  = np.zeros((max_samples_test, self.time_horizon))

        # for j in range(self.time_steps, max_samples_test): 
        #     X_test[j, :, 0:self.n_controlInputs] = X_test_raw[j - self.time_steps : j]
        #     X_test[j, :, -1]                     = Y_test_raw[j - self.time_steps : j]

        #     control_sequences_test[j, :, 0:self.n_controlInputs] = X_test_raw[j : j + self.time_horizon]

        #     Y_test[j, : ] = Y_test_raw[j  : j + self.time_horizon] # ground truth of time horizon 

    
        return X_train, Y_train, control_sequences
    



    def predictForecast(self, ground_truth, control_sequence, input_data):
        ''''
        control_seqeunces : this should be the sequence of control actions that 
                            will follow in the next self.time_horizon timesteps. 

        input_data : this would be the past 5 control inputs and corresponding 
                    system outputs at that point 

        --------------------------------------------------------
        output : this is used to recursively build the desired prediction horizon 

        '''
        
        predictions = []

        current_input = input_data

        with tf.GradientTape() as tape :
            tape.watch(self.model.trainable_variables)

            for j in range(self.time_horizon): 

            
                #--- changed model.predict to just model 
                prediction = self.model(current_input, control_sequence)

                # pdb.set_trace()

                predictions.append(prediction)

                u_next = control_sequence[j]

                current_input[:-1]   = current_input[1:]

                current_input[0][-1][:self.n_controlInputs] = u_next 
                current_input[0][-1,-1] = prediction 

            
            loss = self.forcastMSE(ground_truth, predictions)
        
        # pdb.set_trace()
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
            
        return loss, gradients, np.array(predictions)

    def forcastMSE(self, prediction, ground_truth): 
        
        # MSE = np.mean((prediction-ground_truth)**2)
        return tf.reduce_mean(tf.square(ground_truth - prediction))
 
      

    def backprop(self, gradients) : 
        
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))



    def createBatches(self, X_train, Y_train, control_sequences, batch_size): 
        
        
        x_batches = []
        y_batches = []

        for step in range(0, X_train.shape[0],batch_size): 
            x_batch  = X_train[step: step + batch_size]
            y_batch  = Y_train[step: step + batch_size]

            x_batches.append(x_batch)
            y_batches.append(y_batch)
        
        return x_batches, y_batches
    
    def trainRegime(self, X_train, Y_train, control_sequences , 
                    n_epochs, batch_size):
        ''''
        X_train           : Complete dataset X_train data - control inputs 
                            and corresponding system outputs [q1, q2, rt]
        Y_train           : Complete dataset Y_train data - ground truths for 
                            the entire prediction horizon 
        control_sequences : Set of future control sequences for each X_train 
                            sample to allow for recursive prediction of large 
                            time horizons 
        n_epochs          : Number of training epochs per batch 

        batch_size        : Number of samples in each batch (X_train is split 
                            into batches for training - each batch is iterated 
                            for n_epochs number of times)
        
        '''

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
                    input_data = x_batch[sample].reshape((1,self.time_steps, 
                                                          self.n_InputFeatures))

                    ground_truth = tf.convert_to_tensor(y_batch[sample])

                    loss, gradients,forecast = self.predictForecast(ground_truth, 
                                                                    control_sequence, 
                                                                    input_data)
    

                    batch_forecasts.append(forecast)
                    batch_losses.append(loss)

                    self.backprop(gradients)
                
                    print(f"step {step + 1}: loss = {loss:.4f}")

                    

    def livePlot(self, forecast):
        forecast = forecast.reshape(50,)

        # forecast_coord = np.array([[self.t[i+j]]])
        pass 