import tensorflow as tf 
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
from keras.optimizers import Adam 

class OnlineLSTM_Multi: 
    def __init__(self, time_steps, for_period, dt, duration, horizon, 
                n_controlInputs, n_controlOutputs): 
        self.time_steps         = time_steps
        self.for_period         = for_period
        self.model              = None 
        self.optimizer          = None 
        self.buffer_size        = None 
        self.buffer             = None 
        
        self.n_controlInputs    = n_controlInputs 
        self.n_controlOutputs   = n_controlOutputs  
        self.n_InputFeatures    = n_controlInputs + n_controlOutputs

        self.total_steps        = int(duration/dt)
        self.t                  = np.linspace(0, duration, 1000)
        self.time_steps         = time_steps
        self.time_horizon       = horizon

        self.forecast           = None 
        self.final_predictions  = []
        self.losses             = [] 


    def initializeModel(self, hidden_neurons): 
        self.model = Sequential() 
        self.model.add(LSTM(hidden_neurons, return_sequences=True, 
                    input_shape=(self.time_steps, self.n_InputFeatures)))
        self.model.add(LSTM(hidden_neurons))
        self.model.add(Dense(self.for_period))

        self.optimizer = Adam(clipvalue=0.5)
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics=['mse'])

        return self.model

    def createBuffer(self, buffer_size): 
        self.buffer_size = buffer_size
        self.buffer = np.zeros((buffer_size, self.n_InputFeatures))

    def createTrainingBatch(self): 

        max_samples     = self.buffer_size - self.time_steps - self.for_period

        # print('buffer:', self.buffer)
        
        U           = self.buffer[: , 0:self.n_controlInputs]
        Y           = self.buffer[: , -1]

        # print('U shape', U.shape)
        # print('Y shape', Y.shape)

        X_train     = np.zeros((max_samples, self.time_steps, self.n_InputFeatures))
        Y_train     = np.zeros((max_samples, self.for_period))

        # print('X_train shape,', X_train.shape)
        # print('Y_train shape', Y_train.shape)
        

        for j in range(0, max_samples): ## THIS RANGE IS WRONG  
            

            X_train[j, :,0:self.n_controlInputs] = U[j : j + self.time_steps]
            X_train[j, :, -1] = Y[j : j + self.for_period]
            
            Y_train[j, :] = Y[j : j + self.for_period]

        return X_train, Y_train

    
    def trainStep(self, X_train, Y_train, n_epochs): 
        X_train_tensor = tf.convert_to_tensor(X_train, dtype=float)
        Y_train_tensor = tf.convert_to_tensor(Y_train, dtype=float)
        
        loss = self.model.fit(X_train_tensor, Y_train_tensor, epochs = n_epochs)

        self.model.save_weights('weights.txt')
        # self.model.set_weights(self.model.get_weights())
        print("Batch loss: ", loss.history['loss'])

    
    def runtimeReduction_trainStep(self, X_train, Y_train, n_epochs):

        X_train_tensor = tf.convert_to_tensor(X_train, dtype=float)
        Y_train_tensor = tf.convert_to_tensor(Y_train, dtype=float)

        with tf.GradientTape() as tape:  # api to get gradients wrt to inputs 
            predictions = self.model(X_train_tensor)
            loss = tf.losses.mean_squared_error(Y_train_tensor, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss 
    

    def initializeForecastArray(self): 
        self.forecast = np.zeros((self.time_horizon))
        new_input = np.zeros((self.time_steps, self.n_InputFeatures))
        new_input[:, :self.n_controlInputs]   = self.buffer[:,:self.n_controlInputs][-self.time_steps:]
        new_input[:, -self.n_controlOutputs:] = self.buffer[:,-self.n_controlOutputs:][-self.time_steps:]
        
        # make a prediction 
        prediction = self.model.predict(np.array([new_input]))
        self.forecast[0] = prediction

        return new_input

    def predictAndForecast(self, u_new, new_input, k): 
        
        new_input[:-1] = new_input[1:] # get rid of the oldest datapoint 

        new_input[-1][:self.n_controlInputs] = u_new

        new_input[-1][-self.n_controlOutputs:] = self.forecast[k-1] # use the last prediction of system behavior 
        # as an input 
        
        next_pred = self.model.predict(np.array([new_input]))
        self.forecast[k] = next_pred

        return new_input, next_pred
    
    def storeForecast(self): 
        self.final_predictions.append(self.forecast)
        print("Forecast Complete")

    def updateBuffer(self, u_truth, y, i):
        self.buffer[:-1,:] = self.buffer[1:,:]
        self.buffer[-1, :self.n_controlInputs] = u_truth[i+1]
        self.buffer[-1][-1] = y[i+1]

    
    def forecastLoss(self, forecast, ground_truth, i): 
        ground_truth = ground_truth[i+1: i+1+self.time_horizon]
        difference = np.abs(ground_truth-forecast)
        MSE = np.sum(difference**2)/len(ground_truth)
        self.losses.append(MSE)

        return MSE 

