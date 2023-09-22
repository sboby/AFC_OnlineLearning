import tensorflow as tf 
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM 
from keras.optimizers import Adam 

class OnlineRNNUpdate:
    def __init__(self, time_steps, for_period, dt, duration, horizon): 
        # how many input values 
        self.time_steps = time_steps 
        # how many timesteps into the furture to predict the output 
        self.for_period = for_period
        self.model = None 
        self.optimizer = None 
        self.buffer_size = None 
        self.buffer = None

        self.total_steps = int(duration/dt )
        self.t = np.linspace(0, duration, 1000)
        self.time_horizon = horizon 


        # ------------- storing results -------------------------- 
        # create an array to store all predicitions for the horizon 
        # per timestep a prediction horizon is filled with predictions 
        # ---> forecast 
        self.forecast = None 
        self.final_predictions = []
        self.losses = []


    def initializeModel(self, hidden_neurons): 
        self.model = Sequential() 
        self.model.add(LSTM(hidden_neurons, return_sequences=True))
        self.model.add(LSTM(hidden_neurons))
        self.model.add(Dense(self.for_period))

        self.optimizer = Adam(clipvalue=0.5)
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics=['mse'])

        return self.model

    
    def createBuffer(self, buffer_size, n_features): 
        self.buffer_size = buffer_size
        self.buffer = np.zeros((buffer_size, n_features))

    def createTrainingBatch(self): 
        print('Creating training regime')
        max_samples = self.buffer_size-self.time_steps-self.for_period
        
        U = self.buffer[:,0]
        Y = self.buffer[:,1]

        X_train = np.zeros((max_samples, self.time_steps, 2))
        Y_train = np.zeros((max_samples, self.for_period))

        for j in range(self.time_steps, max_samples): 
            # there's 2 input features now 
            X_train[j,:, 0] = U[j-self.time_steps: j]  
            X_train[j,:, 1] = Y[j-self.time_steps: j]  

            Y_train[j,:] = Y[j: j+ self.for_period]

        return X_train, Y_train

    
    def trainStep(self, X_train, Y_train, n_epochs): 
        X_train_tensor = tf.convert_to_tensor(X_train, dtype=float)
        Y_train_tensor = tf.convert_to_tensor(Y_train, dtype=float)
        
        loss = self.model.fit(X_train_tensor, Y_train_tensor, epochs = n_epochs)

        self.model.set_weights(self.model.get_weights())
        print("Batch loss: ", loss.history['loss'])

    
    
    
    def initializeForecastArray(self): 
        
        self.forecast = np.zeros((self.time_horizon))
        #create the input sequence for the new timestep 
        new_input = np.zeros((self.time_steps,2))
        new_input[:,0] = self.buffer[:,0][-self.time_steps:]
        new_input[:,1] = self.buffer[:,1][-self.time_steps:]
        
        # make a prediction 
        prediction = self.model.predict(np.array([new_input]))
        self.forecast[0] = prediction

        return new_input

        

    def predictAndForecast(self, u_new, new_input, k):
        '''
        create forecast of system behavior for for_period 
        number of timesteps ahead 
        '''

        new_input[:-1] = new_input[1:]
        new_input[-1][0] = u_new
        new_input[-1][1] = self.forecast[k-1]
        
        next_pred = self.model.predict(np.array([new_input]))
        self.forecast[k] = next_pred

        return new_input, next_pred


    def storeForecast(self): 
        self.final_predictions.append(self.forecast)
        print("Forecast Complete")
        

    def updateBuffer(self, u_new, y_new): 
        new_data = np.array([u_new, y_new])
        self.buffer[:-1,:] = self.buffer[1:,:]
        self.buffer[-1,:] = new_data


    def forecastLoss(self, forecast, ground_truth, i): 
        ground_truth = ground_truth[i+1: i+1+self.time_horizon]
        difference = np.abs(ground_truth-forecast)
        MSE = np.sum(difference**2)/len(ground_truth)
        self.losses.append(MSE)

        return MSE 
    
