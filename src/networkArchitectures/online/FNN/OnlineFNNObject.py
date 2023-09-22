import tensorflow as tf 
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM , Dense, Flatten
from keras.optimizers import Adam 

class OnlineFNN:
    def __init__(self, time_steps, for_period, dt, duration, horizon,  
                n_controlInputs, n_controlOutputs): 
         
        self.for_period             = for_period
        self.model                  = None 
        self.optimizer              = None 
        self.buffer_size            = None 
        self.buffer                 = None

        self.n_controlInputs        = n_controlInputs
        self.n_controlOutputs       = n_controlOutputs
        self.n_InputFeatures        = n_controlInputs + n_controlOutputs

        self.total_steps            = int(duration/dt )
        self.t                      = np.linspace(0, duration, 1000)
        self.time_steps             = time_steps 
        self.time_horizon           = horizon 


        self.forecast               = None 
        self.final_predictions      = []
        self.losses                 = []


    def initializeModel(self, hidden_neurons): 
        self.model = Sequential() 
        self.model.add(Flatten(input_shape=(self.time_steps, 2)))
        self.model.add(Dense(100, activation='tanh'))
        self.model.add(Dense(100, activation='tanh'))
        self.model.add(Dense(1))

        self.optimizer = Adam(clipvalue=0.5)
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics=['mse'])

        return self.model

    
    def createBuffer(self, buffer_size): 
        self.buffer_size = buffer_size
        self.buffer = np.zeros((buffer_size, self.n_InputFeatures))

    def createTrainingBatch(self): 
        
        max_samples = self.buffer_size - self.time_steps - self.for_period
        
        U           = self.buffer[:, 0:self.n_controlInputs]
        Y           = self.buffer[:, -1]

        X_train     = np.zeros((max_samples, self.time_steps, self.n_controlInputs))
        Y_train     = np.zeros((max_samples, self.for_period))

        for j in range(0, max_samples): 
             
            X_train[j, :, 0:self.n_controlInputs] = U[j : j + self.time_steps]  
            
            Y_train[j,:] = Y[j: j+ self.for_period]

        return X_train, Y_train

    
    def trainStep(self, X_train, Y_train, n_epochs): 
        X_train_tensor = tf.convert_to_tensor(X_train, dtype=float)
        Y_train_tensor = tf.convert_to_tensor(Y_train, dtype=float)
        
        for i in range(n_epochs): 
            print('trainstep')

            self.model.fit(X_train_tensor, Y_train_tensor, epochs=1, 
                           batch_size=1, shuffle=False)
            
            
    def onlinetrainStep(self, batch_size, n_epochs): 
        U = self.buffer[: , 0:self.n_controlInputs][-self.time_steps-2:-2]
        Y = self.buffer[: , -1]

        X_train     = np.zeros((batch_size, self.time_steps, self.n_controlInputs))
        Y_train     = np.zeros((batch_size, self.for_period))

        X_train[0, :, 0:self.n_controlInputs] = U 
        Y_train                               = Y[-self.for_period]

        X_train_tensor = tf.convert_to_tensor(X_train, dtype=float)
        Y_train_tensor = tf.convert_to_tensor(Y_train, dtype=float)

        loss = self.model.fit(X_train_tensor, Y_train_tensor, batch_size=batch_size, 
                              epochs=n_epochs, shuffle=False)
        
        self.model.set_weights(self.model.get_weights())
        


    
    def initializeForecastArray(self): 
        
        self.forecast = np.zeros((self.time_horizon))
        new_input = np.zeros((self.time_steps, self.n_controlInputs))

        # new_input[:,:] = self.buffer[][-self.time_steps:]

        new_input[:, :]   = self.buffer[:,:self.n_controlInputs][-self.time_steps:]
        # new_input[:, -self.n_controlOutputs:] = self.buffer[:,-self.n_controlOutputs:][-self.time_steps:]
        
        # make a prediction 
        prediction = self.model.predict(np.array([new_input]))
        self.forecast[0] = prediction[0,0]

        return new_input
        

    def predictAndForecast(self, u_new, new_input, k):
        '''
        create forecast of system behavior for for_period 
        number of timesteps ahead 
        '''

        new_input[:-1] = new_input[1:] # get rid of the oldest datapoint 

        new_input[-1][:self.n_controlInputs] = u_new

        # this is skipped for method 1 : past system outputs are not directly a 
        # network input 
        # new_input[-1][-self.n_controlOutputs:] = self.forecast[k-1] # use the last prediction of system behavior 
        # as an input 
        
        next_pred = self.model.predict(np.array([new_input]))
        self.forecast[k] = next_pred[0,0]

        return new_input, next_pred


    def storeForecast(self): 
        self.final_predictions.append(self.forecast)
        print("Forecast Complete")
        

    def updateBuffer(self, u_truth, y, i): 

        self.buffer[:-1,:] = self.buffer[1:,:]
        self.buffer[-1,:self.n_controlInputs] = u_truth [i+1]
        self.buffer[-1][-1] = y[i+1]


    def forecastLoss(self, forecast, ground_truth, i): 
        ground_truth = ground_truth[i+1: i+1+self.time_horizon]
        difference = np.abs(ground_truth-forecast)
        MSE = np.sum(difference**2)/len(ground_truth)
        self.losses.append(MSE)

        return MSE 
    
