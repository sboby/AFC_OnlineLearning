from OnlineFNNObject import OnlineFNN
import pandas as pd 
import numpy as np 
import pdb as pdb
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib.animation as animation 
from matplotlib.animation import FuncAnimation, PillowWriter
from io import BytesIO
from PIL import Image 
import math as m 
# from sklearn.preprocessing import MinMaxScaler


sns.set(style='darkgrid')

#-------------DATA DESCRIPTION -----------------
n_controlInputs  = 2
n_controlOutputs = 1
total_features   = n_controlInputs + n_controlOutputs


#------------LOADING THE DATASET -------------

df = pd.read_csv('tentacleDataset_2.01.txt', delimiter=' ')

df_scaled = 2*(df - df.min()) / (df.max() - df.min()) - 1


skip = 20
q1 = np.array(df_scaled['q1,'])[::skip]
q2 = np.array(df_scaled['q2,'])[::skip]
rt = np.array(df_scaled['concentration'])[::skip]


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

U_truth = np.zeros((len(t), n_controlInputs))
U_truth[:, 0:n_controlInputs] = np.array([q1,q2]).T

# pdb.set_trace()
#------------ NETWORK PARAMETERS -------------
hidden_neurons  = 10

# ---------NETWORK INITIALIZATION----------------------------------------
onlineFNN = OnlineFNN \
(
    time_steps=time_steps, 
    for_period=for_periods, 
    dt=dt, duration=duration, 
    horizon=time_horizon, 
    n_controlInputs=n_controlInputs, 
    n_controlOutputs=n_controlOutputs
)
onlineFNN.initializeModel(hidden_neurons=hidden_neurons)

N = len(onlineFNN.t)-for_periods-time_horizon
onlineFNN.createBuffer(buffer_size=buffer_size)

#----- STORING TRAINING LOSSES --------------- 

training_losses = np.zeros(len(rt))

#-----------TRYING TO PLOT SHIT ------------------------
n_epochs = 5


# plt.ion() 
fig = plt.figure(figsize=(8,8))

ax1 = fig.add_subplot(1, 1, 1) 


ax1.legend()
plt.legend()


# N = len(t) - skip - time_horizon
N = q1.shape[0] - time_horizon*2
# N = 1000

imlist = []
timeStepSkipPlot = 1
periodFrame = dt*timeStepSkipPlot
fps = 1/periodFrame
fpsFactor = fps / 5000
frameSkip = m.ceil ( fpsFactor )


print(N)
# -------------
for i in range(N): 

    print('this is timestep:', i)



    if i == (onlineFNN.buffer_size + 1): 
        print('Initial training in progress')

        X_train, Y_train = onlineFNN.createTrainingBatch()
        onlineFNN.trainStep(X_train, Y_train, n_epochs=100 )

        print('initial training over')
        

    elif i > (onlineFNN.buffer_size + 1): 
        print('online update in progress')
    
        X_train, Y_train = onlineFNN.createTrainingBatch()
        onlineFNN.trainStep(X_train, Y_train, n_epochs=20 )
        
        
       

    else:
        onlineFNN.updateBuffer(u_truth=U_truth, y=rt, i=i)

        continue


    new_input = onlineFNN.initializeForecastArray()
    # print(new_input)

    for k in range(1, onlineFNN.forecast.shape[0]): 
        u_new = U_truth[i+k]
        new_input, next_pred = onlineFNN.predictAndForecast(u_new, new_input, k=k)

        
    onlineFNN.storeForecast()

    # # exporting ------------------------------
    with open("forCosimo_17.txt", 'a') as file: 
        file.write(np.array2string(onlineFNN.forecast) + '\n')



    # loss = onlineLSTM.forecastLoss(onlineLSTM.forecast, rt, i)

    # training_losses[i] = loss


    # ------------ plotting -----------------
    forecast = onlineFNN.forecast
    
    forecast_coords = np.array([[t[i+j], forecast[j]] for j in range(len(forecast))])
    forecast_t = forecast_coords[:, 0]
    forecast_y = forecast_coords[:, 1]




    ax1.cla() 
    ax1.plot()
    ax1.plot(t[i:i+time_horizon], rt[i:i+time_horizon], label='True Forecast')
    ax1.plot(forecast_t, forecast_y, label='Network Forecast')
    ax1.plot(t[:i], rt[:i], label='Actual System Behavior')
    ax1.set_title(' Online Rescusive LSTM Forecasting Performance: ' + str(i))
    # ax1.set_xlim(t[0], 10)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_xlim(5, 50)
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('System Output')
    ax1.legend()

    # ax2.cla() 
    # ax2.plot(t[:i], training_losses[:i])
    
    # # ax2.set_xlim(t[0], 10)
    # ax2.set_xlim(5, duration)
    # ax2.set_ylim(0, 1)
    # ax2.set_xlabel('Time [s]')
    # ax2.set_ylabel('Mean Squared Error')
    # ax2.set_title('Network Learning Loss')

    
    
    # plt.legend()


    buf = BytesIO()
    plt.savefig(buf, dpi=200, format=None)
    buf.seek(0)
    im = Image.open(buf)

    imlist.append(im)


    plt.pause(0.1)
    
    onlineFNN.updateBuffer(u_truth=U_truth, y=rt, i=i)

    if i >= (1500 ):
        imlist[0].save('forCosimo_17.gif', loop=0, save_all=True, append_images=imlist[frameSkip::frameSkip],
                        fps = fps/frameSkip, dpi = 20, duration = frameSkip*1000/fps) 
        break 