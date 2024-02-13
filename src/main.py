import numpy as np
import math
import random
import statistics as stat
import scipy.stats
import scipy.special
import array
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# C is the number of servers
def Heavy_Traffic(arr_rate, ser_rate,arr_var, ser_var,C):
    rho = arr_rate/ser_rate
    eta = 1/2
    c = 1
    z = 1 + (c**2 -1)*eta
    Block = math.sqrt(z/rho)*scipy.stats.norm(0, 1).pdf((C-rho)/math.sqrt(rho*z))/scipy.stats.norm(0, 1).cdf((C-rho)/math.sqrt(rho*z))
    return Block

def Hayward(arr_rate, ser_rate, arr_var, ser_var, C):
    rho = arr_rate/ser_rate
    eta = 1/2
    c = 1
    z = 1 + (c**2 -1)*eta
    f1 = scipy.special.gammainc((C+1)/z, rho/z)
    f2 = scipy.special.gamma((C+1)/z)
    f3 = scipy.special.gammainc(C/z, rho/z)
    f4 = scipy.special.gamma(C/z)
    Block = (f1/f2 - f3/f4)*f2/f1

def Erlang_B(arr_rate,ser_rate,C):
    rho = arr_rate/ser_rate
    Blocking_Probability = 1
    for i in range(C+1):
        Blocking_Probability = rho*Blocking_Probability/(rho*Blocking_Probability + i)
    return Blocking_Probability

def collect_data(uni_a,uni_b,uni_c, uni_d,C,N): #C = number of servers, N = number of data points,
    Block = np.ones(N)
    Block2 = np.ones(N)
    Block3 = np.ones(N)
    uni_vec_1 = np.random.uniform(uni_a,uni_b,N)  
    uni_vec_2 = np.random.uniform(uni_c,uni_d,N)
    uni_vec_3 = np.random.randint(1,C,N)
    for i in range(N):
        Block[i] = Erlang_B(uni_vec_1[i],uni_vec_2[i],uni_vec_3[i])
        Block2[i] = Heavy_Traffic(uni_vec_1[i],uni_vec_2[i],uni_vec_1[i],uni_vec_2[i],uni_vec_3[i])
        Block3[i] = Hayward(uni_vec_1[i],uni_vec_2[i],uni_vec_1[i],uni_vec_2[i],uni_vec_3[i])
    return uni_vec_1,uni_vec_2,uni_vec_3,Block,Block2,Block3

def simulate_queue(N,K,arrival,service):
    """Simulate a queueing system with N customers and K servers.
    
    K is servers and threshold in terms of blocking. It is a blocking queue.
    Arrival time + service time if not blocked. Arrival time if blocked.
    From the # of departures we can get the queue length. Blocking means you
    don't get service at all.

    Want to calculate the fraction of people who get blocked via simulation.
    Ideally we want the blocking probability. Using the estimate as a data point.
    """
    queue_upon_arrival = np.zeros(N);
    blocked = np.zeros(N);
    blocked2 = np.zeros(N);
    # It is in terms of interarrival times. 
    arrival_times = np.cumsum(arrival);
    departure_times =  np.zeros(N);

# Computing the queue length upon arrival
    queue_upon_arrival[0] = 0;
    blocked[0] = 0;
    blocked2[0] = 0;
    new_index = 0
    num_departed = 0
    departure_times[0] = arrival_times[0] + service[0];
    for i in range(new_index,N-1):
        count = 0 ;
        #print(new_index)
        for j in range(1,i):
            num_departed = num_departed + (departure_times[j] < arrival_times[i+1])
            count = count + (departure_times[j] - arrival_times[i] > 0)
   
        queue_upon_arrival[i] = count
        new_index = int(np.max(num_departed - blocked2[i] - K,0))
        departure_times[i] = arrival_times[i] + service[i]*(count < K )
        blocked[i] = (count < K)
        blocked2[i] = blocked2[i-1] + (1-blocked[i])
    mean_blocked = 1-np.mean(blocked)
    mean_queue = np.mean(queue_upon_arrival)
    return mean_blocked,mean_queue

  # Generating the Data  
N=10
C=100
uni_vec_1,uni_vec_2,uni_vec_3,Block,Block2,Block3 = collect_data(1,100,1,10,C,N)
m = 5
X = np.column_stack((uni_vec_1, uni_vec_2))
#X = np.column_stack((X, rho))
test_array = Block[N-m:N]
#actual_array = mean_sys[N-m,N]
rho = np.divide(uni_vec_1, uni_vec_2)


l_linear_model = LinearRegression().fit(X[0:N-m], Block[0:N-m])
l_prediction_array = []
for i in range(m):
    l_prediction = l_linear_model.intercept_ + l_linear_model.coef_[0]*X[N-m+i][0] \
    + l_linear_model.coef_[1]*X[N-m+i][1]
    l_prediction_array.append(l_prediction)
#print('l actual: ', l_actual_array)
#print('l predict: ', l_prediction_array)
#print('intercept:', l_linear_model.intercept_)
#print('slope:', l_linear_model.coef_)
means_squared_error_l = math.pow(np.linalg.norm(test_array-l_prediction_array), 2)/m
print('means_squared_error for l: ', means_squared_error_l)
print(l_linear_model.intercept_, l_linear_model.coef_)
print('rho values: ', rho[N-m:N])
print('test data: ', test_array)
print('heavy_traffic: ', Block2[N-m:N])
print('lin reg predict: ', l_prediction_array)
#print(X[N-m:N])
#print('shape of Y', Block.shape)
#print(arrival_rate)
#print('shape of X', X.shape)
print(X)
Nstar = 4000
M=5
blocking = np.zeros(M)
queue_length = np.zeros(M)
for i in range(M):
    lam = 1/X[N-m+i][0]
    mu = 1/X[N-m+i][1]
    print(i)
    blocking[i],queue_length[i] = simulate_queue(Nstar,C,np.random.exponential(scale=lam, size=Nstar),np.random.exponential(scale=mu, size=Nstar))
print('simulated blocking probability:', blocking)

 # deep neural network
    # ReLU: 100 layer
n_relu100 = MLPRegressor(hidden_layer_sizes=100, activation='relu', max_iter=10000).fit(X[0:N-m], Block[0:N-m])
n_relu100_prediction = n_relu100.predict(X[N-m:N])


print('\nrelu100 predict: ', n_relu100_prediction)
DataSet = X  # Define DataSet variable based on X
S = N
littles_law = Block

AMSE_n_relu100 = math.pow(np.linalg.norm(test_array-n_relu100_prediction), 2) / m
AMSE_relu100 = []
AMSE_relu100.append(AMSE_n_relu100)
print('\nerror for relu100: ', AMSE_n_relu100)

# deep neural network
    # ReLU: 100 layer
n_relu100 = MLPRegressor(hidden_layer_sizes=100, activation='relu', max_iter=10000).fit(DataSet[0:S-m], littles_law[0:S-m])
n_relu100_prediction = n_relu100.predict(DataSet[S-m:S])
#print('\nrelu100 predict: ', n_relu100_prediction)
AMSE_n_relu100 = math.pow(np.linalg.norm(actual_array-n_relu100_prediction), 2) / m
AMSE_relu100 = []
AMSE_relu100.append(AMSE_n_relu100)
print('\nerror for relu100: ', AMSE_n_relu100)

# deep neural network
    # ReLU: 400 layer
n_relu400 = MLPRegressor(hidden_layer_sizes=400, activation='relu', max_iter=10000).fit(DataSet[0:S-m], littles_law[0:S-m])
n_relu400_prediction = n_relu400.predict(DataSet[S-m:S])
#print('\nrelu400 predict: ', n_relu400_prediction)
AMSE_n_relu400 = math.pow(np.linalg.norm(actual_array-n_relu400_prediction), 2) / m
AMSE_relu400 = []
AMSE_relu400.append(AMSE_n_relu400)
print('\nerror for relu400: ', AMSE_n_relu400)

 # deep neural network
    # TANH: 100 layer
n_tanh100 = MLPRegressor(hidden_layer_sizes=100, activation='tanh', max_iter=10000).fit(DataSet[0:S-m], littles_law[0:S-m])
n_tanh100_prediction = n_tanh100.predict(DataSet[S-m:S])
#print('\ntanh100 predict: ', n_tanh100_prediction)
AMSE_n_tanh100 = math.pow(np.linalg.norm(actual_array-n_tanh100_prediction), 2) / m
AMSE_tanh100 = []
AMSE_tanh100.append(AMSE_n_tanh100)
print('\nerror for tanh100: ', AMSE_n_tanh100)



 # deep neural network
    # SIG: 100 layer
n_sig100 = MLPRegressor(hidden_layer_sizes=100, activation='logistic', max_iter=10000).fit(DataSet[0:S-m], littles_law[0:S-m])
n_sig100_prediction = n_sig100.predict(DataSet[S-m:S])
#print('\nsig100 predict: ', n_sig100_prediction)
AMSE_n_sig100 = math.pow(np.linalg.norm(actual_array-n_sig100_prediction), 2) / m
AMSE_sig100 = []
AMSE_sig100.append(AMSE_n_sig100)
print('\nerror for sig100: ', AMSE_n_sig100)


#knn regression model k = 5
k_prediction_array_5 = []
for i in range(m):
    test_point = np.array(littles_law[S-m+i])
    k_nearest_neighbors, k_prediction_5 = knn(Z, test_point, k=5, distance_fn=euclidean_distance, choice_fn=mean)
    k_prediction_array_5.append(k_prediction_5)
#print('k actual: ', littles_law[S-m:S])
#print('k predict: ', k_prediction_array_5)
means_squared_error_k_5 = math.pow(np.linalg.norm(actual_array-k_prediction_array_5), 2)/m
print('means_squared_error for k=5: ', means_squared_error_k_5)

#knn regression model k = 10
k_prediction_array_10 = []
for i in range(m):
    test_point = np.array(littles_law[S-m+i])
    k_nearest_neighbors_10, k_prediction_10 = knn(Z, test_point, k=10, distance_fn=euclidean_distance, choice_fn=mean)
    k_prediction_array_10.append(k_prediction_10)