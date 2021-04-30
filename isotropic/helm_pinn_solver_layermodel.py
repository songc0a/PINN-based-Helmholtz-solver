"""
@author: Chao Song
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
import cmath

np.random.seed(1234)
tf.set_random_seed(1234)

fre = 5.0
PI = 3.1415926
omega = 2.0*PI*fre
niter = 50000
nz = 101
nx = 101
misfit = []
misfit1 = []
class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, x, y, u0_real, u0_imag, m, m0, layers, omega):
        
        X = np.concatenate([x, y], 1)
        
        self.lb = X.min(0)
        self.ub = X.max(0)
                
        self.X = X
        
        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.u0_real = u0_real
        self.u0_imag = u0_imag
        self.m = m
        self.m0 = m0

        self.layers = layers

        self.omega = omega
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)  

        # tf placeholders 
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        
        self.du_real_pred, self.du_imag_pred, self.f_real_pred, self.f_imag_pred = self.net_NS(self.x_tf, self.y_tf)

        # loss function we define  
        self.loss = tf.reduce_sum(tf.square(self.f_real_pred)) + tf.reduce_sum(tf.square(self.f_imag_pred))
        
        # optimizer used by default (in original paper)        
            
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1 * np.finfo(float).eps}) 

        self.optimizer_Adam = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32)+0.0, dtype=tf.float32)
            #b = tf.Variable(tf.random_uniform([1,layers[l+1]], 0.0,0.5,dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.atan(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_NS(self, x, y):
    
        # output scattered wavefield: du_real du_imag, loss function: L du+omega^2 dm u0 

        omega = self.omega
        m = self.m
        m0 = self.m0
        u0_real = self.u0_real
        u0_imag = self.u0_imag

        dureal_and_duimag = self.neural_net(tf.concat([x,y], 1), self.weights, self.biases)
        du_real = dureal_and_duimag[:,0:1]
        du_imag = dureal_and_duimag[:,1:2]

        du_real_x = tf.gradients(du_real, x)[0]
        du_real_y = tf.gradients(du_real, y)[0]
        du_real_xx = tf.gradients(du_real_x, x)[0]
        du_real_yy = tf.gradients(du_real_y, y)[0]

        du_imag_x = tf.gradients(du_imag, x)[0]
        du_imag_y = tf.gradients(du_imag, y)[0]
        du_imag_xx = tf.gradients(du_imag_x, x)[0]
        du_imag_yy = tf.gradients(du_imag_y, y)[0]

        f_real =  omega*omega*m*du_real + du_real_xx + du_real_yy + omega*omega*(m-m0)*u0_real #  L du + omega^2 dm u0 
        f_imag =  omega*omega*m*du_imag + du_imag_xx + du_imag_yy + omega*omega*(m-m0)*u0_imag #  L du + omega^2 dm u0
 
        return du_real, du_imag, f_real, f_imag 

    
    def callback(self, loss):
        print('Loss: %.3e' % (loss))
        misfit1.append(loss) 
      
    def train(self, nIter): 

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            loss_value = self.sess.run(self.loss, tf_dict)
            misfit.append(loss_value)         
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                #misfit.append(loss_value)
                print('It: %d, Loss: %.3e,Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()

            
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)

            
    
    def predict(self, x_star, z_star):
        
        tf_dict = {self.x_tf: x_star, self.y_tf: z_star}
        
        du_real_star = self.sess.run(self.du_real_pred, tf_dict)
        du_imag_star = self.sess.run(self.du_imag_pred, tf_dict)

        return du_real_star, du_imag_star
        
       
if __name__ == "__main__": 
      
    N_train = 2000
    
   # layers = [2, 40, 40, 40, 40, 40, 40, 40, 40, 2]
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 2] # neurons in each layer (used in original paper)
    # Load Data
    data = scipy.io.loadmat('layer_5Hz_test_data.mat')
    du_real_star = data['dU_real_star'] # N=10000 G_imge
    du_imag_star = data['dU_imag_star'] # N=10000 G_imge 

    x_star = data['x_star'] # N=10000 

    z_star = data['z_star'] # N=10000 


    train_data = scipy.io.loadmat('layer_5Hz_train_data.mat')
           
    u0_real_train = train_data['U0_real_train'] # N=10000 G_real 
    u0_imag_train = train_data['U0_imag_train'] # N=10000 G_real 

    x_train = train_data['x_train'] # N=10000 

    z_train = train_data['z_train'] # N=10000

    m_train = train_data['m_train'] # N=10000 

    m0_train = train_data['m0_train'] # N=10000


    # Training
    model = PhysicsInformedNN(x_train, z_train, u0_real_train, u0_imag_train, m_train, m0_train, layers, omega)
    model.train(niter)

    scipy.io.savemat('misfit_adam.mat',{'misfit':misfit})
    scipy.io.savemat('misfit_lbfgs.mat',{'misfit1':misfit1})     

    # Prediction
    du_real_pred, du_imag_pred = model.predict(x_star, z_star)
    
    # Error
    error_du_real = np.linalg.norm(du_real_star-du_real_pred,2)/np.linalg.norm(du_real_star,2)
    error_du_imag = np.linalg.norm(du_imag_star-du_imag_pred,2)/np.linalg.norm(du_imag_star,2)

    print('Error u_real: %e, Error u_imag: %e' % (error_du_real,error_du_imag))    

    scipy.io.savemat('du_real_pred.mat',{'du_real_pred':du_real_pred})
    scipy.io.savemat('du_imag_pred.mat',{'du_imag_pred':du_imag_pred})

    scipy.io.savemat('du_real_star.mat',{'du_real_star':du_real_star})
    scipy.io.savemat('du_imag_star.mat',{'du_imag_star':du_imag_star})


        

             
  
