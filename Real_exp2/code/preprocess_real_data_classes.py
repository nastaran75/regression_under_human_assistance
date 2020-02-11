import sys
from os import listdir
from os.path import isfile, join
import numpy.random as rand
import math
import codecs
import csv 
import random
#import fasttext
from myutil import *
import numpy as np
import numpy.linalg as LA
from scipy.io import arff
import shutil 
from PIL import Image
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
#from real_data_classes import *

def check_sc(data_file):
    
    data=load_data( data_file) 
    # n,dim = data['X'].shape

    # for feature in range( dim ):
    #     data['X'][:,feature] = np.true_divide( data['X'][:,feature], LA.norm( data['X'][:,feature].flatten() ) )
    # save( data, data_file)
    
    d_mat = data['dist_mat']
    save( d_mat, data_file  + '_dist_mat')
    del data['dist_mat']
    save( data, data_file )
    return 


    def map_y(arr):
        return np.array([ x*(float(2)/3) + float(1/3) for x in arr ])
    
    
    # return 

    # data=load_data( data_file )    
    # data['Y'] = map_y( data['Y'])
    # data['test']['Y'] = map_y( data['test']['Y'])
    # save( data, data_file)
    # print np.unique(data['Y'])
    # print np.average(data['c']['0.0'].flatten())
    # x_norm = LA.norm(data['X'],axis=1)
    # a = np.array( [ LA.norm(row)**2 for row in data['X']   ])
    # print np.max( a)
    return 
    # data= load_data( data_file)
    # print data['0.5']['low'].shape
    # plt.plot(data['0.1']['low'])
    # plt.show()
    # return 
    
    # print data['c']['0.001'].shape
    # return
    x_norm = LA.norm(data['X'],axis=1)
    plt.plot( x_norm )
    plt.show()

def check_gaussian():

    n=100
    m=10
    std = float( sys.argv[1])
    p=float( sys.argv[2])
    x = rand.normal(0,std,100)
    plt.plot( x**2 , label='continuous')

    c=[]
    for sample in range(n):
        sum = 0 
        for i in range(m):
            x = np.random.uniform(0,1)
            if x < p:
                sum += 0.25
        c.append(float(sum)/m)
    plt.plot( c, label = 'discrete')
    plt.legend()
    plt.grid()
    # plt.ylim([0,.5])
    plt.show()

def plot_range_of_lambda( data_file):
    
    lamb = float( sys.argv[1])
    # def lower_bound_lambda( c,y,x_m):
    #     l_gamma = float(c)/(y**2)
    #     print l_gamma
    #     return l_gamma*x_m / (1-l_gamma)

    data= load_data( data_file )
    gamma_lower_bound = np.array( [ data['c']['0.5'][i]/float( data['Y'][i]**2 ) for i in range( data['X'].shape[0] ) ] )
    gamma_upper_bound = lamb /( lamb + np.max( LA.norm( data['X'], axis = 1 ).flatten()  )**2 ) 

    plt.plot(  gamma_lower_bound, label = 'gamma lower bound')
    plt.plot( gamma_upper_bound* np.ones( data['X'].shape[0] ) , label = 'gamma upper bound')
    print np.max( LA.norm( data['X'], axis = 1 ).flatten()  )**2

    plt.legend()
    plt.show()

class Generate_human_error:
    
    def __init__(self, data_file):
        # print data_file
        self.data = load_data( data_file )
        if 'c' in self.data:
            del self.data['c']
            del self.data['test']['c']
        self.n, self.dim  = self.data['X'].shape
        # self.normalize_features()
        
        # sc = StandardScaler() 
        # self.data['X'] = sc.fit_transform(self.data['X']) 
        # self.data['test']['X'] = sc.transform( self.data['test']['X']) 

    def normalize_features(self, delta = 1 ):
        
        n,dim = self.data['X'].shape
        for feature in range( dim ):
            self.data['X'][:,feature] = np.true_divide( self.data['X'][:,feature], 100*LA.norm( self.data['X'][:,feature].flatten() ) )
            self.data['test']['X'][:,feature] = np.true_divide( self.data['test']['X'][:,feature], 100*LA.norm( self.data['test']['X'][:,feature].flatten() ) )
        
        print np.max( [ LA.norm(x.flatten()) for x in self.data['X']] )
        # self.data['Y']=np.array([ y if y > 0 else delta for y in self.data['Y']])
        # self.data['test']['Y']=np.array([ y if y > 0 else delta for y in self.data['test']['Y']])

    def white_Gauss(self, std=1, n=1 , upper_bound = False, y_vec = None ):
        init_noise = rand.normal(0,std,n)
        if upper_bound :
            return np.array( [ noise if noise/y < 0.3 else 0.1*y for noise,y in zip(init_noise, y_vec) ])
        else:
    		return init_noise

    def data_independent_noise( self, list_of_std, upper_bound = False ):
        self.data['c'] = {} 
        self.data['test']['c']={}
        for std in list_of_std:
            self.data['c'][str(std)] = self.white_Gauss( std, self.data['Y'].shape[0], upper_bound , self.data['Y'] ) ** 2 
            self.data['test']['c'][str(std)] = self.white_Gauss( std, self.data['test']['Y'].shape[0], upper_bound, self.data['test']['Y']) ** 2 
            
    def variable_std_Gauss( self, std_const ,x ):
        n = x.shape[0]
        x_norm = LA.norm( x, axis=1 ).flatten()
        std_vector = std_const * np.reciprocal( x_norm )
        # print 'rnd shape ', rand.normal( 0, 2 , 1 ).shape
        tmp = np.array( [ rand.normal( 0, s ,1)[0] for s in std_vector  ])
        # print 'tmp.shape', tmp.shape
        return tmp
        
    def data_dependent_noise( self, list_of_std ):
        self.data['c'] = {} 
        self.data['test']['c']={}
        for std in list_of_std:
            self.data['c'][str(std)] = self.variable_std_Gauss( std, self.data['X']) ** 2 
            self.data['test']['c'][str(std)] = self.variable_std_Gauss( std, self.data['test']['X']) ** 2 

    def modify_y_values( self ):
        def get_num_category( y, y_t):
            y = np.concatenate(( y.flatten(), y_t.flatten() ), axis = 0 )
            return np.unique( y ).shape[0]    

        def map_range(v, l, h, l_new, h_new):

            # print '****'
            # print v
            # tmp = float(v-l)*(( h_new - l_new)/float( h-l))+ l_new
            # print tmp
            # return tmp
    		return float(v-l)*(( h_new - l_new)/float( h-l))+ l_new

        num_cat = get_num_category( self.data['Y'], self.data['test']['Y'])
        print num_cat
        self.data['Y'] = np.array( [ map_range(i, 0, 1, float(1)/num_cat, 1 ) for i in self.data['Y']]).flatten()
        self.data['test']['Y'] = np.array( [ map_range(i, 0, 1, float(1)/num_cat, 1 ) for i in self.data['test']['Y']]).flatten()
        
    def get_discrete_noise( self, p , num_cat):
        m=10
        c=[]
        for sample in range( self.n ):
            sum = 0 
            for i in range(m):
                x = np.random.uniform(0,1)
                if x < p:
                    sum += (float(1)/num_cat)**2
            c.append(float(sum)/m)
        return np.array(c)
     
    def discrete_noise( self, list_of_p ):
        
        def get_num_category( y, y_t):
            y = np.concatenate(( y.flatten(), y_t.flatten() ), axis = 0 )
            return np.unique( y ).shape[0]

        num_cat = get_num_category( self.data['Y'], self.data['test']['Y'] )
        if 'c' not in self.data:
            self.data['c'] = {} 
        if 'c' not in self.data['test']:
            self.data['test']['c']={} 
        for p in list_of_p:
            self.data['c'][str(p)] = self.get_discrete_noise( p, num_cat ) 
            self.data['test']['c'][str(p)] = self.get_discrete_noise( p, num_cat ) 

    def vary_discrete( self,  list_of_frac):
        def get_num_category( y, y_t):
            y = np.concatenate(( y.flatten(), y_t.flatten() ), axis = 0 )
            return np.unique( y ).shape[0] 

        def nearest( i ):
            return np.argmin( self.data['dist_mat'][i])   

        self.normalize_features()
        num_cat = get_num_category( self.data['Y'], self.data['test']['Y'])
        n=self.data['X'].shape[0]
        indices = np.arange( n )
        random.shuffle(indices)
        # err =  (( float(1)/num_cat )**2 )/20 
        #print self.data['Y']
        self.data['low']={}
        self.data['c']={}
        self.data['test']['c']={}
        for frac in list_of_frac:
            num_low = int(frac*n)
            self.data['low'][str(frac)]=indices[:num_low]
            self.data['c'][str(frac)] = np.array( [ 0.0001 if i in self.data['low'][str(frac)] else 0.1 for i in range(n) ] )
            self.data['test']['c'][str(frac)] = np.array( [ 0.0001 if nearest(i) in self.data['low'][str(frac)] else 0.15 for i in range( self.data['test']['X'].shape[0]) ] )
        
    def save_data(self, data_file):
        save( self.data , data_file)

def generate_human_error( path, file_name_list):
    option ='vary_discrete'
    list_of_std = [0.2, 0.4, 0.6, 0.8]
    for file_name in file_name_list:
        data_file = path + 'data/' + file_name + '_pca50_mapped_y_discrete'
        obj = Generate_human_error( data_file )
        obj.vary_discrete( list_of_std )
        obj.save_data( path + 'data/' + file_name + '_pca50_mapped_y_'+ option )

def compute_dist_dict( data_file ):
    data = load_data( data_file)
    num_test = data['test']['X'].shape[0]
    num_train = data['X'].shape[0]
    data['dist_mat']=np.zeros((num_test,num_train))
    for te,i in zip(data['test']['X'], range(num_test)):
        for tr,j in zip(data['X'], range(num_train)):
            data['dist_mat'][i,j]=LA.norm( te-tr)
    save( data, data_file )
    return 

    save( data['dist_mat'] , data_file + '_dist_mat')

    dist_dict = {}
    for i, dist_arr in zip(range( num_test), data['dist_mat']):
        dist_dict[str(i)] = np.argmin(dist_arr)
    data['dist_dict'] = dist_dict
    del data['dist_mat']
    save( data, data_file )

def main():
    path = '../Real_Data_Results/'
    file_name = 'stare5'
    generate_human_error( path , [file_name])
    compute_dist_dict( path + 'data/' + file_name + '_pca50_mapped_y_vary_discrete')
    return 
    
if __name__=="__main__":
	main()
