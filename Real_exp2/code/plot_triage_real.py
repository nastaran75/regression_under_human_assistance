import time
import sys
import os
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
from generate_data import generate_data

class plot_triage_real:
	def __init__(self,list_of_K,list_of_std, list_of_lamb,list_of_option,list_of_test_option, flag_synthetic=None):
		self.list_of_K=list_of_K
		self.list_of_std=list_of_std
		self.list_of_lamb=list_of_lamb
		self.list_of_option=list_of_option
		self.list_of_test_option = list_of_test_option
		self.flag_synthetic = flag_synthetic

	def plot_exp2_real(self,res_file,image_path):
		def smooth( vec ):
			vec = np.array(vec)
			tmp = np.array( [ (.25*vec[ind]+.5*vec[ind-1]+.25*vec[ind+1]) for ind  in range(1,vec.shape[0]-1)  ])
			vec[0] = vec[0]*.75+vec[1]*.25
			vec[-1] = vec[-1]*.75+vec[-2]*.25
			vec[1:-1] = tmp
			return vec
				
		option = self.list_of_option[0]
		res=load_data(res_file)
		for lamb in self.list_of_lamb:
			suffix='_lamb_'+str(lamb)
			image_file=image_path +suffix.replace('.','_')		
			for std in self.list_of_std:		
				for test_method in self.list_of_test_option:
					err_K_te=[]	
					for K in self.list_of_K:
						if option in res[str(std)][str(K)][str(lamb)]:
							err_K_te.append(res[str(std)][str(K)][str(lamb)][option]['test_res'][test_method]['error'])
					plt.plot(smooth(err_K_te), label = str(std) )
			plt.legend()
			plt.savefig(image_file)
			# plt.show()

	def get_nearest_human(self,dist,tr_human_ind):
		
		# start= time.time()
		n_tr=dist.shape[0]
		human_dist=float('inf')
		machine_dist=float('inf')
		for d,tr_ind in zip(dist,range(n_tr)):
			if tr_ind in tr_human_ind:
				if d < human_dist:
					human_dist=d
			else:
				if d < machine_dist:
					machine_dist=d
		# print 'Time required -----> ', time.time() - start , ' seconds'
		return (human_dist -machine_dist)

	def get_test_error(self,res_obj,dist_mat,x,y,y_h=None,c=None,K=None):
		w=res_obj['w']
		subset=res_obj['subset']
		n,tr_n=dist_mat.shape
		no_human=int((subset.shape[0]*n)/float(tr_n))

		y_m=x.dot(w)
		err_m=(y-y_m)**2
		if y_h==None:
			err_h=c  
		else: 
			err_h=(y-y_h)**2

		# start = time.time()
		diff_arr=[ self.get_nearest_human(dist,subset) for dist in dist_mat]
		# print 'Time required -----> ', time.time() - start , ' seconds'

		indices=np.argsort(np.array(diff_arr))
		subset_te_r = indices[:no_human]
		subset_machine_r=indices[no_human:]

		if subset_te_r.size==0:
			error_r =  err_m.sum()/float(n)
		else:
			error_r = ( err_h[subset_te_r].sum() + err_m.sum() - err_m[subset_te_r].sum() ) /float(n)


		subset_te_n = np.array([int(i)  for i in range(len(diff_arr)) if diff_arr[i] < 0 ])
		# print 'subset size test', subset_te_n.shape
		subset_machine_n = np.array([int(i)  for i in range(len(diff_arr)) if i not in subset_te_n ])
		# print 'sample to human--> ' , str(subset_te_n.shape[0]), ', sample to machine--> ', str( subset_machine_n.shape[0])

		if subset_te_n.size==0:
			error_n =  err_m.sum()/float(n)
		else:
			error_n = ( err_h[subset_te_n].sum() + err_m.sum() - err_m[subset_te_n].sum() ) /float(n)

		# return {'error':error, 'human_ind':subset_te, 'machine_ind':subset_machine}
		error_n={'error':error_n, 'human_ind':subset_te_n, 'machine_ind':subset_machine_n}
		error_r={'error':error_r, 'human_ind':subset_te_r, 'machine_ind':subset_machine_r}
		return error_n, error_r

					
	def get_train_error(self,plt_obj,x,y,y_h=None,c=None):
		subset = plt_obj['subset']
		w=plt_obj['w']
		n=y.shape[0]
		if y_h==None:
			err_h=c
		else:
			err_h=(y_h-y)**2

		y_m= x.dot(w)
		err_m=(y_m-y)**2
		error = ( err_h[subset].sum()+err_m.sum() - err_m[subset].sum() ) /float(n)
		return {'error':error}

	def compute_result(self,res_file,data_file,option, image_file_prefix =None):
		data=load_data(data_file)
		res=load_data(res_file)
		for std,i0 in zip(self.list_of_std,range( len(self.list_of_std) )):
			for K,i1 in zip(self.list_of_K,range(len(self.list_of_K))):
				for lamb,i2 in zip(self.list_of_lamb,range(len(self.list_of_lamb))):
					if option in res[str(std)][str(K)][str(lamb)]:
						res_obj=res[str(std)][str(K)][str(lamb)][option]
						train_res = self.get_train_error(res_obj,data['X'],data['Y'],y_h=None,c=data['c'][str(std)])
						test_res_n,test_res_r = self.get_test_error(res_obj,data['dist_mat'],data['test']['X'],data['test']['Y'],y_h=None,c=data['test']['c'][str(std)],K=K)
						res[str(std)][str(K)][str(lamb)][option]['test_res']={'ranking':test_res_r,'nearest':test_res_n}
						res[str(std)][str(K)][str(lamb)][option]['train_res']=train_res
					
		save(res,res_file)
			
	def merge_results(self,input_res_files,merged_res_file):

		res={}
		for std in self.list_of_std:
			if str(std) not in res:
				res[str(std)]={}
			for K in self.list_of_K:
				if str(K) not in res[str(std)]:
					res[str(std)][str(K)]={}
				for lamb in self.list_of_lamb:
					if str(lamb) not in res[str(std)][str(K)]:
						res[str(std)][str(K)][str(lamb)]={}
					r=load_data(input_res_files[str(lamb)])
					# print r['0.0'].keys()
					# print res['0.0'].keys()
					res[str(std)][str(K)][str(lamb)] = r[str(std)][str(K)][str(lamb)]
		save(res,merged_res_file)

	def split_res_over_K(self,data_file,res_file,unified_K,option):
		res=load_data(res_file)
		for std in self.list_of_std:
			if str(std) not in res:
				res[str(std)]={}
			for K in self.list_of_K:
				if str(K) not in res[str(std)]:
					res[str(std)][str(K)]={}
				for lamb in self.list_of_lamb:
					if str(lamb) not in res[str(std)][str(K)]:
						res[str(std)][str(K)][str(lamb)]={}
					
					if option not in res[str(std)][str(K)][str(lamb)]:
						res[str(std)][str(K)][str(lamb)][option]={}
					if K != unified_K:
						res_dict = res[str(std)][str(unified_K)][str(lamb)][option]
						if res_dict:
							res[str(std)][str(K)][str(lamb)][option] = self.get_res_for_subset(data_file,res_dict,lamb,K)
		save(res,res_file)

	def get_optimal_pred(self,data,subset,lamb):
		
		n,dim= data['X'].shape
		subset_c=  np.array([int(i) for i in range(n) if i not in subset])	
		X_sub=data['X'][subset_c].T
		Y_sub=data['Y'][subset_c]
		subset_c_l=n-subset.shape[0]
		return LA.inv( lamb*subset_c_l*np.eye(dim) + X_sub.dot(X_sub.T) ).dot(X_sub.dot(Y_sub))

	def get_res_for_subset(self,data_file,res_dict,lamb,K):
		data=load_data(data_file)
		curr_n = int( data['X'].shape[0] * K )
		subset_tr = res_dict['subset'][:curr_n]
		w= self.get_optimal_pred(data,subset_tr,lamb)
		return {'w':w,'subset':subset_tr}

def main():
	#---------Real Data-------------------------------------------
	setting = 'mapped_y_vary_discrete'
	list_of_std=[0.2,.4,.6,.8]
	list_of_lamb=[0.1]
	list_of_K = [  0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 ] 
	list_of_option =['greedy']
	list_of_test_option = ['nearest']
	file_name_list = ['stare5']#,'stare11']
	path = '../Real_Data_Results/'
	obj=plot_triage_real(list_of_K, list_of_std, list_of_lamb, list_of_option, list_of_test_option)
	for file_name in file_name_list:
		data_file = path + 'data/' + file_name + '_pca50_' + setting
		res_file= path + file_name + '_res_pca50_' + setting
		for option in list_of_option:
			unified_K = 0.99
			obj.split_res_over_K(data_file,res_file,unified_K,option)
			obj.compute_result(res_file,data_file,option, 'dummy')
		image_path = path + 'Fig2_'+ file_name 
		obj.plot_exp2_real(res_file,image_path)	

if __name__=="__main__":
	main()
