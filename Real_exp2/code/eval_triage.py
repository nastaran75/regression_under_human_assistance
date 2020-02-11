import time
import os
import sys 
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
from generate_data import generate_data 
from triage_human_machine import triage_human_machine


def print_new_pca_data( data_file_old, data_file_new,n):
	data=load_data( data_file_old )
	data['X']=data['X'][:,:n]
	data['test']['X']=data['test']['X'][:,:n]
	save( data, data_file_new)

def relocate_data( data_file_old, data_file_new ):
	data=load_data( data_file_old)
	save( data, data_file_new )

class eval_triage:
	def __init__(self,data_file,real_flag=None, real_wt_std=None):
		self.data=load_data(data_file)
		self.real=real_flag		
		self.real_wt_std=real_wt_std


	def eval_loop(self,param,res_file,option):	
		res=load_data(res_file,'ifexists')		
		for std in param['std']:
			data_dict = {'X':self.data['X'],'Y':self.data['Y'],'c': self.data['c'][str(std)]}
			triage_obj=triage_human_machine(data_dict,self.real_wt_std)
			if str(std) not in res:
				res[str(std)]={}
			for K in param['K']:
				if str(K) not in res[str(std)]:
					res[str(std)][str(K)]={}
				for lamb in param['lamb']:
					if str(lamb) not in res[str(std)][str(K)]:
						res[str(std)][str(K)][str(lamb)]={}
					# res[str(std)][str(K)][str(lamb)]['greedy'] = triage_obj.algorithmic_triage({'K':K,'lamb':lamb},optim='greedy')
					print 'std-->', std, 'K--> ',K,' Lamb--> ',lamb
					res_dict = triage_obj.algorithmic_triage({'K':K,'lamb':lamb},optim=option)
					res[str(std)][str(K)][str(lamb)][option] = res_dict
		save(res,res_file)


def main():
	#---------Real Data-------------------------------------------
	setting = 'mapped_y_vary_discrete'
	list_of_std=[0.2,.4,.6,.8]
	list_of_lamb=[0.1]
	list_of_option =['greedy']
	file_name_list = ['stare5']#,'stare11']
	path = '../Real_Data_Results/'
	for file_name in file_name_list:
		print('-'*50+'\n'+file_name+'\n\n'+'-'*50)
		data_file = path + 'data/'+file_name+'_pca50_'+setting
		res_file= path + file_name + '_res_pca50_'+setting
		for option in list_of_option:
			list_of_K = [0.99]	
			param={'std':list_of_std,'K':list_of_K,'lamb':list_of_lamb}
			obj=eval_triage(data_file,real_wt_std=True)
			obj.eval_loop(param,res_file,option)	
	
if __name__=="__main__":
	main()

