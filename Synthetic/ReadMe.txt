Inside each code option,  dimension and sigma values must be specified according to paper. 
file_name = ['sigmoid','gauss']
python generate_data.py  : generate the synthetic data.
python eval_triage.py : run various algorithms.
python plot_triage_real.py : generate plots.


To run algorithms for Gaussian:
python eval_triage_real.py -f gauss -s .001 -l .005

To generate plots for sigmoid:
python eval_triage_real.py -f sigmoid -s .001 -l .001

To generate plots for Gaussian:
python plot_triage_real.py -f gauss -s .001 -l .005

To generate plots for sigmoid:
python plot_triage_real.py -f sigmoid -s .001 -l .001


To generate Fig3 of the paper for sigmoid:
python generate_data.py vary_sigmoid
python Synthetic_exp3.py vary_sigmoid

To generate Fig3 of the paper for gaussian:
python generate_data.py vary_gauss
python Synthetic_exp3.py vary_gauss


