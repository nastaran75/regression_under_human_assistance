Figure 1 of the paper is generated and saved in Synthetic_fig3

Figure 2 of the paper is generated and saved in Synthetic_data


Inside each code option,  dimension and sigma values must be specified according to paper. 
file_name = ['sigmoid','Gaussian']
python generate_data.py  : generate the synthetic data.
python eval_triage.py : run various algorithms.
python plot_triage_real.py : generate plots.


To run algorithms for Gaussian:
python eval_triage_real.py -f Gauss -s .001 -l .005

To generate plots for sigmoid:
python eval_triage_real.py -f sigmoid -s .001 -l .001

To generate plots for Gaussian:
python plot_triage_real.py -f Gauss -s .001 -l .005

To generate plots for sigmoid:
python plot_triage_real.py -f sigmoid -s .001 -l .001
