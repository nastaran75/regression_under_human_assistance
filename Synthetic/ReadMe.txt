python generate_data.py  : generate the synthetic data.
python eval_triage.py : run various algorithms.
python plot_triage_real.py : generate plots.


To generate plots for Gaussian: (plots will be saved in Synthetic_data)
python plot_triage_real.py -f Gauss -s .001 -l .005

To generate plots for sigmoid: (plots will be saved in Synthetic_data)
python plot_triage_real.py -f sigmoid -s .001 -l .001


To run algorithms for Gaussian:
python eval_triage_real.py -f Gauss -s .001 -l .005

To run algorithm for sigmoid:
python eval_triage_real.py -f sigmoid -s .001 -l .001

