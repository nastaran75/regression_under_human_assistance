To run various algorithms:
python eval_triage.py -f filename -l lambda -s error_probability -o option 
To check plots:
python plot_triage_real.py -f filename -l lambda -s error_probability 

The list of file name and list of options in code/eval_triage.py contains the possible list of file names and options.
Lambda and error probability values are mentioned in paper


You can run following commands to regenerate the figures in the paper: (Figures will be saved in Real_Data_Results folder)

To produce Messidor plots:
python plot_triage_real.py -f messidor -l 1 -s 0.1

To produce stare11 plots:
python plot_triage_real.py -f stare11 -l 1 -s 0.1

To produce stare5 plots:
python plot_triage_real.py -f stare5 -l 0.5 -s 0.1

To produce Hatespeech plots:
python plot_triage_real.py -f hatespeech -l 0.01 -s 0

To run various algorithms:

To run algorithm for Messidor dataset:
python eval_triage.py -f messidor -l 1 -s 0.1


To run algorithm for stare11 dataset:
python eval_triage.py -f stare11 -l 1 -s 0.1


To run algorithm for Stare-D dataset:
python eval_triage.py -f stare5 -l 0.5 -s 0.1


To run algorithm for Hatespeech dataset:
python eval_triage.py -f hatespeech -l 0.01 -s 0

