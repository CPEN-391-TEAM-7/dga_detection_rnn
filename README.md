## Update Feb 10

simple RNN model trained at exported at /model_export with 98% accuracy

### usage

python3 test_rnn_binary.py -d somedomain

Will output a score between 0 and 1, 0 for normal domain and 1 for dga

### example
python3 test_rnn_binary.py -d google.com

### Dependency
tensorflow >= 2.3.0


## Data Collection

Data for dga: https://data.netlab.360.com/dga/

Data for white list domain: https://majestic.com/reports/majestic-million
