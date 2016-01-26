
python test_minst.py -u 200 -a 'sigmoid' -i 1000 -b 1 -n 500 -g 'adam' -l 0.000125 -r False -d 0.0 -s True -e None -k 0.15 -o True -m test200x15 -w weight_200x15

python test_minst.py -u 100 -a 'sigmoid' -i 1000 -b 1 -n 500 -g 'adam' -l 0.000125 -r False -d 0.0 -s True -e None -k 0.30 -o True -m test100x30 -w weight_100x30

python test_minst.py -u 60 -a 'sigmoid' -i 1000 -b 1 -n 500 -g 'adam' -l 0.000125 -r False -d 0.0 -s True -e None -k 0.5 -o True -m test60x50 -w weight_60x50

python test_minst.py -u 30 -a 'sigmoid' -i 1000 -b 1 -n 500 -g 'adam' -l 0.000125 -r False -d 0.0 -s True -e None -k 0.5 -o True -m test30x50 -w weight_30x50