python test_minst.py -a 'softplus' -u 10 -b 1 -g adam -l 0.000125 -p 0.000000125 -n 1000  -i 5000 -m ./model_softplus_10_batch1000_class0.ckpt

python test_minst.py -a 'sigmoid' -u 10 -b 1 -g adam -l 0.000125 -p 0.000125 -n 1000  -i 5000 -m ./model_sigmoid_10_batch1000_class0.ckpt

python test_minst.py -a 'softplus' -u 80 -b 1 -g adam -l 0.000125 -p 0.000000125 -n 1000  -i 5000 -m ./model_softplus_80_batch1000_class0.ckpt

python test_minst.py -a 'sigmoid' -u 80 -b 1 -g adam -l 0.000125 -p 0.000125-n 1000 -i 5000 -m ./model_sigmoid_80_batch1000_class0.ckpt




