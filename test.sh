#python test_minst.py -a 'softplus' -u 20 -b 1 -s True -r True  -g adam -l 0.000125 -p 0.00001 -n 250  -i 5000 -w ./soft_sym_reg_weights.txt -m ./model_softplus_20_batch250_class0_sym_reg.ckpt

#python test_minst.py -a 'softplus' -u 20 -b 1 -s False -r True -g adam -l 0.000125 -p 0.00001 -n 250  -i 5000 -w ./soft_nosym_reg_weights.txt -m ./model_softplus_20_batch250_class0_nosym_reg.ckpt

python test_minst.py -a 'softplus' -u 20 -b 1 -s False -r False -g adam -l 0.000125 -p 0.00001 -n 250  -i 5000 -w ./soft_nosym_noreg_weights.txt -m ./model_softplus_20_batch250_class0_nosym_noreg.ckpt

python test_minst.py -a 'sigmoid' -u 20 -b 1 -s True -r True -g adam -l 0.000125 -p 0.001 -n 250  -i 5000 -w ./sig_sym_reg_weights.txt -m ./model_sigmoid_20_batch250_class0_sym_reg.ckpt

python test_minst.py -a 'sigmoid' -u 20 -b 1 -s False -r True -g adam -l 0.000125 -p 0.001 -n 250 -i 5000 -w ./sig_nosym_reg_weights.txt -m ./model_sigmoid_20_batch250_class0_nosym_reg.ckpt

python test_minst.py -a 'sigmoid' -u 20 -b 1 -s False -r False -g adam -l 0.000125 -p 0.001 -n 250 -i 5000 -w ./sig_nosym_noreg_weights.txt -m ./model_sigmoid_20_batch250_class0_nosym_noreg.ckpt


