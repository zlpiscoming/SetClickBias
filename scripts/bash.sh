#1458 2259 2261 2821 2997 3358 3386 3427 3476
#['3476', '2259', '2821', '3358', '0',    '3386', '3427', '1458', '2997']
#['0606', '1019', '1019', '0606', '0315', '0606', '0606', '0606', '1021']
#[ 246,    287,    291,    226,     15,    300,     234,    300,    277]

python exp.py --algo cpo --cus 3476 --address 0606 --bid 246 --withf False --otherwarm False --constraint_type raw
python exp.py --algo cpo --cus 3476 --address 0606 --bid 246 --withf True --otherwarm False --constraint_type dyn
python exp.py --algo ppo --cus 3476 --address 0606 --bid 246 --withf False --otherwarm False --constraint_type without
#NAME="customer = 2259 WF"
#echo ${NAME}
#python exp.py --algo cpo --cus 2259 --address 1019 --bid 287 --withf False --otherwarm False
#NAME="customer = 2259 F"
#echo ${NAME}
#python exp.py --algo cpo --cus 2259 --address 1019 --bid 287 --withf True --otherwarm False
#
#NAME="customer = 2821 WF"
#echo ${NAME}
#python exp.py --algo cpo --cus 2821 --address 1019 --bid 291 --withf False --otherwarm False
#NAME="customer = 2821 F"
#echo ${NAME}
#python exp.py --algo cpo --cus 2821 --address 1019 --bid 291 --withf True --otherwarm False
#
#NAME="customer = 3358 WF"
#echo ${NAME}
#python exp.py --algo cpo --cus 3358 --address 0606 --bid 226 --withf False --otherwarm False
#NAME="customer = 3358 F"
#echo ${NAME}
#python exp.py --algo cpo --cus 3358 --address 0606 --bid 226 --withf True --otherwarm False
#
#NAME="customer = 3386 WF"
#echo ${NAME}
#python exp.py --algo cpo --cus 3386 --address 0606 --bid 300 --withf False --otherwarm True
#NAME="customer = 3386 F"
#echo ${NAME}
#python exp.py --algo cpo --cus 3386 --address 0606 --bid 300 --withf True --otherwarm True
#
#NAME="customer = 3427 WF"
#echo ${NAME}
#python exp.py --algo cpo --cus 3427 --address 0606 --bid 234 --withf False --otherwarm True
#NAME="customer = 3427 F"
#echo ${NAME}
#python exp.py --algo cpo --cus 3427 --address 0606 --bid 234 --withf True --otherwarm True
#
#NAME="customer = 1458 WF"
#echo ${NAME}
#python exp.py --algo cpo --cus 1458 --address 0606 --bid 300 --withf False --otherwarm True
#NAME="customer = 1458 F"
#echo ${NAME}
#python exp.py --algo cpo --cus 1458 --address 0606 --bid 300 --withf True --otherwarm True
#
#NAME="customer = 2997 WF"
#echo ${NAME}
#python exp.py --algo cpo --cus 2997 --address 1021 --bid 277 --withf False --otherwarm True
#NAME="customer = 2997 F"
#echo ${NAME}
#python exp.py --algo cpo --cus 2997 --address 1021 --bid 277 --withf True --otherwarm True