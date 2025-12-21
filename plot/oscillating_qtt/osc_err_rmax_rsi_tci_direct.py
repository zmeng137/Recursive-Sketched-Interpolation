import numpy as np
import matplotlib.pyplot as plt

rank = {}
rank['f1f2'] = []
rank['f1f2f2'] = []
rank['f1f1f2'] = []
rank['f1f1f2f2'] = [] 
rank['f1f1f1f2f2'] = []

relerror = {}

cont_no=2; rmax=5; oversampling=10: 0.09700581573010185
cont_no=2; rmax=10; oversampling=10: 0.0040065461318637245
cont_no=2; rmax=15; oversampling=10: 5.989002420101887e-05
cont_no=2; rmax=20; oversampling=10: 3.3601966785815754e-06
cont_no=2; rmax=25; oversampling=10: 3.9873095752288126e-08
cont_no=2; rmax=30; oversampling=10: 5.4416955999994505e-09

rounding rmax=5: 0.011825581626197628
rounding rmax=10: 4.472627215835557e-05
rounding rmax=15: 3.0851801264194874e-08
rounding rmax=20: 4.324351236180593e-09
rounding rmax=25: 4.324351307400081e-09
rounding rmax=30: 4.324351155509327e-09