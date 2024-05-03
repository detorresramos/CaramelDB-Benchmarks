import os
from carameldb import Caramel

keys = [str(i) for i in range(1000)]
values = [[0, 0] for i in range(700)] + [[i, i] for i in range(300)]
csf = Caramel(keys, values, verbose=False)
savepath = "file.csf"
csf.save(savepath)
print(os.stat(savepath).st_size)


keys = [str(i) for i in range(1000)]
values = [[i, i] for i in range(1000)]
csf = Caramel(keys, values, verbose=False)
savepath = "file.csf"
csf.save(savepath)
print(os.stat(savepath).st_size)
