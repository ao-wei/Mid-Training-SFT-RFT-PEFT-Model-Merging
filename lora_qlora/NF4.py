import numpy as np

# 有些会用非对称的量化范围，也就是-0代表-8，用二进制表示就是1000，这样的一个表示范围就是[-8, 7]
# 用-7到7的原因是为了保证量化的对称性（正负范围相等），能使得零点精确对应到0
int4_row = np.array([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7])

factor = np.abs(int4_row).max()
int4_level = int4_row / factor

