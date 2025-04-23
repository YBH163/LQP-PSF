'''
from matplotlib.path import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch

# 假设您已经有了一组顶点
vertices = np.array([[0, 0], [1,0], [2,0], [2,1], [3,0], [3,1]])  # 示例顶点

# 创建多边形
polygon = Polygon(vertices, closed=True)

# 创建路径
# path = Path(vertices[:,0], vertices[:,1])
path = Path(vertices)

# 创建路径上的多边形
path_patch = PathPatch(path)

# 2. 将多边形添加到图表中
fig, ax = plt.subplots()
ax.add_patch(path_patch)

#3. 填充多边形的区域
ax.fill(0, 0, polygon.get_facecolor(), alpha=0.5)
'''

'''
from matplotlib.path import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch

# 假设您已经有了一组顶点
vertices = np.array([[0, 0], [1, 0], [2, 0], [2, 1], [3, 0], [3, 1]])  # 示例顶点

# 创建多边形
polygon = Polygon(vertices, closed=True)

# 创建 Path 对象
path = Path(vertices)

# 创建路径上的多边形
path_patch = PathPatch(path, facecolor='none', edgecolor='r', lw=2)

# 创建图形和轴
fig, ax = plt.subplots()

# 将多边形添加到图表中
ax.add_patch(polygon)  # 添加多边形
ax.add_patch(path_patch)  # 添加路径上的多边形

# 填充多边形的区域
ax.fill(*vertices[:, 0], *vertices[:, 1], 'b', alpha=0.5)  # 使用顶点坐标填充区域

# 设置坐标轴标签和标题
ax.set_xlabel('theta')
ax.set_ylabel('theta_dot')
ax.set_title('Terminal Set in theta-theta_dot Plane')

# 显示网格
ax.grid(True)

# 保存图像到文件
plt.savefig('terminal_set.png')

# 显示图像
plt.show()
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# 定义顶点，确保按照逆时针顺序
vertices = np.array([[0, 0], [1, 0], [2, 1], [1, 2], [0, 1], [0, 0]])  # 注意最后一个点与第一个点相同，以闭合多边形

# 创建多边形
polygon = Polygon(vertices, closed=True, fill=True, edgecolor='red', facecolor='blue', linewidth=2, alpha=0.5)

# 创建图形和轴
fig, ax = plt.subplots()

# 将多边形添加到轴上
ax.add_patch(polygon)

# 设置坐标轴范围以包含所有顶点
ax.set_xlim(-2, 3)
ax.set_ylim(-2, 3)

# 设置坐标轴标签和标题
ax.set_xlabel('theta')
ax.set_ylabel('theta_dot')
ax.set_title('Terminal Set in theta-theta_dot Plane')

# 显示网格
ax.grid(True)

# 显示图形
plt.show()
# 保存图像到文件
plt.savefig('test.png')