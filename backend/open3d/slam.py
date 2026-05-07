import open3d as o3d
import numpy as np

device = o3d.core.Device("CPU:0")  # RTX 5060 Ti

# Initialize SLAM model
voxel_size = 0.006  # 6mm
model = o3d.t.pipelines.slam.Model(
    voxel_size, 16, 10000,
    o3d.core.Tensor(np.identity(4)),
    device
)
print("SLAM model initialized on GPU")