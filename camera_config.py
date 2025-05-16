import pyrealsense2 as rs

# 启动管线并配置深度流（宽x高、格式、帧率随你项目而定）
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile  = pipeline.start(config)

# 从 profile 中取出深度流的 video stream profile
depth_stream = profile.get_stream(rs.stream.depth)
video_profile = depth_stream.as_video_stream_profile()
intrinsics = video_profile.get_intrinsics()

# intrinsics 里就包含了常用的参数
fx, fy = intrinsics.fx, intrinsics.fy       # 焦距（像素）
cx, cy = intrinsics.ppx, intrinsics.ppy     # 主点坐标（像素）
w, h   = intrinsics.width, intrinsics.height
model  = intrinsics.model                   # 畸变模型
coeffs = intrinsics.coeffs                  # 畸变系数列表
