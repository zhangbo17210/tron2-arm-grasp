# Data Directory

## raw/
原始采集数据，建议命名格式：
- `demo_episodes_001.npy` — 演示 episodes（joint_pos, joint_vel, action, obs）
- `expert_demos.hdf5` — 专家演示数据
- `real_robot_log.csv` — 真实机器人日志

## processed/
预处理后的数据：
- `train_dataset.npy` — 训练用张量
- `val_dataset.npy` — 验证集
