import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# 1.6m外，283mm半径，60rpm的步兵
# 等价于8度云台摆幅，4hz的正弦运动 的威力加强版
#########################设置##################################
#旋转目标半径
r = 283 #mm
#旋转目标转速
rpm = 120 #rpm

#期望自瞄云台单边摆幅
gimbal_angle = 8 #degree

#云台最大角加速度
max_acc = 200 # rad/s²

#云台转动惯量估计值（仅用于扭矩计算）
inertia = 20000  # kg·mm²

#云台最大角加速度作用时间搜索范围（非线性优化，有时需要反复调整直到曲线重合度较高）
time_length_using_max_acc_min = 38  #ms
time_length_using_max_acc_max = 40 #ms

#控制采样频率
draw_sampling_rate = 4000  #Hz
draw_time_end = 3000  #ms


#输出采样频率（采样间隔将被向下取整为整数ms）
output_sampling_rate = 400  #Hz
#############################################################



# 单位换算中间过程
max_acc_deg = max_acc * (180 / np.pi) / 1e6
amplitude = gimbal_angle /180 * np.pi  #rad
distance = 200 / amplitude + 200 #小角近似
period = 15000 / rpm #ms
draw_sampling_count = int(draw_time_end /1000 * draw_sampling_rate)    
sampling_interval = 1000 / output_sampling_rate
sampling_interval = math.floor(sampling_interval)



def calculate_position(t):
    a = (t % period) / period
    b = -np.pi * 3 / 4 + np.pi / 2 *a
    x = r * np.cos(b)
    y = r * np.sin(b) + distance
    return x, y

def calculate_angle(t):
    x, y = calculate_position(t)
    angle = math.atan2(x, y) * 180 / np.pi
    return angle

def calculate_angular_velocity(t):
    """计算角速度（一阶微分）- degrees/ms"""
    a = (t % period) / period
    b = -np.pi * 3 / 4 + np.pi / 2 * a
    
    # 计算位置
    x = r * np.cos(b)
    y = r * np.sin(b) + distance
    
    # db/dt
    db_dt = np.pi / (2 * period)
    
    # dx/dt 和 dy/dt
    dx_dt = -r * np.sin(b) * db_dt
    dy_dt = r * np.cos(b) * db_dt
    
    # d(atan2(x,y))/dt = (y*dx/dt - x*dy/dt) / (x^2 + y^2)
    angular_velocity_rad = (y * dx_dt - x * dy_dt) / (x**2 + y**2)
    
    # 转换为 degrees/ms
    return angular_velocity_rad * 180 / np.pi

def calculate_angular_acceleration(t):
    """计算角加速度（二阶微分）- degrees/ms²"""
    a = (t % period) / period
    b = -np.pi * 3 / 4 + np.pi / 2 * a
    
    # 计算位置
    x = r * np.cos(b)
    y = r * np.sin(b) + distance
    
    # 一阶导数
    db_dt = np.pi / (2 * period)
    dx_dt = -r * np.sin(b) * db_dt
    dy_dt = r * np.cos(b) * db_dt
    
    # 二阶导数（db/dt是常数，所以d²b/dt² = 0）
    d2x_dt2 = -r * np.cos(b) * db_dt**2
    d2y_dt2 = -r * np.sin(b) * db_dt**2
    
    # 使用商法则计算角加速度
    # d/dt[(y*dx/dt - x*dy/dt) / (x^2 + y^2)]
    numerator = y * dx_dt - x * dy_dt
    denominator = x**2 + y**2
    
    d_numerator_dt = dy_dt * dx_dt + y * d2x_dt2 - dx_dt * dy_dt - x * d2y_dt2
    d_numerator_dt = y * d2x_dt2 - x * d2y_dt2  # 简化后
    
    d_denominator_dt = 2 * x * dx_dt + 2 * y * dy_dt
    
    # 商法则
    angular_acceleration_rad = (d_numerator_dt * denominator - numerator * d_denominator_dt) / (denominator**2)
    
    # 转换为 degrees/ms²
    return angular_acceleration_rad * 180 / np.pi



def calculate_angular_acceleration_with_max_acc(t):
    """计算角加速度（二阶微分）- degrees/ms²"""
    a = (t % period) / period
    b = -np.pi * 3 / 4 + np.pi / 2 * a
    
    # 计算位置
    x = r * np.cos(b)
    y = r * np.sin(b) + distance
    
    # 一阶导数
    db_dt = np.pi / (2 * period)
    dx_dt = -r * np.sin(b) * db_dt
    dy_dt = r * np.cos(b) * db_dt
    
    # 二阶导数（db/dt是常数，所以d²b/dt² = 0）
    d2x_dt2 = -r * np.cos(b) * db_dt**2
    d2y_dt2 = -r * np.sin(b) * db_dt**2
    
    # 使用商法则计算角加速度
    # d/dt[(y*dx/dt - x*dy/dt) / (x^2 + y^2)]
    numerator = y * dx_dt - x * dy_dt
    denominator = x**2 + y**2
    
    d_numerator_dt = dy_dt * dx_dt + y * d2x_dt2 - dx_dt * dy_dt - x * d2y_dt2
    d_numerator_dt = y * d2x_dt2 - x * d2y_dt2  # 简化后
    
    d_denominator_dt = 2 * x * dx_dt + 2 * y * dy_dt
    
    # 商法则
    angular_acceleration_rad = (d_numerator_dt * denominator - numerator * d_denominator_dt) / (denominator**2)

    if((t % period) < time_length_using_max_acc and t>period):
        return max_acc_deg
    
    if((period - (t % period)) < time_length_using_max_acc ):
        return -max_acc_deg
        
    # 转换为 degrees/ms²
    return angular_acceleration_rad * 180 / np.pi

def integrate_acceleration_to_velocity(times):
    """对角加速度进行数值积分得到角速度"""
    velocities = [calculate_angular_velocity(times[0])]  # 初始角速度为0
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        acc = calculate_angular_acceleration_with_max_acc(times[i-1])
        velocities.append(velocities[-1] + acc * dt)
    return velocities

def integrate_acceleration_to_angle(times):
    """对角加速度进行二阶积分得到角度"""
    velocities = [calculate_angular_velocity(times[0])]  # 初始角速度为0
    angles = [calculate_angle(times[0])]  # 初始角度从原始函数获取
    
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        acc = calculate_angular_acceleration_with_max_acc(times[i-1])
        # 更新角速度
        new_velocity = velocities[-1] + acc * dt
        velocities.append(new_velocity)
        # 更新角度（使用梯形法则）
        angles.append(angles[-1] + (velocities[-2] + velocities[-1]) / 2 * dt)
    
    return angles, velocities

def calculate_overlap_duration(time_length, times, threshold=0.1):
    """计算在给定time_length下，两个角度曲线重合的总时长"""
    global time_length_using_max_acc
    time_length_using_max_acc = time_length
    
    angles_original = [calculate_angle(t) for t in times]
    angles_int, _ = integrate_acceleration_to_angle(times)
    
    # 计算差值小于阈值的总时长
    overlap_count = sum(1 for i in range(len(times)) if abs(angles_original[i] - angles_int[i]) < threshold)
    overlap_duration = overlap_count * (times[1] - times[0])
    
    return overlap_duration

if __name__ == "__main__":
    times = np.linspace(0, draw_time_end, draw_sampling_count)
    
    # 定义目标函数（返回负值因为要最大化重合时长）
    def objective(time_length):
        overlap = calculate_overlap_duration(time_length, times, threshold=0.1)
        print(f"time_length = {time_length:.3f} ms, 重合时长 = {overlap:.2f} ms")
        return -overlap  # 返回负值用于最小化
    
    # 使用 scipy 优化器搜索最优的 time_length_using_max_acc
    print("使用 scipy 优化器搜索最优的 time_length_using_max_acc...\n")
    
    result = minimize_scalar(objective, bounds=(time_length_using_max_acc_min, time_length_using_max_acc_max), method='bounded', 
                            options={'xatol': 0.01})
    
    best_time_length = result.x
    best_overlap = -result.fun
    
    print(f"\n最优结果: time_length_using_max_acc = {best_time_length:.3f} ms")
    print(f"重合时长 = {best_overlap:.2f} ms (占比 {best_overlap/1000*100:.1f}%)\n")
    
    # 使用最优值绘图
    time_length_using_max_acc = best_time_length
    
    # 第一组：原始解析函数
    angles = [calculate_angle(t) for t in times]
    omega = [calculate_angular_velocity(t) for t in times]
    acceleration = [calculate_angular_acceleration(t) for t in times]
    
    # 第二组：带最大加速度限制的函数及其积分
    acceleration_with_max = [calculate_angular_acceleration_with_max_acc(t) for t in times]
    angles_integrated, omega_integrated = integrate_acceleration_to_angle(times)

    max_toque = max_acc * inertia / 1e6
    print(f"最大扭矩需求约为(上限): {max_toque:.4f} Nm\n")
    max_speed = np.max(np.abs(omega_integrated)) *1000
    print(f"最大角速度需求约为: {max_speed:.4f} deg/s\n")


    # 改为纵向排布的三个子图
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # 角度对比
    axs[0].plot(times, angles, label='Original', linewidth=2)
    axs[0].plot(times, angles_integrated, label='With Max Acc (Integrated)', linestyle='--', linewidth=2)
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Angle (degrees)')
    axs[0].set_title('Angle vs Time')
    axs[0].legend()
    axs[0].grid()

    # 角速度对比
    axs[1].plot(times, omega, label='Original', color='orange', linewidth=2)
    axs[1].plot(times, omega_integrated, label='With Max Acc (Integrated)', color='red', linestyle='--', linewidth=2)
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Angular Velocity (degrees/ms)')
    axs[1].set_title('Angular Velocity vs Time')
    axs[1].legend()
    axs[1].grid()

    # 角加速度对比
    axs[2].plot(times, acceleration, label='Original', color='green', linewidth=2)
    axs[2].plot(times, acceleration_with_max, label='With Max Acc', color='purple', linestyle='--', linewidth=2)
    axs[2].set_xlabel('Time (ms)')
    axs[2].set_ylabel('Angular Acceleration (degrees/ms²)')
    axs[2].set_title('Angular Acceleration vs Time')
    axs[2].legend()
    axs[2].grid()
    
    plt.tight_layout()
    plt.show()





    # 采样输出
    # 向下取整
    sample_times = []
    t = 0
    while t <= times[-1]:
        sample_times.append(t)
        t += sampling_interval
    
    # 对 angles_integrated 和 omega_integrated 进行线性插值
    sample_angles = np.interp(sample_times, times, angles_integrated)
    sample_velocities = np.interp(sample_times, times, omega_integrated)

    sample_velocities = sample_velocities*1000/180*np.pi  # 转换为 rad/s
    
    # 写入txt文件
    with open('sample_output.txt', 'w', encoding='utf-8') as f:
        f.write("// 云台采样输出文件\n")
        f.write("// 格式：{time(ms),pitch(deg),pitch_spd(rad/s),yaw(deg),yaw_spd(rad/s)},\n")
        f.write("{\n")
        for i in range(len(sample_times)):
            time = sample_times[i]
            angle = sample_angles[i]
            velocity = sample_velocities[i]
            f.write(f"{{{time:.1f},0.0,0.0,{angle:.6f},{velocity:.6f}}},\n")
        
        count = len(sample_times)
        f.write("}\n")
        f.write(f"count = {count}\n")
    
    print(f"\n采样数据已写入 sample_output.txt，共 {len(sample_times)} 个采样点")

    print("\n读取采样输出文件并绘制...")

    # 读取文件
    read_times = []
    read_pitch = []
    read_pit_spd = []
    read_yaw = []
    read_yaw_spd = []

    with open('sample_output.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('{') and line.strip().endswith('},'):
                # 解析格式：{time,pitch,pit_spd,yaw,yaw_spd},
                content = line.strip()[1:-2]  # 去掉 { 和 },
                values = [float(x) for x in content.split(',')]
                read_times.append(values[0])
                read_pitch.append(values[1])
                read_pit_spd.append(values[2])
                read_yaw.append(values[3])
                read_yaw_spd.append(values[4])

    # 创建4个子图
    fig2, axs2 = plt.subplots(4, 1, figsize=(10, 14))

    # 子图1：时间 vs Yaw角度
    axs2[0].plot(read_times, read_yaw, color='blue', linewidth=2)
    axs2[0].set_xlabel('Time (ms)')
    axs2[0].set_ylabel('Yaw Angle (degrees)')
    axs2[0].set_title('Yaw Angle vs Time')
    axs2[0].grid()

    # 子图2：时间 vs Yaw角速度
    axs2[1].plot(read_times, read_yaw_spd, color='red', linewidth=2)
    axs2[1].set_xlabel('Time (ms)')
    axs2[1].set_ylabel('Yaw Angular Velocity (rad/s)')
    axs2[1].set_title('Yaw Angular Velocity vs Time')
    axs2[1].grid()

    # 子图3：时间 vs Pitch角度
    axs2[2].plot(read_times, read_pitch, color='green', linewidth=2)
    axs2[2].set_xlabel('Time (ms)')
    axs2[2].set_ylabel('Pitch Angle (degrees)')
    axs2[2].set_title('Pitch Angle vs Time')
    axs2[2].grid()

    # 子图4：时间 vs Pitch角速度
    axs2[3].plot(read_times, read_pit_spd, color='orange', linewidth=2)
    axs2[3].set_xlabel('Time (ms)')
    axs2[3].set_ylabel('Pitch Angular Velocity (rad/s)')
    axs2[3].set_title('Pitch Angular Velocity vs Time')
    axs2[3].grid()

    plt.tight_layout()
    plt.show()