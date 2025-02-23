import mujoco
import mujoco.viewer as viewer

def init_actuator(model, data, ctrl_mode="pos_mode", kp=5000, kv=200):
    """
    初始化执行器控制模式。

    参数:
        model: mujoco.MjModel 对象
        data: mujoco.MjData 对象
        ctrl_mode: 控制模式，可选 "pos_mode", "vel_mode", "tqe_mode"
        kp: 位置控制的比例增益（仅在 ctrl_mode == "pos_mode" 时使用）
        kv: 速度控制的增益（仅在 ctrl_mode == "pos_mode" 或 "vel_mode" 时使用）
    """
    # 遍历所有执行器
    for i in range(model.nu):
        if ctrl_mode == "pos_mode":
            # 位置控制模式
            model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED  # gaintype = fixed
            model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE  # biastype = affine
            model.actuator_gainprm[i] = kp  # gainprm = kp
            model.actuator_biasprm[i, 0:3] = [0, -kp, -kv]  # 设置前 3 个 biasprm

        elif ctrl_mode == "vel_mode":
            # 速度控制模式
            model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED  # gaintype = fixed
            model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE  # biastype = affine
            model.actuator_gainprm[i] = kv  # gainprm = kv
            model.actuator_biasprm[i, 0:3] = [0, 0, -kv]  # 设置前 3 个 biasprm

        elif ctrl_mode == "tqe_mode":
            # 扭矩控制模式
            model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED  # gaintype = fixed
            model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE  # biastype = none
            model.actuator_gainprm[i] = 1  # gainprm = 1
            model.actuator_biasprm[i, :] = 0  # 清空 biasprm
            model.actuator_ctrlrange[i] = [-1000, 1000]  # ctrlrange = [-50, 50]
        else:
            raise ValueError(f"未知的控制模式: {ctrl_mode}")
