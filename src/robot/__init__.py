# robot/__init__.py

# 导入核心功能
from .actuator_init import init_actuator
from .robot_fun import Robot_mj

# 可以定义包的版本或其他元信息
__version__ = "0.1.0"
__all__ = ["init_actuator", "Robot_mj"]  # 定义外部可访问的内容