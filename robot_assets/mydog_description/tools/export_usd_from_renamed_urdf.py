#!/usr/bin/env python3
"""
脚本功能：从重命名后的mydog_rl_ready.urdf导出USD文件
"""

import os
import subprocess
from pathlib import Path

def export_usd_from_urdf():
    """从URDF导出USD文件"""
    project_root = "/home/nszb/python_text/robot_assets/mydog_description"
    urdf_path = os.path.join(project_root, "urdf", "mydog_rl_ready.urdf")
    usd_output_path = os.path.join(project_root, "usd", "mydog_rl_ready.usd")
    
    # 确保USD目录存在
    usd_dir = os.path.dirname(usd_output_path)
    os.makedirs(usd_dir, exist_ok=True)
    
    print(f"正在从URDF导出USD文件...")
    print(f"输入URDF: {urdf_path}")
    print(f"输出USD: {usd_output_path}")
    
    try:
        # 使用urdf-to-usd工具或其他可用方法转换
        # 这里提供一个示例命令，具体实现可能需要根据系统中安装的工具调整
        cmd = [
            "python", 
            "-m", "urdf2usd", 
            urdf_path, 
            usd_output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("USD导出成功!")
        print(result.stdout)
        
    except FileNotFoundError:
        print("警告: urdf2usd工具未找到，尝试手动创建USD路径")
        print("请确保已安装urdf2usd或Omniverse Isaac Gym中的相应工具")
        
        # 创建一个基础USD文件作为占位符
        with open(usd_output_path, 'w') as f:
            f.write("""#usda 1.0
# 
# This is a placeholder USD file for the MyDog robot.
# It will be replaced when you run the actual URDF to USD conversion.

def Xform MyDog (
    prepend references = </common/ground_plane>
)
{
    def Xform base_link
    {
    }
}
""")
        print(f"已创建USD占位符文件: {usd_output_path}")
    
    except subprocess.CalledProcessError as e:
        print(f"USD导出失败: {e}")
        print(f"stderr: {e.stderr}")
        print("请确保urdf2usd工具已正确安装")

if __name__ == "__main__":
    export_usd_from_urdf()