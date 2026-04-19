#!/usr/bin/env python3
"""
脚本功能：将MyDog URDF文件及关联的STL网格文件重命名为Isaac Lab兼容的命名方式
支持的输入文件：mydog.urdf 或 mydog_abs.urdf
输出文件：mydog_rl_ready.urdf
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from pathlib import Path
import shutil
import re
def create_semantic_mapping():
    """只定义四足机器人需要的标准编号映射，不做额外推断。"""
    return {
        "11": "lf_hip",
        "12": "lf_thigh",
        "13": "lf_calf",
        "14": "lf_foot",

        "21": "rf_hip",
        "22": "rf_thigh",
        "23": "rf_calf",
        "24": "rf_foot",

        "31": "rh_hip",
        "32": "rh_thigh",
        "33": "rh_calf",
        "34": "rh_foot",

        "41": "lh_hip",
        "42": "lh_thigh",
        "43": "lh_calf",
        "44": "lh_foot",
    }

def update_urdf(input_path, output_path, stl_dir):
    """更新URDF文件中的命名"""
    print(f"正在处理URDF文件: {input_path}")
    
    # 解析XML
    tree = ET.parse(input_path)
    root = tree.getroot()
    
    # 获取命名映射
    mapping = create_semantic_mapping()
    
    # 更新所有link名称
    for link in root.findall('.//link'):
        old_name = link.get('name')
        if old_name and old_name in mapping:
            new_name = mapping[old_name]
            print(f"  更新link名称: {old_name} -> {new_name}")
            link.set('name', new_name)
    
    # 更新所有joint名称和父子关系
    for joint in root.findall('.//joint'):
        old_name = joint.get('name')
        if old_name and old_name in mapping:
            new_name = mapping[old_name]
            print(f"  更新joint名称: {old_name} -> {new_name}")
            joint.set('name', new_name)
        
        # 更新parent link
        parent_elem = joint.find('parent')
        if parent_elem is not None:
            old_parent = parent_elem.get('link')
            if old_parent and old_parent in mapping:
                new_parent = mapping[old_parent]
                print(f"  更新parent link: {old_parent} -> {new_parent}")
                parent_elem.set('link', new_parent)
        
        # 更新child link
        child_elem = joint.find('child')
        if child_elem is not None:
            old_child = child_elem.get('link')
            if old_child and old_child in mapping:
                new_child = mapping[old_child]
                print(f"  更新child link: {old_child} -> {new_child}")
                child_elem.set('link', new_child)
    
    # 更新mesh文件路径
    for geometry in root.findall('.//geometry/mesh'):
        filename = geometry.get('filename')
        if filename:
            # 提取文件名部分
            original_stl_name = os.path.basename(filename).replace('.STL', '').replace('.stl', '')
            
            if original_stl_name.isdigit() and original_stl_name in mapping:
                new_stl_name = mapping[original_stl_name]
                # 构建新的绝对路径
                new_filename = f"/home/nszb/python_text/robot_assets/mydog_description/meshes/{new_stl_name}.STL"
                print(f"  更新mesh路径: {filename} -> {new_filename}")
                geometry.set('filename', new_filename)
    
    # 保存更新后的URDF
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"已保存更新后的URDF到: {output_path}")
def rename_stl_files(mesh_dir):
    """
    只复制/重命名四足标准 STL:
    11~44 中的 16 个文件。
    其他文件直接跳过，不报错。
    """
    mesh_dir = Path(mesh_dir)
    mapping = create_semantic_mapping()

    print(f"正在处理STL文件：{mesh_dir}")

    for file in mesh_dir.iterdir():
        if not file.is_file():
            continue

        if file.suffix.lower() != ".stl":
            continue

        stem = file.stem.strip()

        # 只接受完全等于 11~44 的名字
        if stem not in mapping:
            print(f"跳过非标准STL文件: {file.name}")
            continue

        new_name = mapping[stem] + file.suffix
        new_path = mesh_dir / new_name

        if new_path.exists():
            print(f"已存在，跳过: {new_path.name}")
            continue

        shutil.copy2(file, new_path)
        print(f"复制: {file.name} -> {new_name}")

def main():
    """主函数"""
    project_root = "/home/nszb/python_text/robot_assets/mydog_description"
    urdf_dir = os.path.join(project_root, "urdf")
    mesh_dir = os.path.join(project_root, "meshes")
    tools_dir = os.path.join(project_root, "tools")
    
    # 创建工具目录
    os.makedirs(tools_dir, exist_ok=True)
    
    # 查找输入URDF文件
    input_urdf = None
    if os.path.exists(os.path.join(urdf_dir, "mydog_abs.urdf")):
        input_urdf = os.path.join(urdf_dir, "mydog_abs.urdf")
    elif os.path.exists(os.path.join(urdf_dir, "mydog.urdf")):
        input_urdf = os.path.join(urdf_dir, "mydog.urdf")
    
    if not input_urdf:
        print("错误: 找不到mydog_abs.urdf或mydog.urdf文件")
        return
    
    output_urdf = os.path.join(urdf_dir, "mydog_rl_ready.urdf")
    
    print("=" * 60)
    print("开始处理MyDog URDF重命名")
    print(f"输入文件: {input_urdf}")
    print(f"输出文件: {output_urdf}")
    print("=" * 60)
    
    # 重命名STL文件
    rename_stl_files(mesh_dir)
    
    # 更新URDF文件
    update_urdf(input_urdf, output_urdf, mesh_dir)
    
    print("=" * 60)
    print("处理完成!")
    print(f"新的RL准备好的URDF文件位置: {output_urdf}")
    print("对应的STL文件已经重命名并放置在meshes目录中")
    print("=" * 60)

if __name__ == "__main__":
    main()