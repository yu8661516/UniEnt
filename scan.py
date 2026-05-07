import os
import time

# ===================== 适配 UniEnt 项目 + WSL 环境 =====================
# 自动获取当前目录（无需手动修改！）
ROOT = os.getcwd()
# 忽略：缓存、权重文件夹、数据集大文件、版本控制、临时文件
IGNORE_DIRS = (
    "__pycache__", ".git", "ckpt", "data", "logs", "tmp", 
    ".vscode", ".idea", "dist", "build"
)
# 只扫描：UniEnt 核心文件（Python代码、文档、配置、脚本）
TARGET_EXTS = (
    ".py", ".md", ".sh", ".yaml", ".yml", ".txt", 
    ".gitignore", "README", "LICENSE"
)
# ====================================================================

def scan(path, indent=0):
    try:
        for item in sorted(os.listdir(path)):
            full_path = os.path.join(path, item)
            # 跳过忽略的文件夹
            if os.path.isdir(full_path) and item in IGNORE_DIRS:
                continue
            # 缩进展示层级
            space = "  " * indent
            
            if os.path.isdir(full_path):
                print(f"{space}📂 {item}/")
                scan(full_path, indent + 1)
            else:
                # 只保留核心文件
                if item.endswith(TARGET_EXTS) or item in ["README.md", "requirements.txt"]:
                    size = os.path.getsize(full_path)
                    size_str = f"{size/1024:.1f}KB" if size > 1024 else f"{size}B"
                    print(f"{space}📄 {item}  ({size_str})")
    except:
        pass

if __name__ == "__main__":
    print("="*60)
    print(f"📁 UniEnt 项目核心文件扫描 | 路径: {ROOT}")
    print("="*60)
    scan(ROOT)
    print("="*60)
    print("✅ 扫描完成！")
