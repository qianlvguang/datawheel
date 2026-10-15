"""飞轮流水线主入口 (Pipeline Generator)。
对接 generate.py 统一总控 CLI，支持全量执行与各维度参数透传。
"""
from generate import main, generate_demo, DEFAULT_OUTPUT_DIR

OUTPUT_DIR = DEFAULT_OUTPUT_DIR
IMAGE_DIR = OUTPUT_DIR / "demo" / "images"


def run_demo_flywheel_batch(num_samples: int = 6):
    """向后兼容接口：执行快速 Demo 生成"""
    res = generate_demo(output_dir=OUTPUT_DIR / "demo", debug=False)
    return res["dataset_path"]


if __name__ == "__main__":
    main()
