#!/usr/bin/env bash
# 批量预拉取 SWE-bench 镜像
# 用法: ./pull_swebench_images.sh [subset] [split] [并发数]
#   subset: verified (默认) / lite / full / 本地数据集路径
#   split:  test (默认) / dev（本地路径时固定为 test）
#   并发数: 默认 4

set -euo pipefail

SUBSET="${1:-verified}"
SPLIT="${2:-test}"
WORKERS="${3:-4}"

python3 - "$SUBSET" "$SPLIT" "$WORKERS" <<'EOF'
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tqdm"])
    from tqdm import tqdm

DATASET_MAP = {
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite":     "princeton-nlp/SWE-Bench_Lite",
    "full":     "princeton-nlp/SWE-Bench",
}

subset, split, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])

# 加载数据集
local = Path(subset)
if local.is_dir() and (local / "dataset_dict.json").exists():
    from datasets import load_from_disk
    dataset = load_from_disk(subset)[split]
else:
    from datasets import load_dataset
    dataset = load_dataset(DATASET_MAP.get(subset, subset), split=split)

images = []
for inst in dataset:
    iid = inst["instance_id"]
    id_docker = iid.replace("__", "_1776_")
    images.append(f"docker.io/swebench/sweb.eval.x86_64.{id_docker}:latest".lower())

total = len(images)
print(f"共 {total} 个镜像，并发数 {workers}\n")

results = {"ok": 0, "skip": 0, "fail": 0}

def pull(image):
    # 已存在则跳过
    r = subprocess.run(["docker", "image", "inspect", image],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode == 0:
        return "skip", image
    r = subprocess.run(["docker", "pull", image],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode == 0:
        return "ok", image
    return "fail", image

failed = []
with ThreadPoolExecutor(max_workers=workers) as executor:
    futures = {executor.submit(pull, img): img for img in images}
    with tqdm(total=total, unit="img", dynamic_ncols=True) as bar:
        for future in as_completed(futures):
            status, image = future.result()
            results[status] += 1
            bar.set_postfix(ok=results["ok"], skip=results["skip"], fail=results["fail"])
            bar.update(1)
            if status == "fail":
                failed.append(image)

print(f"\n完成: {results['ok']} 拉取成功, {results['skip']} 跳过(已存在), {results['fail']} 失败")
if failed:
    print("失败的镜像:")
    for img in failed:
        print(f"  {img}")
    sys.exit(1)
EOF
