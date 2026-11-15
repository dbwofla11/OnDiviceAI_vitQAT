"""여러 테스트 파일이 함께 쓰는 fixture."""
import json

import pytest
from PIL import Image


@pytest.fixture
def food_fixture(tmp_path):
    """진짜 torchvision Food101 로더가 읽을 수 있는 아주 작은 가짜 Food-101 폴더.

    101개 클래스 x (train 3장, test 1장)의 40x40 단색 이미지다. 클래스 수가 101개여야
    make_data의 클래스 수 검사를 통과한다. 반환값은 data_root로 쓸 폴더 경로다.
    """
    root = tmp_path / "food-101"
    (root / "meta").mkdir(parents=True)
    manifests = {"train": {}, "test": {}}
    for c in range(101):
        name = f"class_{c:03d}"
        (root / "images" / name).mkdir(parents=True)
        for split, count in [("train", 3), ("test", 1)]:
            manifests[split][name] = []
            for i in range(count):
                rel = f"{name}/{split}_{i}"
                Image.new("RGB", (40, 40), (c * 2, 30 + i * 40, 80)).save(root / "images" / f"{rel}.jpg")
                manifests[split][name].append(rel)
    for split, manifest in manifests.items():
        (root / "meta" / f"{split}.json").write_text(json.dumps(manifest))
    return tmp_path
