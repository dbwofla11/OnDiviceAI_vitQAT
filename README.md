# OnDiviceAI_vitQAT

Food-101 ViT의 양자화와 token pruning 비교 실험.

| 폴더 | 실험 |
|---|---|
| [A1](docs/A1/README.md) | FP32 dense 기준선 |
| [A2](docs/A2/README.md) | QAT-only |
| [A3](docs/A3/execution.md) | QAT 후 CLS attention pruning |
| [A4](docs/A4/README.md) | CLS attention pruning + QAT 공동 학습 |
| [B1](docs/B1/README.md) | Cropr + Linear QAT(LSQ/LSQ+) 공동 학습 |
| [B2](docs/B2/README.md) | Cropr + QAT 공동 학습 |

각 실험은 `model.py`, `train.py`, `inference.py`와 자체 양자화·pruning 구현을 소유한다.
`utils/`에는 데이터·checkpoint·계측·반복 실행 같은 공통 기반 기능만 둔다.
모든 QAT 실험의 추론은 학습한 quantizer를 그대로 쓰므로 학습과 양자화 방식(대칭/비대칭)이 같다.
B1은 정수 latency를 CPU(INT8 커널)와 TensorRT(RTX 4090)로 잰다([B1/README.md](B1/README.md)). 다른 실험에는 정수 실행 경로가 없다.
[폴더 구조와 이전 경로 안내](docs/experiment-folder-structure.md),
[공통 학습 조건·지표](docs/experiments.md).

```powershell
conda env create -f environment.yml
conda activate vitqat
python -m A2.train --data-root data --output A2/runs/seed42
python -m A2.inference --eval-checkpoint A2/runs/seed42/best.pt --data-root data --output A2/runs/eval
python -m pytest tests A3/tests A4/tests B1/tests B2/tests -q
```

Python 의존성은 [requirements.txt](requirements.txt) 하나로 관리한다. `environment.yml`도 이 파일을 참조한다.
