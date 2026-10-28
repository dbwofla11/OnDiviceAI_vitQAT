# OnDiviceAI_vitQAT

W2 가중치·A8 attention의 TensorRT INT8 export, 정확도·실험 지표 및 벤치마크: [docs/tensorrt-int8.md](docs/tensorrt-int8.md).
온디바이스 AI - vit , QAT , ptq

```powershell
conda env create -f environment.yml
conda activate vitqat
```

Food-101 scratch 학습 실험 **1-1 FP32 ViT / 1-2 QAT-only ViT**의 실행 방법과 지표 정의는
[docs/experiments.md](docs/experiments.md)를 참고하세요.
