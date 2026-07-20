## tune_3shot (few=3, NWPU)
- **HBBmAP@0.5**: 0.281, **HBBmAP@0.5:0.95**: 0.112
- Novel: airplane 0.552 | baseball-diamond 0.514 | tennis-court 0.012
- Base: storage-tank 0.709 | harbor 0.299 | vehicle 0.285 | ship 0.237 | ground-track-field 0.106 | bridge 0.081 | basketball-court 0.014

## 整体对比 (NWPU 少样本微调)

| 实验 | 总体 mAP@0.5 | 总体 mAP@0.5:0.95 |
|------|-------------|-------------------|
| 3shot | 28.1% | 11.2% |
| 5shot | 35.2% | 15.0% |

### Novel 类别对比
| 类别 | 3shot | 5shot | 提升 |
|------|-------|-------|------|
| airplane | 55.2% | 58.9% | +3.7% |
| baseball-diamond | 51.4% | 61.5% | +10.1% |
| tennis-court | 1.2% | 18.2% | +17.0% |
