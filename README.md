# pairwise-terrain-lod-crack-stitch

## 领域

游戏、图形与多媒体

## 主题

地形 LOD 裂缝缝合

## 说明

边界顶点对齐、层级滞回与尺寸分桶。

初始状态下 `graphics/terrain_lod_crack_stitch.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
