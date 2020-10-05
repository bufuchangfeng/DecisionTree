# DecisionTree

## 数据

* mnist_all.mat 

  mnist手写数字识别数据集

## 模型

* ID3
* C4.5
* CART
* RandomForest
* Adaboost
* GBDT
* XGBoost
* LightGBM
* Catboost

## 结果

|         模型         |          数据          |  结果  |
| :------------------: | :--------------------: | :----: |
|   custom ID3未剪枝   | 1000张训练+10000张测试 | 0.6468 |
|   custom ID3预剪枝   |                        |        |
|   custom ID3后剪枝   |                        |        |
|      skearn ID3      | 1000张训练+10000张测试 | 0.6331 |
|  custom C4.5未剪枝   | 1000张训练+10000张测试 | 0.6173 |
|  custom C4.5预剪枝   |                        |        |
|  custom C4.5后剪枝   |                        |        |
|  custom CART未剪枝   | 1000张训练+10000张测试 | 0.6721 |
|  custom CART后剪枝   |                        |        |
|     sklearn CART     | 1000张训练+10000张测试 | 0.6486 |
| custom RandomForest  | 1000张训练+10000张测试 | 0.7844 |
| sklearn RandomForest | 1000张训练+10000张测试 | 0.8254 |
|   custom Adaboost    | 1000张训练+10000张测试 | 0.7216 |
|   sklearn Adaboost   | 1000张训练+10000张测试 | 0.7325 |
|     custom GBDT      |                        |        |
|     sklearn GBDT     |                        |        |
|    custom XGBoost    |                        |        |
|       XGBoost        |                        |        |
|   custom LightGBM    |                        |        |
|       LightGBM       |                        |        |
|   custom CatBoost    |                        |        |
|       CatBoost       |                        |        |

## 说明

* 未考虑缺失值处理

* 在ID3中，处理连续值时，是选择使information gain最大的切分点，

  在C4.5中是仍然选择使information gain最大的切分点还是选择使gain ratio最大的切分点？

  我实现的时候选择了使gain ratio最大的切分点。

* 使用60000张训练很慢，因此只用了1000张

*  我实现的决策树之所以慢，一个可能的原因是采用了连续的特征，而不是将图像二值化

* 

  



## 参考

* <https://github.com/serengil/decision-trees-for-ml>
* <https://zhuanlan.zhihu.com/p/145215188>
* <https://github.com/Rudo-erek/decision-tree>
* 