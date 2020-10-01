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
|  custom CART未剪枝   |                        |        |
|  custom CART预剪枝   |                        |        |
|  custom CART后剪枝   |                        |        |
|     sklearn CART     |                        |        |
| custom RandomForest  |                        |        |
| sklearn RandomForest |                        |        |
|   custom Adaboost    |                        |        |
|   sklearn Adaboost   |                        |        |
|     custom GBDT      |                        |        |
|     sklearn GBDT     |                        |        |
|    custom XGBoost    |                        |        |
|       XGBoost        |                        |        |
|   custom LightGBM    |                        |        |
|       LightGBM       |                        |        |
|   custom CatBoost    |                        |        |
|       CatBoost       |                        |        |

## 决策树总结

* 未考虑缺失值处理

* 在ID3中，处理连续值时，是选择使information gain最大的切分点，

  在C4.5中是仍然选择使information gain最大的切分点还是选择使gain ratio最大的切分点？

  我实现的时候选择了使gain ratio最大的切分点。

## 说明

* 使用60000张训练很慢，因此只用了1000张
*  

## 问题 

* 

## 参考

