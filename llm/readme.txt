支持以下几种任务
1.单分类
2.多分类
3.回归
4.Embedding
5.文本生成
6.问答

Config.py 
--设定参数（量化参数，训练参数，基模型参数，文件目录参数）

LLMBase.py
--LLM的基本操作，通过transformers库，适配多种LLM,在import时，修改对应的LLM

LLMCommon.py
--常见的公共操作

LLMDataset.py
--数据集常见操作

LLMClassify.py
--单分类任务

LLMMulClassify.py
--多分类任务

LLMRegression.py
--回归任务

LLMEmbedding.py
--Embedding任务

LLMGenerate.py
--文本生成任务

LLMQA.py
--问答任务
