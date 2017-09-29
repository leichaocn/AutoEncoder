# AutoEncoder
实现一个去噪自编码器。
#key point
1.自编码
   通过对数据降唯，解离出高阶特征，这些高阶特征可用于近似重构输入。
2.去噪
   通过加入噪声，
   Vincent在2008年的论文中《Extracting and Composing Robust Features》，译成中文就是"提取、编码出具有鲁棒性的特征"。怎么才能使特征很鲁棒呢？就是以一定概率分布（通常使用二项分布）去擦除原始input矩阵，即每个值都随机置0,  这样看起来部分数据的部分特征是丢失了。以这丢失的数据x'去计算y，计算z，并将z与原始x做误差迭代，这样，网络就学习了这个破损（原文叫Corruputed）的数据。
   这个破损的数据是很有用的，原因有二：
   其一，通过与非破损数据训练的对比，破损数据训练出来的Weight噪声比较小。降噪因此得名。原因不难理解，因为擦除的时候不小心把输入噪声给×掉了。
   其二，破损数据一定程度上减轻了训练数据与测试数据的代沟。由于数据的部分被×掉了，因而这破损数据一定程度上比较接近测试数据。（训练、测试肯定有同有异，当然我们要求同舍异）。
