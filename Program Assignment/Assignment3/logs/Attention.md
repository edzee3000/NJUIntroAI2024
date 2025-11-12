注意我在原本框架代码上做了一些细微的修改

因为原本框架代码里面把训练数据、模型数据、结果数据全都混杂在同一个logs日志文件夹中
导致我查找对应的路径非常麻烦

因此我将它们都分开了，但是在这里就得注意问题了，我在play.py文件中的__init__函数当中添加了一个默认参数flag_AI=True
在test.py当中需要手动传入flag_AI=False这个参数保证不会自动
```
os.makedirs(self.log_folder, exist_ok=True)
```
不然的话每一次在learn.py以及test.py当中调用```env = AliensEnvPygame(level=0, render=False)```这段代码的话都得在PlayDatas这个目录下面再新创建一个空的目录，特别烦人……不如一劳永逸


另外在play.py文件当中的save_gif函数我也手动添加了一个path默认参数用来手动传递存储gif的路径

大致修改有这些，这个文件当作大作业中的一些实验记录













