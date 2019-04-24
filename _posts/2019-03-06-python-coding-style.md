# 如何编写高质量的Pythonic风格代码？

我知道有些新人肯定不了解Pythonic是什么，也许在某些论坛看到过这个词语。其实，它的意思很简单。这是Python的开发者用来表示代码风格的名词。它是在Python开发过程中指定的一种指南，一种习惯。宗旨是 直观、简洁、易读

1、不用害怕长变量名
长一点的变量名，有时候是为了让程序更容易理解和阅读。并且，有的编辑器已经支持自动提示，所以不用太担心敲键盘太多的烦恼。
比如: user_info 就是比 ui 的可读性高很多:
user_info = {'name':'xiyouMc','age':'保密','address':'Hangzhou'}
2、避免使用容易混淆的名称
尽量不要使用 内建 的函数名来表示其他含义的名称。比如 list、dict等。不要使用 o(字符O的小写，很容易被当做数字0)，1(字母 L 的小写，也容易和数字 1 混淆)
其次，变量名最好和你要解决的问题联系起来。
3、尽量不要使用大小写来区分不同的对象
比如 b是一个树脂类型的变量，但 A 是 String 类型，虽然在编码过程中容易区分这两者的含义，但是没啥卵用，它并不会给其他阅读代码的人带来福利。反而，带来的呕吐的感觉。
4、其次，最重要的一点是，多看源码，学习别人的风格
Github 上有数不胜数的优秀代码，比如web框架里面有名的Flask、Requests，还有爬虫界的Scrapy，这些都是经典中的经典，并且都是比较好的理解pythonic代码风格精髓的例子。
5、最后，你实在是懒得不想关注这些，只想写代码，那么。。。
我推荐一个神器，在你写完代码之后，执行这个神器就可以看到检测代码风格后的结果。
PEP8，全称，"Python Enhancement Proposal #8"，它列举除了很多对代码的布局、注释、命名的要求。
pip install -U pep8 #来安装 pep8
然后用它来检测代码：
➜ /Users/xiyoumc >pep8 --first pornHubSpider.py
pornHubSpider.py:1:1: E265 block comment should start with '# '
pornHubSpider.py:19:43: E124 closing bracket does not match visual indentation
pornHubSpider.py:22:16: E251 unexpected spaces around keyword / parameter equals
pornHubSpider.py:53:5: E301 expected 1 blank line, found 0
pornHubSpider.py:71:22: W503 line break before binary operator

同时，如果对pep8感兴趣的话，可以留言，我可以开个系列来讲解 PEP8里面的变量、函数、类、木块和包，这样就会更加容易的理解Pythonic风格。
最后，如若我写的对大家有点帮助，那么关注公众号 DeveloperPython，你将会收到关于Python技术第一时间的推送。

编写Pythonic代码
避免劣化代码

    避免只用大小写来区分不同的对象；
    避免使用容易引起混淆的名称，变量名应与所解决的问题域一致；
    不要害怕过长的变量名；

代码中添加适当注释

    行注释仅注释复杂的操作、算法，难理解的技巧，或不够一目了然的代码；
    注释和代码要隔开一定的距离，无论是行注释还是块注释；
    给外部可访问的函数和方法（无论是否简单）添加文档注释，注释要清楚地描述方法的功能，并对参数，返回值，以及可能发生的异常进行说明，使得外部调用的人仅看docstring就能正确使用；
    推荐在文件头中包含copyright申明，模块描述等；
    注释应该是用来解释代码的功能，原因，及想法的，不该对代码本身进行解释；
    对不再需要的代码应该将其删除，而不是将其注释掉；

适当添加空行使代码布局更为优雅、合理

    在一组代码表达完一个完整的思路之后，应该用空白行进行间隔，推荐在函数定义或者类定义之间空两行，在类定义与第一个方法之间，或需要进行语义分隔的地方空一行，空行是在不隔断代码之间的内在联系的基础上插入的；
    尽量保证上下文语义的易理解性，一般是调用者在上，被调用者在下；
    避免过长的代码行，每行最好不要超过80字符；
    不要为了保持水平对齐而使用多余的空格；

编写函数的几个原则

    函数设计要尽量短小，嵌套层次不宜过深；
    函数申明应做到合理、简单、易于使用，函数名应能正确反映函数大体功能，参数设计应简洁明了，参数个数不宜过多；
    函数参数设计应考虑向下兼容；
    一个函数只做一件事，尽量保证函数语句粒度的一致性；

将常量集中到一个文件

Python没有提供定义常量的直接方式，一般有两种方法来使用常量；

    通过命名风格来提醒使用者该变量代表的意义为常量，如常量名所有字母大写，用下划线连接各个单词，如MAX_NUMBER，TOTLE等；
    通过自定义的类实现常量功能，常量要求符合两点，一是命名必须全部为大写字母，二是值一旦绑定便不可再修改；
```
class _const:

    class ConstError(TypeError): pass
    class ConstCaseError(ConstError): pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            rasie self.ConstError, "Can't change const.%s" % name
        if not name.isupper():
            raise self.ConstCaseError, "const name '%s' is not all uppercase" % name

        self.__dict__[name] = value

import sys
sys.modules[__name__] = _const()
```


# ==============================================================================

基本遵从 PEP 准则

…… 但是，命名和单行长度更灵活。

PEP8 涵盖了诸如空格、函数/类/方法之间的换行、import、对已弃用功能的警告之类的寻常东西，大都不错。

应用这些准则的最佳工具是 flake8，还可以用来发现一些愚蠢的语法错误。

PEP8 原本只是一组指导原则，不必严格甚至虔诚地信奉。一定记得阅读 PEP8 「愚蠢的一致性就是小人物的小妖精」一节。若要进一步了解，可以听一下 Raymond Hettinger 的精彩演讲，「超越 PEP8」。

唯一引起过多争议的准则事关单行长度和命名。要调整起来也不难。
灵活的单行长度

若是厌烦 flake8 死板的单行长度不得超过 79 个字符的限制，完全可以忽略或修改这一准则。这仍然不失为一条不错的经验法则，就像英语中句子不能超过 50 个单词，段落不能超过 10 个句子之类的规则一样。这是 flake8 配置文件 的链接，可以看到 max-line-length配置选项。值得注意的是，可以给要忽略 flake8 检查的那一行加上 # noqa 注释，但是请勿滥用。

尽管如此，超过九成的代码行都不应该超过 79 个字符，原因很简单，「扁平胜于嵌套」。如果函数每一行都超出了 79 个字符，肯定有别的东西出错了，这时要看看代码而不是 flake8 配置。
一致的命名

关于命名，遵循几条简单的准则就可以避免众多足以影响整个小组的麻烦。
推荐的命名规则

下面这些准则大多改编自 Pacoo 小组。

    类名：驼峰式 和首字母缩略词：HTTPWriter 优于 HttpWriter。
    变量名：lower_with_underscores。
    方法名和函数名：lower_with_underscores。
    模块名：lower_with_underscores.py。（但是不带下划线的名字更好！）
    常量名：UPPER_WITH_UNDERSCORES。
    预编译的正则表达式：name_re。

通常都应该遵循这些准则，除非要参照其他工具的命名规范，比如数据库 schema 或者消息格式。

还可以用 驼峰式 给类似类却不是类的东西命名。使用 驼峰式 的主要好处在于让人们以「全局名词」来关注某个东西，而不是看作局部标记或动词。值得注意的是，Python 给 True，False 和 None 这些根本不是类的东西命名也是用 驼峰式。
不要用前缀后缀

…… 比如 _prefix 或 suffix_ 。函数和方法名可以用 _prefix 标记来暗示其是「私有的」，但是最好只在编写预期会广泛使用的 API 以及用 _prefix 标记来隐藏信息的时候谨慎使用。

PEP8 建议使用结尾的下划线来避免与内置关键字重名，比如：
```python
sum_ sum(some_long_list)
print(sum_)
```
临时这样用也可以，不过最好还是选一个别的名字。

用 __mangled 这种双下划线前缀给类/实例/方法命名的情况非常少，这实际上涉及特殊的名字修饰，非常罕见。不要起 __dunder__ 这种格式的名字，除非要实现 Python 标准协议，比如 __len__；这是为 Python 内部协议保留的命名空间，不应该在其中增加自定义的东西。
不要用单字符名字

（不过）一些常见的单字符名字可以接受。

在 lambda 表达式中，单参数函数可以命名为 x 。比如：`encode = lambda x: x.encode("utf-8", "ignore")`

使用 self 及类似的惯例

应该：

    永远将方法的第一个变量命名为 self
    永远将 @classmethod 的第一个参数命名为 cls
    永远在变量参数列表中使用 *args 和 **kwargs

不要在这些地方吹毛求疵

不遵循如下准则没有什么好处，干脆照它说的做。
永远继承自 object 并使用新式类
```python

# bad
class JSONWriter:
    pass
 
# good
class JSONWriter(object):
    pass
```
对于 Python 2 来说遵循这条准则很重要。不过由于 Python 3 所有的类都隐式继承自 object，这条准则就没有必要了。
不要在类中重复使用实例标记
```

# bad
class JSONWriter(object):
    handler = None
    def __init__(self, handler):
        self.handler = handler
 
# good
class JSONWriter(object):
    def __init__(self, handler):
        self.handler = handler

```

用 isinstance(obj, cls), 不要用 type(obj) == cls

因为 isinstance 涵盖更多情形，包括子类和抽象基类。同时，不要过多使用 isinstance，因为通常应该使用鸭子类型！
用 with 处理文件和锁

with 语句能够巧妙地关闭文件并释放锁，哪怕是在触发异常的情况下。所以：
```

# bad
somefile = open("somefile.txt", "w")
somefile.write("sometext")
return
 
# good
with open("somefile.txt", "w") as somefile:
    somefile.write("sometext")
return

```
和 None 相比较要用 is

None 值是一个单例，但是检查 None 的时候，实际上很少真的要在 左值上调用 __eq__。所以：
```

# bad
if item == None:
    continue
 
# good
if item is None:
   continue

```

好的写法不仅执行更快，而且更准确。使用 == 并不会更简洁，所以请记住本规则！
不要修改 sys.path

通过 sys.path.insert(0, "../") 等操作来控制 Python 的导入方法或许让人心动，但是要坚决避免这样做。

Python 有一套有几分复杂，却易于理解的模块路径解决方法。可以通过 PYTHONPATH 或诸如 setup.py develop 的技巧来调整 Python 导入模块的方法。还可以用 -m 运行 Python 得到需要的效果，比如使用 python -m mypkg.mymodule 而不是 python mypkg/mymodule.py。代码能否正常运行不应依赖于当前执行 Python 的工作路径。David Beazley 用 PDF 幻灯片再一次扭转了大家的看法，值得一读，“Modules and Packages: Live and Let Die!”。
尽量不要自定义异常类型

…… 如果一定要，也不要创建太多。

### 短文档字符串应是名副其实的单行句子
```

# bad
def reverse_sort(items):
    """
    sort items in reverse order
    """
 
# good
def reverse_sort(items):
    """Sort items in reverse order."""

```
把三引号 """ 放在同一行，首字母大写，以句号结尾。四行精简到两行，__doc__ 属性没有糟糕的换行，最吹毛求疵的人也会满意的！
文档字符串使用 reST

标准库和大多数开源项目皆是如此。Sphinx 提供支持，开箱即用。赶紧试试吧！Python requests 模块由此取得了极佳的效果。看看requests.api 模块的例子。
删除结尾空格

最挑剔也不过如此了吧，可是若做不到这一点，有些人可能会被逼疯。不乏能自动搞定这一切的编辑器；这是我用 vim 的实现。
文档字符串要写好

下面是在函数文档字符串中使用 Sphinx 风格的 reST 的快速参考：
```python
def get(url, qsargs=None, timeout=5.0):
    """Send an HTTP GET request.
 
    :param url: URL for the new request.
    :type url: str
    :param qsargs: Converted to query string arguments.
    :type qsargs: dict
    :param timeout: In seconds.
    :rtype: mymodule.Response
    """
return request('get', url, qsargs=qsargs, timeout=timeout)

>>> 不要为文档而写文档。写文档字符串要这样思考：
>>> 好名字 + 显式指明默认值 优于 罗嗦的文档 + 类型的详细说明
```

也就是说，上例中没有必要说 timeout 是 float，默认值 5.0，显然是 float。在文档中指出其语义是「秒」更有用，就是说 5.0 意思是 5 秒钟。同时调用方不知道 qsargs 应该是什么，所以用 type 注释给出提示，调用方也无从知道函数的预期返回值是什么，所以 rtype注释是合适的。

最后一点。吉多·范罗苏姆曾说过，他对 Python 的主要领悟是「读代码比写代码频率更高」。直接结论就是有些文档有用，更多的文档有害。

基本上只需要给预计会频繁重用的函数写文档。如果给内部模块的每一个函数都写上文档，最后只能得到更加难以维护的模块，因为重构代码之时文档也要重构。不要「船货崇拜」文档字符串，更不要用工具自动生成文档。
范式和模式
是函数还是类

通常应该用函数而不是类。函数和模块是 Python 代码重用的基本单元，还是最灵活的形式。类是一些 Python 功能的「升级路径」，比如实现容器，代理，描述符，类型系统等等。但是通常函数都是更好的选择。

或许有人喜欢为了更好地组织代码而将关联的函数归在类中。但这是错的。关联的函数应该归在模块中。

尽管有时可以把类当作「小型命名空间」（比如用 @staticmethod）比较有用，一组方法更应该对同一个对象的内部操作有所贡献，而不仅仅作为行为分组。

与其创建 TimeHelper 类，带有一堆不得不引入子类才能使用的方法，永远不如直接为时间相关的函数创建 lib.time 模块。类会增殖出更多的类，会增加复杂性，降低可读性。
生成器和迭代器

生成器和迭代器是 Python 中最强大的特性 —— 应该掌握迭代器协议，yield 关键字和生成器表达式。

生成器不仅仅对要在大型数据流上反复调用的函数十分重要，而且可以让自定义迭代器更加简单，从而简化了代码。将代码重构为生成器通常可以在使得代码在更多场景下复用，从而简化代码。

Fluent Python 的作者 Lucinao Ramalho 通过 30 分钟的演讲，「迭代器和生成器： Python 之道」，给出了一个出色的，快节奏的概述。Python Essential Reference 和 Python Cookbook 的作者 David Beazley 有个深奥的三小时视频教程，「生成器：最后的前沿」，给出了令人满足的生成器用例的详细阐述。因为应用广泛，掌握这一主题非常有必要。
声明式还是命令式

声明式编程优于命令式编程。代码应该表明你想要做什么，而不是描述如何去做。Python 的函数式编程概览介绍了一些不错的细节并给出了高效使用该风格的例子。

使用轻量级的数据结构更好，比如 列表，字典，元组和集合。将数据展开，编写代码对其进行转换，永远要优于重复调用转换函数/方法来构建数据。


「纯」函数和迭代器更好

这是个从函数式编程社区借来的概念。这种函数和迭代器亦被描述为「无副作用」，「引用透明」或者有「不可变输入/输出」。
一个简单的例子，要避免这种代码：
```

# bad
def dedupe(items):
    """Remove dupes in-place, return items and # of dupes."""
    seen = set()
    dupe_positions = []
    for i, item in enumerate(items):
        if item in seen:
            dupe_positions.append(i)
        else:
            seen.add(item)
    num_dupes = len(dupe_positions)
    for idx in reversed(dupe_positions):
        items.pop(idx)
    return items, num_dupes


# good
def dedupe(items):
    """Return deduped items and # of dupes."""
    deduped = set(items)
    num_dupes = len(items) - len(deduped)
    return deduped, num_dupes

```
这是个惊人的例子。函数不仅更加纯粹，而且更加精简了。不仅更加精简，而且更好。这里的纯粹是说 assert dedupe(items) == dedupe(items) 在「好」版本中恒为真。在「坏」版本中， num_dupes 在第二次调用时恒为 0，这会在使用时导致难以理解的错误。

这个例子也阐明了命令式风格和声明式风格的区别：改写后的函数读起来更像是对需要的东西的描述，而不是构建需要的东西的一系列操作。
简单的参数和返回值类型更好

函数应该尽可能处理数据，而不是自定义的对象。简单的参数类型更好，比如字典，集合，元组，列表，int，float 和 bool。从这些扩展到标准库类型，比如 datetime, timedelta, array, Decimal 以及 Future。只有在真的必要时才使用自定义类型。

判断函数是否足够精简有个不错的经验法则，问自己参数和返回值是否总是可以 JSON 序列化。结果证明这个经验法则相当有用：可以 JSON 序列化通常是函数在并行计算时可用的先决条件。但是，就本文档而言，主要的好处在于：可读性，可测试性以及总体的函数简单性。
避免「传统的」面向对象编程

在「传统的面向对象编程语言」中，比如 Java 和 C++ ，代码重用是通过类的继承和多态或者语言声称的类似机制实现的。对 Python 而言，尽管可以使用子类和基于类的多态，事实上在地道的 Python 程序中这些功能极少使用。

通过模块和函数实现代码重用更为普遍，通过鸭子类型实现动态调度更为常见。如果发现自己通过超类实现代码重用，停下来，重新思考。如果发现自己大量使用多态，考虑一下是否用 Python 的 dunder 协议或者鸭子类型策略会更好。

看一下另一个不错的 Python 演讲，一位 Python 核心贡献者的 「不要再写类了」。演讲者建议，如果构建的类只有一个命名像一个类的方法（比如 Runnable.run()），那么实际上只是用函数模拟了一个类，这时候应该停下来。因为在 Python 中，函数是「最高级的」类型，没有理由这样做。
### Mixin 有时也没问题

可以使用 Mixin 实现基于类的代码重用，同时不需要走极端使用类型层次。但是不要滥用。「扁平胜于嵌套」也适用于类型层次，所以应该避免仅仅为了分解行为而引入不必要的必须层次的一层。

Mixin 实际上不是 Python 的特性，多亏了 Python 支持多重继承。可以创建基类将功能「注入」到子类中，而不必构成类型层次的「重要」组成部分，只需要将基类列入 bases列表中的第一个元素。
```python	
class APIHandler(AuthMixin, RequestHandler):
    """Handle HTTP/JSON requests with security."""
```
要考虑顺序，同时不妨记住：bases 自底向上构成层次结构。这里可读性的好处在于关于这个类所需要知道的一切都包含在类定义本身：「它混入了权限行为，是专门定制的 Tornado RequestHandler。」


















### 小心框架

Python 有大量的 web，数据库等框架。Python 语言的一大乐趣在于创建自定义框架相当简单。使用开源框架时，应该注意不要将自己的「核心代码」和框架本身结合得过于紧密。

考虑为自己的代码创建框架的时候应当慎之又慎。标准库有很多内置的东西，PyPI 有的就更多了，而且通常你不会需要它。
### 尊重元编程

Python 通过一些特性来支持 「元编程」，包括修饰器，上下文管理器，描述符，import 钩子，元类和抽象语法树（AST）转换。

应该能够自如地使用并理解这些特性，作为 Python 的核心组成部分这些特性有着充分地支持。但是应当意识到使用这些特性之时，也引入了复杂的失败场景。所以，要把为自己的代码创建元编程工具与决定「创建自定义框架」同等对待。它们意味着同一件事情。真要这么做的时候，把元编程工具写成独立的模块，写好文档！
不要害怕 「双下划线」方法

许多人将 Python 的元编程工具和其对 「双下划线」或「dunder」方法（比如 __getattr__）的支持混为一谈。

正如博文 ——「Python 双下划线，双倍惊喜」—— 所言，双下划线没有什么「特殊的」。它们只不过是 Python 核心开发人员为所有的 Python 内部协议所起的轻量级命名空间。毕竟，__init__ 也是双下划线，没有什么神奇的。

的确，有些双下划线比其他的会导致更多令人困惑的结果，比如，没有很好的理由就重载操作符通常不是什么好主意。但是它们中也有许多，比如 __repr__，__str__，__len__ 以及 __call__ 是 Python 语言的完整组成部分，应该在地道的 Python 代码中充分利用。不要回避！
代码风格小禅理

作为一位核心 Python 开发者，Barry Warsaw 曾经说过「Python 之禅」（PEP 20）被用作 Python 代码风格指南使他沮丧，因为这本是为 Python 的内部设计所作的一首小诗。也就是语言的设计以及语言的实现本身。不过必须承认，PEP 20 中有些行可以当作相当不错的地道 Python 代码指南，所以我们就把它加上了。
美胜于丑

这一条有些主观，实际上等同于问：接手代码的人会被折服还是感到失望？如果接手的人就是三年后的你呢？
显胜于隐

有时为了重构以去除重复的代码，会有一点抽象。应该能够将代码翻译成显现的英语并且大致了解它是干什么的。不应该有太多的「神奇之处」。
扁平胜于嵌套

这一条很好理解。最好的函数没有嵌套，既不用循环也不用 if 语句。第二好的函数只有一层嵌套。如果有两层及以上的嵌套，最好重构成更小的函数。

可读性确实重要

不要害怕用 # 添加行注释。也不要滥用或者写过多文档。一点点逐行解释，通常很有帮助。不要害怕使用稍微长一些的名字，因为描述性更好。将 「response」写成「rsp」没有任何好处。使用 doctest 风格的例子在文档字符串中详细说明边界情况。简洁至上！
错误不应被放过

单独的except: pass 子句危害最大。永远不要使用。制止所有的异常实在危险。将异常处理限制在一行代码，并且总是将 except 处理器限制在特定的类型下。除此之外，可以自如地使用 logging 模块和 log.exception(…)。
如果实现难以解释，那就是个坏主意

这虽是通用软件工程原则，但是特别适用于 Python 代码。大多数 Python 函数和对象都可以有易于解释的实现。如果难以解释，很可能是一个坏主意。通常可以通过「分而治之」将一个难以解释的函数重写成易于解释的函数，也就是分割成多个函数。
测试是个好主意

好吧，我们篡改了「Python 之禅」中的这一行，原文中「命名空间」才是个绝妙的好主意。

不过说正经的，优雅却没有测试的代码简直比哪怕是最丑陋却测试过的代码还要差劲。至少丑陋的代码可以重构成优雅的，但是优雅的代码却不能重构为可以证明是正确的代码，至少不写测试是做不到的！所以，写测试吧！拜托！
平分秋色

我们把宁愿不去解决的争论放在这个部分。不要因为这些重写别人的代码。这里的东西可以自由地交替使用。
str.format 还是重载格式化操作符 %

str.format 更健壮，然而 % 使用 "%s %s" printf 风格的字符串更加简洁。两者会永远共存。
但是选择哪一个都没有问题。我们没有强制规定。
if item 还是 if item is not None

本条和之前的对于 None 是用 == 还是 is 没有关系。这里我们实际上利用了 Python 的 「真实性规则」来处理 if item，这实际上是「item 不是 None 或者空字符串」的简写。

Python 中的真实性有些复杂。显然第二种写法对于某些错误而言更安全。但是第一种写法在 Python 代码中非常常见，而且更短。对此我们并没有强制规定。
隐式多行字符串还是三引号 """

Python 编译器在语法分析时，如果多个字符串之间没有东西，会自动将其拼接为一个字符串。比如：


标准工具和项目结构

我们选择了一些「最佳组合」工具，以及像样的 Python 项目会用到的最小初始结构。
标准库

    import datetime as dt: 永远像这样导入 datetime
    dt.datetime.utcnow(): 优于 .now(), 后者使用的是当地时间
    import json: 数据交换的标准
    from collections import namedtuple: 用来做轻量级数据类型
    from collections import defaultdict: 用来计数/分组
    from collections import deque: 快速的双向队列
    from itertools import groupby, chain: 为了声明式风格
    from functools import wraps: 用来编写合乎规范的装饰器
    argparse: 为了构建「健壮的」命令行工具
    fileinput: 用来快速构建易于和 UNIX 管道结合使用的工具
    log = logging.getLogger(__name__): 足够好用的日志
    from __future__ import absolute_import: 修复导入别名

常见第三方库

    python-dateutil 用来解析时间和日历
    pytz 用来处理时区
    tldextract 为了更好地处理 URL
    msgpack-python 比 JSON 更加紧凑地编码
    futures 为了 Future/pool 并发原语
    docopt 用来快速编写一次性命令行工具
    py.test 用来做单元测试，与 mock 和 hypothesis 结合使用

本地开发项目框架

对所有的 Python 包和库而言：

    根目录下不要有 __init__.py：目录名用作包名！
    mypackage/__init__.py 优于 src/mypackage/__init__.py
    mypackage/lib/__init__.py 优于 lib/__init__.py
    mypackage/settings.py 优于 settings.py
    README.rst 用来给新手描述本项目；使用 rst
    setup.py 用来构建简单工具，比如 setup.py develop
    requirements.txt 是为 pip 准备的包依赖环境
    dev-requirements.txt 是为 tests/local 准备的额外的依赖环境
    Makefile 用来简化 (!!!) build/lint/test/run 步骤

另外，永远记得详细说明包依赖环境。
灵感来源

下面这些链接或许可以给你启发，有助于书写具有良好风格和品味的 Python 代码。

    Python 标准库中的 [Counter class](https://github.com/python/cpython/blob/57b569d8af2b3263c5d9e6d75fb308f89ea17ac6/Lib/collections/__init__.py#L446-L841) 代码, 作者是 Raymond Hettinger
    [rq.queue](https://github.com/rq/rq/blob/master/rq/queue.py) 模块, 原作者是 Vincent Driessen
    本文作者还写过 这篇关于 [“Pythonic”](http://amontalenti.com/2010/11/03/pythonic-means-idiomatic-and-tasteful) 代码的博文

出发吧，写更具 Python 风格的代码！












