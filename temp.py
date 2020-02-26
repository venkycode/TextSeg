import inspect


def f1(): f2()


def f2():
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        print('caller name:', calframe)


f1()

