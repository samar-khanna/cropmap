import os
import functools


def create_dirs(*dirs):
    """
    Creates directories based on paths passed in as arguments.
    """

    def f_mkdir(p):
        if not os.path.isdir(p):
            print(f"Creating directory {p}")
            os.makedirs(p)

    for p in dirs:
        f_mkdir(p)


def rgetattr(obj, attr, *args):
    """
    Recursively gets attributes using chained attributes in attr
    @param obj: Object from which to get attributes
    @param attr: String description of attributes, nested properties separated by '.'
    @param args: Additional args
    @return:
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
