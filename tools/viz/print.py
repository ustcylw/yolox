import os, sys
from prettytable import PrettyTable as PTable, ALL
import textwrap













def show_dict(cls=None, title='', **kwargs):
    data = {}
    data.update({k: textwrap.wrap(f'{v}', width=20) for k, v in kwargs.items()})
    if cls is not None:
        data.update({k: v if not isinstance(v, list) else '\n'.join([str(vi) for vi in v]) for k, v in cls.__dict__.items() if not(k.startswith('__') or k.endswith('__'))})
        # data.update({k: v for k, v in cls.__dict__.items() if not(k.startswith('__') or k.endswith('__'))})
    ptb = PTable()
    ptb.title = title
    ptb.border = True
    ptb.max_width = 80
    ptb.preserve_internal_border = True
    ptb.field_names = list(['key', 'value'])
    ptb.add_rows(data.items())
    for field_name in ptb.field_names:
        ptb.align[field_name] = 'c'
        ptb.valign[field_name] = 'm'
    print(ptb.get_string())
    

