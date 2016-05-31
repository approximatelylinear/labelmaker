
import pdb
from pprint import pformat
from models_es import ESClusterDataFrame

def test_get_info():
    m = ESClusterDataFrame(db_name='marlboro')
    info = m.get_info()
    print pformat(info)


def test_get_filters():
    # m = ESClusterDataFrame(db_name='marlboro')
    # info = m.get_info()
    # print pformat(info)
    # pdb.set_trace()
    filters = {
        'date': '2014-01-01',
        'content': 'love',
        'community': ['hot-streak', 'black-blog'],
        'status': 'approved',
        'limit': 100,
        'is_labeled': True,
        'word_count': 5,
    }
    m = ESClusterDataFrame(filters=filters, db_name='marlboro')
    fltrs = m.get_filters()
    print pformat(fltrs)


def test_find_posts():
    # m = ESClusterDataFrame(db_name='marlboro')
    # info = m.get_info()
    # print pformat(info)
    # pdb.set_trace()
    filters = {
        'date': '2014-10-01',
        # 'content': '*',
        'community': ['hot-streak', 'black-blog'],
        'status': 'approved',
        'limit': 100,
        # 'is_labeled': True,
        'word_count': 5,
    }
    m = ESClusterDataFrame(filters=filters, db_name='marlboro')
    m.find_posts()


if __name__ == '__main__':
    # test_get_info()
    # test_get_filters()
    test_find_posts()


