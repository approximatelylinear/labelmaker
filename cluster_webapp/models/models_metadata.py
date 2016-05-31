
#   Stdlib
import os
import datetime
import regex
import pdb
import json
import copy
import traceback
from pprint import pformat

#   3rd party
import regex
import yaml
import requests
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Text, Index, ForeignKey
import dateutil
from dateutil.parser import parse as date_parse
from termcolor import colored

THIS_DIR = os.path.realpath(os.path.dirname('__file__'))

Base = declarative_base()

def load_fixtures(fname):
    with open(os.path.join(THIS_DIR, fname), 'rbU') as f:
        docs = yaml.load_all(f)
        for doc in docs:
            yield doc


def get_timestamp():
    return datetime.datetime.utcnow()


def sql_2_dict(obj):
    """
    Converts a sqlalchemy ORM object to a dictionary.
    """
    return dict([
        (k, getattr(obj, k))
            for k in dir(obj)
                if k == '_id' or not (
                    k.startswith('_') or
                    k in ['metadata'] or
                    callable(getattr(obj, k))
                )
    ])


class Dataset(Base):
    __tablename__ = 'datasets'
    id = Column(String, primary_key=True)
    name = Column(String)
    name_db = Column(String)
    manager = Column(String) # Model for each entry.
    brand = Column(String)
    source = Column(String)
    path = Column(String) # TBD: add convention for the host/port.
    type = Column(String) # Tweet, Post, ugc_pm
    url_fts = Column(String, nullable=True)
    date_create = Column(DateTime)
    timestamp = Column(DateTime)
    parent = Column(String, ForeignKey("datasets.id"), nullable=True)
    user = Column(String, ForeignKey("users.id"))

    @staticmethod
    def create_id(item):
        return '_'.join([item['brand'], item['source'], item['name']])


class User(Base):
    __tablename__ = 'users'
    id = Column(String, primary_key=True)
    name = Column(String)


def create_all(engine):
    Base.metadata.create_all(engine)

def create_sessionmaker(engine):
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    return Session

ENGINE = create_engine('sqlite:///db_metadata.sqlite3',)
SESSION = create_sessionmaker(ENGINE)


def list_datasets():
    session = SESSION()
    objs = session.query(Dataset).all()
    items = [ (obj.name, obj.id) for obj in objs ]
    return items


def setup_datasets(session):
    items = load_fixtures('fixtures_dataset.yaml')
    items = list(items)
    print pformat(items)
    for item in items:
        #   id: brand + source + name
        item['id'] = Dataset.create_id(item)
        item['date_create'] = date_parse(item['date_create'])
    session.add_all([Dataset(**item) for item in items])
    session.commit()
    items_db = session.query(Dataset).all()
    print pformat(items_db)


def setup():
    print "Setting up metadata database..."
    create_all(ENGINE)
    session = SESSION()
    setup_datasets(session)


if __name__ == '__main__':
    setup()
    items = list_datasets()
    print pformat(items)

