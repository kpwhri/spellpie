import sqlite3

import logging
import pkg_resources


class DbHandler(object):

    TABLES = ['terms1', 'terms2', 'terms3']

    def __init__(self):
        self.conn = sqlite3.connect(
            pkg_resources.resource_filename('spellpie', 'data/terms.db')
        )

    def load_wordlist(self, table=None):
        cur = self.conn.cursor()
        if table:
            if table.lower() in self.TABLES:
                tables = [table.lower()]
            else:
                logging.warning('Table "{}" not found. Loading all tables.'.format(table))
                tables = self.TABLES
        else:
            tables = self.TABLES
        res = []
        for table in tables:
            cur.execute('SELECT term FROM {}'.format(table))
            res += [x[0] for x in cur.fetchall()]
        return res
