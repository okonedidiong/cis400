import sqlite3
import sys
import os

import re as jet
#import sqlite3 as fuel
import matplotlib.pyplot
import numpy
from collections import Counter as steel



conn = sqlite3.connect('database.sqlite')

c = conn.cursor()

meme = "SELECT lower(body)      \
    FROM May2015                \
    WHERE LENGTH(body) < 40     \
    and LENGTH(body) > 20       \
    and lower(body) LIKE 'jet fuel can''t melt%' \
    LIMIT 100";


beams = []

for illuminati in conn.execute(meme):
    illuminati = jet.sub('[\"\'\\,!\.]', '', (''.join(illuminati)))
    illuminati = (illuminati.split("cant melt"))[1]
    illuminati = illuminati.strip()
    beams.append(illuminati)

bush = steel(beams).most_common()
labels, values = zip(*bush)
indexes = numpy.arange(len(labels))

matplotlib.pyplot.barh(indexes, values)
matplotlib.pyplot.yticks(indexes, labels)
matplotlib.pyplot.tight_layout()
matplotlib.pyplot.savefig('dankmemes.png')

