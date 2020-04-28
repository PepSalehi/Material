import os
import sys
import json
import pprint


class Data:

    def __init__(self, dirname):
        self.events = self.read_events(dirname)
        pprint.pprint(self.events)

    def read_events(self, dirname):
        filename = os.path.join(dirname, "event_list.json")
        with open(filename, mode="r", encoding="utf-8") as filehandle:
            try:
                events = json.load(filehandle)
            except ValueError as e:
                sys.exit('try to read file: {1} error: {2}'.format(filename, e))
        return events
