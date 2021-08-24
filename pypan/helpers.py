"""Helper functions for PyPan."""

from datetime import datetime as dt
from datetime import timedelta as td

class OneLineProgress():
    """Displays a progress bar in one line.
    Written by Zach Montgomery.
    """
    
    def __init__(self, total, msg='', show_etr=True):
        self.total = total
        self.msg = msg
        self.count = 0
        self.show_etr = show_etr
        self.start = dt.now()
        self.roll_timer = dt.now()
        self.roll_count = -1
        self.roll_delta = 0.2
        self.run_time = 0.0
        self.display()
    
    def increment(self):
        self.count += 1
    
    def decrement(self):
        self.count -= 1
    
    def __str__(self):
        pass
    
    def __len__(self):
        l = len(str(self))
        self.decrement()
        return l
    
    def Set(self, count):
        self.count = count
    
    def display(self):
        rolling = '-\\|/'
        roll_delta = (dt.now()-self.roll_timer).total_seconds()
        
        p2s = False
        if roll_delta >= self.roll_delta or self.roll_count == -1:
            p2s = True
            self.roll_timer = dt.now()
            self.roll_count += 1
            if self.roll_count >= len(rolling):
                self.roll_count = 0
        
        perc = self.count/self.total*100.
        self.increment()
        
        if not p2s and perc < 100.: return
        
        s = '\r' + ' '*(len(self.msg)+50) + '\r'
        s += self.msg + ' '*4
        
        # j = 0
        for i in range(10):
            if perc >= i*10:
                j = i
        
        if perc < 100.:
            s += u'\u039e'*j + rolling[self.roll_count] + '-'*(9-j)
        else:
            s += u'\u039e'*10
        
        # for i in range(1,11):
            # if i*10 <= perc:
                # s += u'\u039e'
            # else:
                # s += '-'
        s += ' '*4 + '{:7.3f}%'.format(perc)
        if not self.show_etr:
            if perc >= 100.: s += '\n'
            print(s, end='')
            return
        
        if perc <= 0:
            etr = '-:--:--.------'
            s += ' '*4 + 'ETR = {}'.format(etr)
        elif perc >= 100.:
            self.run_time = dt.now()-self.start
            s += ' '*4 + 'Run Time {}'.format(self.run_time) + '\n'
        else:
            time = (dt.now()-self.start).total_seconds()
            etr = td(seconds=time / perc * 100. - time)
            s += ' '*4 + 'ETR = {}'.format(etr)
        print(s, end='')
        return