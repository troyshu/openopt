class OpenOptException(BaseException):
    def __init__(self,  msg):
        self.msg = msg
    def __str__(self):
        return self.msg
        #pass

#def ooassert(cond, msg):
#    assert cond, msg

def oowarn(msg):
    print('OpenOpt Warning: %s' % msg)

def ooerr(msg):
    print('OpenOpt Error: %s' % msg)
    raise OpenOptException(msg)

pwSet = set()
def ooPWarn(msg):
    if msg in pwSet: return
    pwSet.add(msg)
    oowarn(msg)
    
def ooinfo(msg):
    print('OpenOpt info: %s' % msg)

def oohint(msg):
    print('OpenOpt hint: %s' % msg)

def oodisp(msg):
    print(msg)

def oodebugmsg(p,  msg):
    if p.debug: print('OpenOpt debug msg: %s' % msg)
