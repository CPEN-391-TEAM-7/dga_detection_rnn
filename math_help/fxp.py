class Fxp(object):

    def __init__(self, byteobj):
        # 0 is encoded as 48 in utf-8, 1 is encoded as 49
        self.num = [x-48 for x in list(byteobj)]
        print(self.num)
        
    def __mul__(self, other):
        print('__mul__')
        return other

    def __rmul__(self, other):
        print('__rmul__')
        return other
