import math
class Fxp(object):

    def __init__(self, obj):
        # 0 is encoded as 48 in utf-8, 1 is encoded as 49
        if isinstance(obj, bytes):
            self.num = [x-48 for x in list(obj)]
        else:
            self.num = obj
        
    def __mul__(self, other):
        result = [0]*32
        for i, c in enumerate(reversed(self.num)):
            tmp = [0]*16
            for j, d in enumerate(reversed(other.num)):
                tmp[j] = c*d
            overflow = 0
            for k, e in enumerate(tmp):
                nxtoverflow = math.floor((result[i+k] + overflow + e)/2)
                result[i+k] = (result[i+k] + overflow + e)%2
                overflow = nxtoverflow
        result.reverse()
        return Fxp([result[0]]+result[5:20])

    def __rmul__(self, other):
        result = [0]*32
        for i, c in enumerate(reversed(self.num)):
            tmp = [0]*16
            for j, d in enumerate(reversed(other.num)):
                tmp[j] = c*d
            overflow = 0
            for k, e in enumerate(tmp):
                nxtoverflow = math.floor((result[i+k] + overflow + e)/2)
                result[i+k] = (result[i+k] + overflow + e)%2
                overflow = nxtoverflow
        result.reverse()
        return Fxp([result[0]]+result[5:20])

    def value(self):
        if self.num == None:
            return None
        else:
            result = 0
            if self.num[0] == 0:
                for i in range(1,16):
                    if self.num[i] == 1:
                        result += math.pow(2,3-i)
            else:
                for i in range(1,16):
                    if self.num[i] == 0:
                        result += math.pow(2,3-i)
                result *= -1
            return result
