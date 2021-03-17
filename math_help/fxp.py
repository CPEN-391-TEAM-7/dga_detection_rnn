import math

class Fxp(object):

    def __init__(self, obj):
        # 0 is encoded as 48 in utf-8, 1 is encoded as 49
        if isinstance(obj, bytes):
            self.num = [x-48 for x in list(obj)]
        elif isinstance(obj, str):
            self.num = [int(x) for x in list(obj)]
        else:
            self.num = obj
        
    def __mul__(self, other):
        result = [0]*32
        sign = 0
        tmpA = self.num.copy()
        tmpB = other.num.copy()
        if tmpA[0] == 1:
            sign = (sign + 1)%2
            tmpA = self.complement(tmpA)
        if  tmpB[0] == 1:
            sign = (sign + 1)%2
            tmpB = self.complement(tmpB)
        for i, c in enumerate(reversed(tmpA)):
            tmp = [0]*16
            for j, d in enumerate(reversed(tmpB)):
                tmp[j] = c*d
            overflow = 0
            for k, e in enumerate(tmp):
                nxtoverflow = math.floor((result[i+k] + overflow + e)/2)
                result[i+k] = (result[i+k] + overflow + e)%2
                overflow = nxtoverflow
        result.reverse()
        if sign:
            return Fxp([sign]+self.complement(result[9:24]))
        else:
            return Fxp([sign]+result[9:24])

    def __rmul__(self, other):
        result = [0]*32
        sign = 0
        tmpA = self.num.copy()
        tmpB = other.num.copy()
        if tmpA[0] == 1:
            sign = (sign + 1)%2
            tmpA = self.complement(tmpA)
        if  tmpB[0] == 1:
            sign = (sign + 1)%2
            tmpB = self.complement(tmpB)
        for i, c in enumerate(reversed(tmpA)):
            tmp = [0]*16
            for j, d in enumerate(reversed(tmpB)):
                tmp[j] = c*d
            overflow = 0
            for k, e in enumerate(tmp):
                nxtoverflow = math.floor((result[i+k] + overflow + e)/2)
                result[i+k] = (result[i+k] + overflow + e)%2
                overflow = nxtoverflow
        result.reverse()
        if sign:
            return Fxp([sign]+self.complement(result[5:20]))
        else:
            return Fxp([sign]+result[5:20])

    def __add__(self, other):
        result = [0]*16
        tmpA = self.num.copy()
        tmpA.reverse()
        tmpB = other.num.copy()
        tmpB.reverse()
        overflow = 0
        for i in range(len(tmpA)):
            nxtoverflow = math.floor((tmpA[i] + tmpB[i] + overflow) / 2)
            result[i] = (tmpA[i] + tmpB[i] + overflow)%2
            overflow = nxtoverflow
        result.reverse()
        return Fxp(result)

    def __radd__(self, other):
        return self.__add__(other)

    def value(self):
        if self.num == None:
            return None
        else:
            result = 0
            if self.num[0] == 0:
                for i in range(1,16):
                    if self.num[i] == 1:
                        result += math.pow(2,7-i)
            else:
                for i in range(1,16):
                    if self.num[i] == 0:
                        result += math.pow(2,7-i)
                result *= -1
            return result
    
    def complement(self, a):
        for i,c in enumerate(a):
            if c == 1:
                a[i] = 0
            else:
                a[i] = 1
        return a
