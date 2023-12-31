# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Forge

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Response(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsResponse(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Response()
        x.Init(buf, n + offset)
        return x

    # Response
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Response
    def Outputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from Forge.Tensor import Tensor
            obj = Tensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Response
    def OutputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Response
    def OutputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

def ResponseStart(builder): builder.StartObject(1)
def ResponseAddOutputs(builder, outputs): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(outputs), 0)
def ResponseStartOutputsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def ResponseEnd(builder): return builder.EndObject()
