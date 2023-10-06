#!/usr/bin/env python
import os
import sys
import time
import flatbuffers
sys.path.append("forge/")
from Forge.Request import *
from Forge.Response import *
from Forge.Tensor import *
from Forge.DataType import *

FORGE_IN_FD = 3
FORGE_OUT_FD = 4

class ForgeTensor(object):
  def __init__(self, forgetensor):
    self.tensor = forgetensor
    self.ndarray = self.tensor.DataAsNumpy()
  def dims(self):
    return self.tensor.DimsAsNumpy().tolist()
  def as_ndarray(self):
    tdtype = {
      DataType.Bool: np.dtype('bool'),
      DataType.Int8: np.dtype('int8'),
      DataType.Uint8: np.dtype('uint8'),
      DataType.Int16: np.dtype('int16'),
      DataType.Uint16: np.dtype('uint16'),
      DataType.Int32: np.dtype('int32'),
      DataType.Uint32: np.dtype('uint32'),
      DataType.Int64: np.dtype('int64'),
      DataType.Uint64: np.dtype('uint64'),
      DataType.Fp16: np.dtype('float16'),
      DataType.Fp32: np.dtype('float32'),
      DataType.Fp64: np.dtype('float64')
    }[self.tensor.Datatype()]
    arr = np.frombuffer(self.to_bytes(), dtype=tdtype)
    return arr.reshape(self.tensor.DimsAsNumpy())
  def to_bytes(self):
    return self.ndarray.tobytes()
#  def to_string(self, codec='utf-8'):
#    # TODO
#    return self.ndarray.tostring().decode(codec)
  def to_string_list(self):
    strlist = []
    byte = self.to_bytes()
    while len(byte) > 4:
      ssize = int.from_bytes(byte[:4], byteorder='little')
      strlist.append(byte[4:ssize+4].decode('utf-8'))
      byte = byte[ssize+4:]
    return strlist


def buildTensor(builder, tname, tval):
  tic = time.time(); tocn = 0
  dims = []
  if type(tval) == str:
    tval = [tval]
  if type(tval) == np.ndarray:
    tname = builder.CreateString(tname)
    dims = list(tval.shape)
  elif type(tval) == list:
    for item in tval:
      if type(item) != str:
        return
    tname = builder.CreateString(tname)
    dims = [len(tval)]
  else:
    return
  TensorStartDimsVector(builder, len(dims))
  for d in reversed(dims):
    builder.PrependUint64(d)
  tdims = builder.EndVector(len(dims))
   
  if type(tval) == np.ndarray:
    ttype = {
      np.dtype('bool'): DataType.Bool,
      np.dtype('int8'): DataType.Int8,
      np.dtype('uint8'): DataType.Uint8,
      np.dtype('int16'): DataType.Int16,
      np.dtype('uint16'): DataType.Uint16,
      np.dtype('int32'): DataType.Int32,
      np.dtype('uint32'): DataType.Uint32,
      np.dtype('int'): DataType.Int64,
      np.dtype('int64'): DataType.Int64,
      np.dtype('uint64'): DataType.Uint64,
      np.dtype('float'): DataType.Fp64,
      np.dtype('half'): DataType.Fp16,
      np.dtype('float16'): DataType.Fp16,
      np.dtype('float32'): DataType.Fp32,
      np.dtype('float64'): DataType.Fp64,
    }[tval.dtype]
  elif type(tval) == list:
    ttype = DataType.String
   
  if type(tval) == np.ndarray:
    data = tval.tobytes()
  elif type(tval) == list:
    data = b''
    for s in tval:
      sbytes = s.encode('utf-8')
      ssize = len(sbytes).to_bytes(4, byteorder='little')
      data = data + ssize + sbytes
  
  tdata = builder.CreateByteVector(data)
  TensorStart(builder)
  TensorAddName(builder, tname)
  TensorAddDims(builder, tdims)
  TensorAddDatatype(builder, ttype)
  TensorAddData(builder, tdata)
  return TensorEnd(builder)
  
def toInputBuf(inputs):
  builder = flatbuffers.Builder(1024)
  magic = builder.CreateString('forge:0.1')

  inputlist = []
  for k, v in inputs.items():
    inputlist.append(buildTensor(builder, k, v))

  RequestStartInputsVector(builder, len(inputlist))
  for ix in reversed(inputlist):
    builder.PrependUOffsetTRelative(ix)
  ts = builder.EndVector(len(inputlist))

  RequestStart(builder)
  RequestAddMagic(builder, magic)
  RequestAddInputs(builder, ts)
  rs = RequestEnd(builder)

  builder.Finish(rs)
  buf = builder.Output()
  return buf

def fromInputBuf(buf):
  return Request.GetRootAsRequest(buf, 0)

def toOutputBuf(outputs):
  builder = flatbuffers.Builder(1024)
  outputlist = []
  for k, v in outputs.items():
    outputlist.append(buildTensor(builder, k, v))

  ResponseStartOutputsVector(builder, len(outputlist))
  for ox in reversed(outputlist):
    builder.PrependUOffsetTRelative(ox)
  ts = builder.EndVector(len(outputlist))

  ResponseStart(builder)
  ResponseAddOutputs(builder, ts)
  rs = ResponseEnd(builder)

  builder.Finish(rs)
  buf = builder.Output()
  return buf

def fromOutputBuf(buf):
  return Response.GetRootAsResponse(buf, 0)


def run(handler):
  while True:
    ss = os.read(FORGE_IN_FD, 8)
    if len(ss) < 8: break
    size = int.from_bytes(ss, byteorder='little')
    if not size: continue
    buf = os.read(FORGE_IN_FD, size)
    req = Request.GetRootAsRequest(buf, 0)
    requests = {}
    for i in range(req.InputsLength()):
      itensor = req.Inputs(i)
      requests[itensor.Name().decode('utf-8')] = ForgeTensor(itensor)
    responses = handler(requests)
    reslist = []
    builder = flatbuffers.Builder(1024)
    for name, val in responses.items():
      reslist.append(buildTensor(builder, name, val))
    ResponseStartOutputsVector(builder, len(reslist))
    for o in reversed(reslist):
      builder.PrependUOffsetTRelative(o)
    otlist = builder.EndVector(len(reslist))

    ResponseStart(builder)
    ResponseAddOutputs(builder, otlist)
    res = ResponseEnd(builder)

    builder.Finish(res)
    buf = builder.Output()
    bufsize = len(buf).to_bytes(8, byteorder='little')
    os.write(FORGE_OUT_FD, bufsize)
    os.write(FORGE_OUT_FD, buf)
