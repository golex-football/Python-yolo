# -*- coding: utf-8 -*-
# Dynamic protobuf message for InputFrame (matches uploaded frame_message.proto)
from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import message_factory

_FILE_NAME = "frame_message.proto"
_PACKAGE = "golex.virtualtracking"

def _fdp():
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = _FILE_NAME
    fdp.package = _PACKAGE
    fdp.syntax = "proto3"
    msg = fdp.message_type.add()
    msg.name = "InputFrame"

    def add(name, number, ftype, label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL):
        fld = msg.field.add()
        fld.name = name
        fld.number = number
        fld.label = label
        fld.type = ftype

    add("schema", 1, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    add("frame_id", 2, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    add("timestamp", 3, descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE)
    add("width", 4, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add("height", 5, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add("pixel_format", 6, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)
    add("frame_data", 7, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES)
    return fdp

_pool = descriptor_pool.Default()
try:
    _pool.Add(_fdp())
except Exception:
    pass

_desc = _pool.FindMessageTypeByName(f"{_PACKAGE}.InputFrame")
_factory = message_factory.MessageFactory(_pool)
InputFrame = _factory.GetPrototype(_desc)
