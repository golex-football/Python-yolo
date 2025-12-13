# -*- coding: utf-8 -*-
# Dynamic protobuf message for YoloPacket (matches uploaded yolo_packet.proto)
from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import message_factory

_FILE_NAME = "yolo_packet.proto"
_PACKAGE = "golex.virtualtracking"

def _fdp():
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = _FILE_NAME
    fdp.package = _PACKAGE
    fdp.syntax = "proto3"
    msg = fdp.message_type.add()
    msg.name = "YoloPacket"

    def add(name, number, ftype, label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL):
        fld = msg.field.add()
        fld.name = name
        fld.number = number
        fld.label = label
        fld.type = ftype

    add("frame_id", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    add("raw_frame", 2, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES)
    add("raw_width", 3, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add("raw_height", 4, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add("mask_frame", 5, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES)
    add("mask_width", 6, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add("mask_height", 7, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)

    fld = msg.field.add()
    fld.name = "boxes"
    fld.number = 8
    fld.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    fld.type = descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT
    return fdp

_pool = descriptor_pool.Default()
try:
    _pool.Add(_fdp())
except Exception:
    pass

_desc = _pool.FindMessageTypeByName(f"{_PACKAGE}.YoloPacket")
_factory = message_factory.MessageFactory(_pool)
YoloPacket = _factory.GetPrototype(_desc)
