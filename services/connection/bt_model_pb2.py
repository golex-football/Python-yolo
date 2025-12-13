# -*- coding: utf-8 -*-
"""
Dynamic protobuf messages matching BroadTrack src/model/protobuf/*.proto

This avoids requiring protoc/grpc_tools at install time.
Package: golex.virtualtracking.model
Messages:
  - RawFrame
  - MaskFrame
  - Box
  - BoxFrame
  - YoloPacket
  - CapturePacket
Enum:
  - PixelFormats (PF_GRAY8=0, PF_BGR24=1)
"""
from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import message_factory

_PACKAGE = "golex.virtualtracking.model"

def _add_common(pool: descriptor_pool.DescriptorPool):
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "model/protobuf/common.proto"
    fdp.package = _PACKAGE
    fdp.syntax = "proto3"

    enum = fdp.enum_type.add()
    enum.name = "PixelFormats"
    v0 = enum.value.add(); v0.name = "PF_GRAY8"; v0.number = 0
    v1 = enum.value.add(); v1.name = "PF_BGR24"; v1.number = 1
    try:
        pool.Add(fdp)
    except Exception:
        pass

def _add_raw_frame(pool: descriptor_pool.DescriptorPool):
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "model/protobuf/raw-frame.proto"
    fdp.package = _PACKAGE
    fdp.syntax = "proto3"
    fdp.dependency.append("model/protobuf/common.proto")

    msg = fdp.message_type.add()
    msg.name = "RawFrame"
    def add(name, number, ftype, type_name=None):
        fld = msg.field.add()
        fld.name = name
        fld.number = number
        fld.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        fld.type = ftype
        if type_name:
            fld.type_name = type_name

    add("timestamp", 1, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    add("width", 2, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add("height", 3, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add("pixel_format", 4, descriptor_pb2.FieldDescriptorProto.TYPE_ENUM, f"{_PACKAGE}.PixelFormats")
    add("frame_data", 5, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES)

    try:
        pool.Add(fdp)
    except Exception:
        pass

def _add_mask_frame(pool: descriptor_pool.DescriptorPool):
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "model/protobuf/mask-frame.proto"
    fdp.package = _PACKAGE
    fdp.syntax = "proto3"
    fdp.dependency.append("model/protobuf/common.proto")

    msg = fdp.message_type.add()
    msg.name = "MaskFrame"
    def add(name, number, ftype, type_name=None):
        fld = msg.field.add()
        fld.name = name
        fld.number = number
        fld.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        fld.type = ftype
        if type_name:
            fld.type_name = type_name

    add("timestamp", 1, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    add("width", 2, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add("height", 3, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add("pixel_format", 4, descriptor_pb2.FieldDescriptorProto.TYPE_ENUM, f"{_PACKAGE}.PixelFormats")
    add("frame_data", 5, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES)

    try:
        pool.Add(fdp)
    except Exception:
        pass

def _add_box(pool: descriptor_pool.DescriptorPool):
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "model/protobuf/box.proto"
    fdp.package = _PACKAGE
    fdp.syntax = "proto3"

    msg = fdp.message_type.add()
    msg.name = "Box"
    def add(name, number, ftype):
        fld = msg.field.add()
        fld.name = name
        fld.number = number
        fld.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        fld.type = ftype

    add("x", 1, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add("y", 2, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add("width", 3, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add("height", 4, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add("score", 5, descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT)
    add("class_id", 6, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)

    try:
        pool.Add(fdp)
    except Exception:
        pass

def _add_box_frame(pool: descriptor_pool.DescriptorPool):
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "model/protobuf/box-frame.proto"
    fdp.package = _PACKAGE
    fdp.syntax = "proto3"
    fdp.dependency.append("model/protobuf/box.proto")

    msg = fdp.message_type.add()
    msg.name = "BoxFrame"
    def add(name, number, ftype, label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL, type_name=None):
        fld = msg.field.add()
        fld.name = name
        fld.number = number
        fld.label = label
        fld.type = ftype
        if type_name:
            fld.type_name = type_name

    add("width", 1, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add("height", 2, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add("boxes", 3, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        type_name=f"{_PACKAGE}.Box")

    try:
        pool.Add(fdp)
    except Exception:
        pass

def _add_yolo_packet(pool: descriptor_pool.DescriptorPool):
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "model/protobuf/yolo-packet.proto"
    fdp.package = _PACKAGE
    fdp.syntax = "proto3"
    fdp.dependency.extend([
        "model/protobuf/box.proto",
        "model/protobuf/raw-frame.proto",
        "model/protobuf/mask-frame.proto",
        "model/protobuf/box-frame.proto",
    ])

    msg = fdp.message_type.add()
    msg.name = "YoloPacket"

    def add(name, number, ftype, label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL, type_name=None):
        fld = msg.field.add()
        fld.name = name
        fld.number = number
        fld.label = label
        fld.type = ftype
        if type_name:
            fld.type_name = type_name

    add("frame_id", 1, descriptor_pb2.FieldDescriptorProto.TYPE_INT64)
    # Note: field 2 intentionally unused in BroadTrack schema.
    add("raw_frame", 3, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, type_name=f"{_PACKAGE}.RawFrame")
    add("mask_frame", 4, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, type_name=f"{_PACKAGE}.MaskFrame")
    add("box_frame", 5, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, type_name=f"{_PACKAGE}.BoxFrame")

    try:
        pool.Add(fdp)
    except Exception:
        pass

def _add_capture_packet(pool: descriptor_pool.DescriptorPool):
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "model/protobuf/capture-packet.proto"
    fdp.package = _PACKAGE
    fdp.syntax = "proto3"
    fdp.dependency.append("model/protobuf/raw-frame.proto")

    msg = fdp.message_type.add()
    msg.name = "CapturePacket"

    def add(name, number, ftype, type_name=None):
        fld = msg.field.add()
        fld.name = name
        fld.number = number
        fld.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        fld.type = ftype
        if type_name:
            fld.type_name = type_name

    add("frame_id", 1, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    add("raw_frame", 2, descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE, f"{_PACKAGE}.RawFrame")

    try:
        pool.Add(fdp)
    except Exception:
        pass

_pool = descriptor_pool.Default()
_add_common(_pool)
_add_raw_frame(_pool)
_add_mask_frame(_pool)
_add_box(_pool)
_add_box_frame(_pool)
_add_yolo_packet(_pool)
_add_capture_packet(_pool)

_factory = message_factory.MessageFactory(_pool)

PixelFormats = _pool.FindEnumTypeByName(f"{_PACKAGE}.PixelFormats")
PF_GRAY8 = 0
PF_BGR24 = 1

RawFrame = _factory.GetPrototype(_pool.FindMessageTypeByName(f"{_PACKAGE}.RawFrame"))
MaskFrame = _factory.GetPrototype(_pool.FindMessageTypeByName(f"{_PACKAGE}.MaskFrame"))
Box = _factory.GetPrototype(_pool.FindMessageTypeByName(f"{_PACKAGE}.Box"))
BoxFrame = _factory.GetPrototype(_pool.FindMessageTypeByName(f"{_PACKAGE}.BoxFrame"))
YoloPacket = _factory.GetPrototype(_pool.FindMessageTypeByName(f"{_PACKAGE}.YoloPacket"))
CapturePacket = _factory.GetPrototype(_pool.FindMessageTypeByName(f"{_PACKAGE}.CapturePacket"))
