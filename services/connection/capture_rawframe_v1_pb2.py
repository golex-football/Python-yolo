# -*- coding: utf-8 -*-
"""Dynamic protobuf for Capture-main RawFrame (the one currently sent on ipc:///tmp/capture).

Capture-main (this zip) uses:
  message RawFrame {
    uint64 frame_id = 1;
    uint64 timestamp = 2;
    uint32 width = 3;
    uint32 height = 4;
    PixelFormats pixel_format = 5;
    bytes  frame_data = 6;
  }

We define it under a different package/name to avoid clashing with BroadTrack's RawFrame
(which uses different field numbers).
"""

from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import message_factory

_PACKAGE = "golex.virtualtracking.capture"

def _build_pool() -> descriptor_pool.DescriptorPool:
    pool = descriptor_pool.Default()

    # enum PixelFormats (match BroadTrack values used by Capture-main)
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "capture/pixel_formats.proto"
    fdp.package = _PACKAGE
    fdp.syntax = "proto3"

    enum = fdp.enum_type.add()
    enum.name = "PixelFormats"
    v = enum.value.add(); v.name = "PF_GRAY8"; v.number = 0
    v = enum.value.add(); v.name = "PF_BGR24"; v.number = 1

    msg = fdp.message_type.add()
    msg.name = "CaptureRawFrameV1"

    def add_field(name, num, ftype, label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL, type_name=None):
        f = msg.field.add()
        f.name = name
        f.number = num
        f.label = label
        f.type = ftype
        if type_name:
            f.type_name = type_name

    add_field("frame_id", 1, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    add_field("timestamp", 2, descriptor_pb2.FieldDescriptorProto.TYPE_UINT64)
    add_field("width", 3, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add_field("height", 4, descriptor_pb2.FieldDescriptorProto.TYPE_UINT32)
    add_field("pixel_format", 5, descriptor_pb2.FieldDescriptorProto.TYPE_ENUM, type_name=f".{_PACKAGE}.PixelFormats")
    add_field("frame_data", 6, descriptor_pb2.FieldDescriptorProto.TYPE_BYTES)

    try:
        pool.Add(fdp)
    except Exception:
        # already added
        pass
    return pool

_pool = _build_pool()
_factory = message_factory.MessageFactory(_pool)

CaptureRawFrameV1 = _factory.GetPrototype(_pool.FindMessageTypeByName(f"{_PACKAGE}.CaptureRawFrameV1"))

# convenience enum values
PF_GRAY8 = 0
PF_BGR24 = 1
