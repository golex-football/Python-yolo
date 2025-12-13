# services/broadtrack/proto_runtime.py
# Runtime (no-protoc) protobuf definitions matching BroadTrack src/model/protobuf
# Package: golex.virtualtracking.model
from __future__ import annotations

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

_POOL = descriptor_pool.DescriptorPool()
_FACTORY = message_factory.MessageFactory(_POOL)

def _build():
    fd = descriptor_pb2.FileDescriptorProto()
    fd.name = "model/protobuf/runtime_broadtrack.proto"
    fd.package = "golex.virtualtracking.model"
    fd.syntax = "proto3"

    # enum PixelFormats
    enum_pf = fd.enum_type.add()
    enum_pf.name = "PixelFormats"
    v0 = enum_pf.value.add(); v0.name="PF_GRAY8"; v0.number=0
    v1 = enum_pf.value.add(); v1.name="PF_BGR24"; v1.number=1

    # message RawFrame
    msg = fd.message_type.add(); msg.name="RawFrame"
    f = msg.field.add(); f.name="width"; f.number=1; f.label=1; f.type=13   # TYPE_UINT32
    f = msg.field.add(); f.name="height"; f.number=2; f.label=1; f.type=13
    f = msg.field.add(); f.name="pixel_format"; f.number=3; f.label=1; f.type=14; f.type_name=".golex.virtualtracking.model.PixelFormats"
    f = msg.field.add(); f.name="frame_data"; f.number=4; f.label=1; f.type=12  # TYPE_BYTES

    # message MaskFrame (same fields as RawFrame)
    msg = fd.message_type.add(); msg.name="MaskFrame"
    f = msg.field.add(); f.name="width"; f.number=1; f.label=1; f.type=13
    f = msg.field.add(); f.name="height"; f.number=2; f.label=1; f.type=13
    f = msg.field.add(); f.name="pixel_format"; f.number=3; f.label=1; f.type=14; f.type_name=".golex.virtualtracking.model.PixelFormats"
    f = msg.field.add(); f.name="frame_data"; f.number=4; f.label=1; f.type=12

    # message Box
    msg = fd.message_type.add(); msg.name="Box"
    f = msg.field.add(); f.name="x"; f.number=1; f.label=1; f.type=13
    f = msg.field.add(); f.name="y"; f.number=2; f.label=1; f.type=13
    f = msg.field.add(); f.name="width"; f.number=3; f.label=1; f.type=13
    f = msg.field.add(); f.name="height"; f.number=4; f.label=1; f.type=13
    f = msg.field.add(); f.name="score"; f.number=5; f.label=1; f.type=2   # TYPE_FLOAT
    f = msg.field.add(); f.name="class_id"; f.number=6; f.label=1; f.type=13

    # message BoxFrame
    msg = fd.message_type.add(); msg.name="BoxFrame"
    f = msg.field.add(); f.name="width"; f.number=1; f.label=1; f.type=13
    f = msg.field.add(); f.name="height"; f.number=2; f.label=1; f.type=13
    f = msg.field.add(); f.name="boxes"; f.number=3; f.label=3; f.type=11; f.type_name=".golex.virtualtracking.model.Box"

    # message YoloPacket
    msg = fd.message_type.add(); msg.name="YoloPacket"
    f = msg.field.add(); f.name="frame_id"; f.number=1; f.label=1; f.type=3   # TYPE_INT64
    f = msg.field.add(); f.name="raw_frame"; f.number=3; f.label=1; f.type=11; f.type_name=".golex.virtualtracking.model.RawFrame"
    f = msg.field.add(); f.name="mask_frame"; f.number=4; f.label=1; f.type=11; f.type_name=".golex.virtualtracking.model.MaskFrame"
    f = msg.field.add(); f.name="box_frame"; f.number=5; f.label=1; f.type=11; f.type_name=".golex.virtualtracking.model.BoxFrame"

    # message BroadTrackPacket (used BroadTrack -> Unreal; included for completeness)
    msg = fd.message_type.add(); msg.name="BroadTrackPacket"
    f = msg.field.add(); f.name="frame_id"; f.number=1; f.label=1; f.type=4   # TYPE_UINT64
    f = msg.field.add(); f.name="raw_frame"; f.number=3; f.label=1; f.type=11; f.type_name=".golex.virtualtracking.model.RawFrame"
    f = msg.field.add(); f.name="mask_frame"; f.number=4; f.label=1; f.type=11; f.type_name=".golex.virtualtracking.model.MaskFrame"
    f = msg.field.add(); f.name="box_frame"; f.number=5; f.label=1; f.type=11; f.type_name=".golex.virtualtracking.model.BoxFrame"

    _POOL.Add(fd)

_BUILT = False
def _ensure():
    global _BUILT
    if not _BUILT:
        _build()
        _BUILT = True

def get_message(name: str):
    _ensure()
    desc = _POOL.FindMessageTypeByName(f"golex.virtualtracking.model.{name}")
    return _FACTORY.GetPrototype(desc)

# Expose concrete classes
def RawFrame():
    return get_message("RawFrame")()

def MaskFrame():
    return get_message("MaskFrame")()

def Box():
    return get_message("Box")()

def BoxFrame():
    return get_message("BoxFrame")()

def YoloPacket():
    return get_message("YoloPacket")()

def BroadTrackPacket():
    return get_message("BroadTrackPacket")()

# Enum values
PF_GRAY8 = 0
PF_BGR24 = 1
