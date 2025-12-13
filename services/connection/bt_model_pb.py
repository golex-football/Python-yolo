"""Runtime protobuf definitions matching BroadTrack src/model/protobuf.

This module defines the *exact* wire format BroadTrack expects for:
  package golex.virtualtracking.model
  PixelFormats, RawFrame, MaskFrame, Box, BoxFrame, YoloPacket

We build descriptors at runtime because protoc may not be available.
"""

from __future__ import annotations

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

_POOL = descriptor_pool.Default()


def _build_once() -> None:
    try:
        _POOL.FindFileByName("model/protobuf/yolo-packet.proto")
        return
    except Exception:
        pass

    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "model/protobuf/yolo-packet.proto"
    fdp.package = "golex.virtualtracking.model"
    fdp.syntax = "proto3"

    enum_pf = fdp.enum_type.add()
    enum_pf.name = "PixelFormats"
    v0 = enum_pf.value.add(); v0.name = "PF_GRAY8"; v0.number = 0
    v1 = enum_pf.value.add(); v1.name = "PF_BGR24"; v1.number = 1

    m_raw = fdp.message_type.add(); m_raw.name = "RawFrame"
    f = m_raw.field.add(); f.name = "width";  f.number = 1; f.label = 1; f.type = 13
    f = m_raw.field.add(); f.name = "height"; f.number = 2; f.label = 1; f.type = 13
    f = m_raw.field.add(); f.name = "pixel_format"; f.number = 3; f.label = 1; f.type = 14; f.type_name = ".golex.virtualtracking.model.PixelFormats"
    f = m_raw.field.add(); f.name = "frame_data"; f.number = 4; f.label = 1; f.type = 12

    m_mask = fdp.message_type.add(); m_mask.name = "MaskFrame"
    f = m_mask.field.add(); f.name = "width";  f.number = 1; f.label = 1; f.type = 13
    f = m_mask.field.add(); f.name = "height"; f.number = 2; f.label = 1; f.type = 13
    f = m_mask.field.add(); f.name = "pixel_format"; f.number = 3; f.label = 1; f.type = 14; f.type_name = ".golex.virtualtracking.model.PixelFormats"
    f = m_mask.field.add(); f.name = "frame_data"; f.number = 4; f.label = 1; f.type = 12

    m_box = fdp.message_type.add(); m_box.name = "Box"
    f = m_box.field.add(); f.name = "x"; f.number = 1; f.label = 1; f.type = 13
    f = m_box.field.add(); f.name = "y"; f.number = 2; f.label = 1; f.type = 13
    f = m_box.field.add(); f.name = "width"; f.number = 3; f.label = 1; f.type = 13
    f = m_box.field.add(); f.name = "height"; f.number = 4; f.label = 1; f.type = 13
    f = m_box.field.add(); f.name = "score"; f.number = 5; f.label = 1; f.type = 2
    f = m_box.field.add(); f.name = "class_id"; f.number = 6; f.label = 1; f.type = 13

    m_bf = fdp.message_type.add(); m_bf.name = "BoxFrame"
    f = m_bf.field.add(); f.name = "width";  f.number = 1; f.label = 1; f.type = 13
    f = m_bf.field.add(); f.name = "height"; f.number = 2; f.label = 1; f.type = 13
    f = m_bf.field.add(); f.name = "boxes";  f.number = 3; f.label = 3; f.type = 11; f.type_name = ".golex.virtualtracking.model.Box"

    m_pkt = fdp.message_type.add(); m_pkt.name = "YoloPacket"
    f = m_pkt.field.add(); f.name = "frame_id"; f.number = 1; f.label = 1; f.type = 3
    f = m_pkt.field.add(); f.name = "raw_frame"; f.number = 3; f.label = 1; f.type = 11; f.type_name = ".golex.virtualtracking.model.RawFrame"
    f = m_pkt.field.add(); f.name = "mask_frame"; f.number = 4; f.label = 1; f.type = 11; f.type_name = ".golex.virtualtracking.model.MaskFrame"
    f = m_pkt.field.add(); f.name = "box_frame"; f.number = 5; f.label = 1; f.type = 11; f.type_name = ".golex.virtualtracking.model.BoxFrame"

    _POOL.Add(fdp)


_build_once()
_FACTORY = message_factory.MessageFactory(_POOL)

PixelFormats = _POOL.FindEnumTypeByName("golex.virtualtracking.model.PixelFormats")
RawFrame = _FACTORY.GetPrototype(_POOL.FindMessageTypeByName("golex.virtualtracking.model.RawFrame"))
MaskFrame = _FACTORY.GetPrototype(_POOL.FindMessageTypeByName("golex.virtualtracking.model.MaskFrame"))
Box = _FACTORY.GetPrototype(_POOL.FindMessageTypeByName("golex.virtualtracking.model.Box"))
BoxFrame = _FACTORY.GetPrototype(_POOL.FindMessageTypeByName("golex.virtualtracking.model.BoxFrame"))
YoloPacket = _FACTORY.GetPrototype(_POOL.FindMessageTypeByName("golex.virtualtracking.model.YoloPacket"))

PF_GRAY8 = 0
PF_BGR24 = 1
