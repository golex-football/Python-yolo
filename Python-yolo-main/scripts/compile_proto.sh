#!/usr/bin/env bash
set -euo pipefail
# Compile protobufs into Python. Run from repo root on Linux/WSL:
#   bash scripts/compile_proto.sh
# Requires: pip install grpcio-tools protobuf
python -m grpc_tools.protoc -I services/proto --python_out=services/proto services/proto/messages.proto
echo "Generated services/proto/messages_pb2.py"
