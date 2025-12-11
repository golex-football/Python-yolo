from services.runtime.limits import bootstrap_limits, apply_torch_caps, apply_cv2_caps

# Load env, set priority/affinity/threads before heavy libs import.
bootstrap_limits()

# Try to configure torch/cv2 early. It's OK if they aren't installed in some envs.
try:
    import torch
    apply_torch_caps(torch)
except Exception as e:
    print(f"[limits] torch not configured: {e}")

try:
    import cv2
    apply_cv2_caps(cv2)
except Exception as e:
    print(f"[limits] cv2 caps not applied: {e}")
