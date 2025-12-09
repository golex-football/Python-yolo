import os
from pathlib import Path

def _project_root():
    return Path(__file__).resolve().parents[2]

def _parse_affinity(spec: str):
    out = set()
    spec = (spec or "").strip()
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                a, b = int(a), int(b)
                if a > b:
                    a, b = b, a
                out.update(range(a, b + 1))
            except ValueError:
                pass
        else:
            try:
                out.add(int(part))
            except ValueError:
                pass
    return out

def _load_env_file():
    # Allow override via MOONALT_LIMITS_FILE, else repo_root/limits.env
    user_path = os.environ.get("MOONALT_LIMITS_FILE", "")
    candidates = []
    if user_path:
        candidates.append(Path(user_path))
    candidates.append(_project_root() / "limits.env")

    for p in candidates:
        if p.is_file():
            try:
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
                print(f"[limits] loaded caps from {p}")
            except Exception as e:
                print(f"[limits] failed reading {p}: {e}")
            break

def _apply_nice():
    val = os.environ.get("MOONALT_NICE", "").strip()
    if not val:
        return
    try:
        nice = int(val)
        try:
            os.setpriority(os.PRIO_PROCESS, 0, nice)  # absolute priority if possible
        except Exception:
            os.nice(max(nice, 0))  # relative fallback
        print(f"[limits] process nice -> {nice}")
    except Exception as e:
        print(f"[limits] set nice failed: {e}")

def _apply_affinity():
    cpus = _parse_affinity(os.environ.get("MOONALT_CPU_AFFINITY", ""))
    if not cpus:
        return
    try:
        os.sched_setaffinity(0, cpus)
        print(f"[limits] cpu affinity -> {sorted(cpus)}")
    except Exception as e:
        print(f"[limits] set cpu affinity failed: {e}")

def _apply_thread_envs():
    def setdef(k, v):
        os.environ.setdefault(k, str(v))
    thr = int(os.environ.get("MOONALT_CPU_THREADS", "6"))
    for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        setdef(k, thr)
    setdef("OPENCV_NUM_THREADS", os.environ.get("OPENCV_NUM_THREADS","0"))
    print(f"[limits] threads -> {thr} (OMP/MKL/BLAS/NUMEXPR), OpenCV={os.environ.get('OPENCV_NUM_THREADS')}")

def bootstrap_limits():
    """
    Call automatically via sitecustomize (see sitecustomize.py).
    Ensures env/thread/priority/affinity are applied before heavy imports.
    """
    _load_env_file()
    _apply_nice()
    _apply_affinity()
    _apply_thread_envs()
    return {
        "vram_fraction": float(os.environ.get("MOONALT_TORCH_VRAM_FRACTION", "0.60")),
        "device": os.environ.get("MOONALT_YOLO_DEVICE", "cuda:0"),
    }

def apply_cv2_caps(cv2):
    try:
        n = int(os.environ.get("OPENCV_NUM_THREADS", "0"))
        cv2.setNumThreads(n)
    except Exception:
        pass
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

def apply_torch_caps(torch):
    # Run before first CUDA allocation
    dev = os.environ.get("MOONALT_YOLO_DEVICE", "cuda:0")
    frac = float(os.environ.get("MOONALT_TORCH_VRAM_FRACTION", "0.60"))
    try:
        if "cuda" in dev and torch.cuda.is_available():
            idx = 0
            if ":" in dev:
                try:
                    idx = int(dev.split(":")[1])
                except Exception:
                    idx = 0
            try:
                torch.cuda.set_per_process_memory_fraction(frac, device=idx)
                print(f"[limits] torch VRAM cap -> {frac*100:.0f}% on {dev}")
            except Exception as e:
                print(f"[limits] VRAM cap not applied: {e}")
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    except Exception as e:
        print(f"[limits] torch caps skipped: {e}")
