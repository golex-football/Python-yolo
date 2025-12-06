\# Moonalt YOLO ZMQ Worker – Instructions



This repo contains the \*\*Python side\*\* of the pipeline that:



\- Receives \*\*raw video frames\*\* over \*\*ZeroMQ (ZMQ)\*\*.

\- Runs \*\*YOLOv8 segmentation\*\* (currently `yolov8l-seg.pt`) on GPU.

\- Produces:

&nbsp; - A \*\*binary mask\*\* (players + ball = white, everything else = black).

&nbsp; - A list of \*\*bounding boxes\*\* for `person` and `sports ball`.

&nbsp; - The \*\*original raw frame\*\* (unchanged).

\- Sends all of that out again over ZMQ for \*\*BroadTrack / downstream C++\*\*.



Later, a \*\*DeckLink C++\*\* app will replace the Python OBS capture and push real capture-card frames to this worker using the same ZMQ contract.



---



\## 0. Project Layout (important files)



\- `yolo\_worker\_zmq.py`  

&nbsp; The main YOLO worker:

&nbsp; - PULLs raw frames from ZMQ.

&nbsp; - Runs YOLOv8 segmentation.

&nbsp; - Builds mask + bboxes JSON.

&nbsp; - PUSHes meta + raw frame + mask over ZMQ.



\- `capture\_obs\_zmq.py`  

&nbsp; Dev-only helper:

&nbsp; - Reads from \*\*OBS Virtual Camera\*\*.

&nbsp; - Sends raw frames to the worker over ZMQ, pretending to be the DeckLink app.



\- `debug\_downstream\_zmq.py`  

&nbsp; Dev-only helper:

&nbsp; - Listens on the worker’s output socket.

&nbsp; - Prints JSON meta and can show the mask to verify everything.



\- `requirements.txt`  

&nbsp; Python dependencies (except `torch` / `torchvision`, which are installed specially in Docker).



\- `Dockerfile`  

&nbsp; Builds a \*\*GPU-enabled\*\* Docker image with CUDA, PyTorch, ultralytics, etc.



\- `INSTRUCTIONS.md` (this file)  

&nbsp; Human-readable documentation.



---



\## 1. ZMQ Contracts (what C++ / BroadTrack must send \& receive)



\### 1.1 Input: RAW frame from DeckLink/OBS → YOLO worker



\*\*Socket type (YOLO side):\*\*



\- `ZMQ\_IN\_ENDPOINT` (default: `tcp://127.0.0.1:5555`)

\- Python uses a \*\*PULL\*\* socket.

\- Sender (DeckLink or `capture\_obs\_zmq.py`) uses \*\*PUSH\*\*.



\*\*Multipart message (2 parts):\*\*



1\. \*\*Part 0: JSON meta (UTF-8)\*\*

&nbsp;  ```json

&nbsp;  {

&nbsp;    "schema": "moonalt.raw\_frame\_v1",

&nbsp;    "frame\_id": 123,

&nbsp;    "timestamp\_ms": 1733250000123,

&nbsp;    "width": 1920,

&nbsp;    "height": 1080,

&nbsp;    "pixel\_format": "BGR24"

&nbsp;  }

schema must be "moonalt.raw\_frame\_v1".



frame\_id is a monotonically increasing integer.



timestamp\_ms = Unix time in milliseconds (optional but recommended).



width, height = frame resolution.



pixel\_format = "BGR24" (same as OpenCV CV\_8UC3).



Part 1: raw frame bytes



Exactly width \* height \* 3 bytes.



BGR order, row-major.



This is the “holy raw” that must never be modified inside the Python worker.



The worker reconstructs the frame like this:



python

Copy code

frame\_bgr\_orig = np.frombuffer(frame\_bytes, dtype=np.uint8).reshape(h, w, 3)

1.2 Output: YOLO worker → BroadTrack / downstream

Socket type (YOLO side):



ZMQ\_OUT\_ENDPOINT (default: tcp://127.0.0.1:5556)



Python uses a PUSH socket.



Consumer (BroadTrack C++ / debugger script) uses PULL.



Multipart message (3 parts):



Part 0: JSON meta (UTF-8)

Example:



json

Copy code

{

&nbsp; "schema": "moonalt.human\_bboxes\_v1",

&nbsp; "frame\_id": 123,

&nbsp; "width": 1920,

&nbsp; "height": 1080,

&nbsp; "pixel\_format": "BGR24",

&nbsp; "mask\_format": "GRAY8",

&nbsp; "bboxes": \[

&nbsp;   \[x1, y1, x2, y2],

&nbsp;   \[x1, y1, x2, y2]

&nbsp; ],

&nbsp; "detections": \[

&nbsp;   {

&nbsp;     "id": 0,

&nbsp;     "class\_id": 0,

&nbsp;     "class\_name": "person",

&nbsp;     "conf": 0.87,

&nbsp;     "x1": 100,

&nbsp;     "y1": 200,

&nbsp;     "x2": 160,

&nbsp;     "y2": 350

&nbsp;   },

&nbsp;   {

&nbsp;     "id": 1,

&nbsp;     "class\_id": 32,

&nbsp;     "class\_name": "sports ball",

&nbsp;     "conf": 0.80,

&nbsp;     "x1": 800,

&nbsp;     "y1": 450,

&nbsp;     "x2": 820,

&nbsp;     "y2": 470

&nbsp;   }

&nbsp; ]

}

width, height, pixel\_format = same as original raw frame.



mask\_format = "GRAY8" (one byte per pixel).



bboxes = quick BroadTrack-friendly list \[\[x1, y1, x2, y2], ...].



detections = richer per-object info, in original resolution coordinates:



class\_id:



0 → person



32 → sports ball (COCO)



class\_name is a human-readable string.



Part 1: raw frame bytes (unchanged)



Identical bytes as input.



Size = width \* height \* 3.



Part 2: mask bytes



Single-channel grayscale mask, same width/height, type uint8.



Player + ball pixels ≈ 255.



Background pixels ≈ 0.



Reconstruction on the C++ side (conceptually):



cpp

Copy code

// meta\_json: parse with your JSON lib

// raw\_frame: size width\*height\*3

// mask: size width\*height



// raw\_frame -> cv::Mat(height, width, CV\_8UC3, raw\_frame\_data);

// mask      -> cv::Mat(height, width, CV\_8UC1, mask\_data);

2\. YOLO Models and Classes

We are currently using:



Model: yolov8l-seg.pt (Ultralytics)



Mode: segmentation (people + other objects)



Classes we care about:



class\_id = 0 → "person"



class\_id = 32 → "sports ball"



The worker:



Runs YOLO on a resized version of the frame (to speed things up).



Scales all boxes back to original 1920×1080 coordinates.



Builds a combined mask where:



any pixel belonging to a person or sports ball = white (255).



everything else = black (0).



Applies some smoothing (blur + morphology) to get nicer edges.



3\. Local Dev: Running Everything with OBS

These steps are for developers who want to test on their own machine.



3.1 Requirements

Windows with:



Python 3.11+



A compatible NVIDIA GPU + proper drivers (for CUDA).



OBS Studio + OBS Virtual Camera (to play a soccer video).



This repo checked out at, for example:

C:\\Users\\sina\\Desktop\\soccer\_yolo\_stream



3.2 One-time: Create venv and install deps

In PowerShell:



powershell

Copy code

cd C:\\Users\\sina\\Desktop\\soccer\_yolo\_stream

python -m venv .venv

. .venv/Scripts/Activate.ps1

pip install --upgrade pip

pip install -r requirements.txt

If PyTorch/torchvision are not installed in the venv yet:



powershell

Copy code

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Do not copy the (.venv) PS ... part of the prompt.

Only type the commands after it.



3.3 Terminal 1 – Capture from OBS and send via ZMQ

Open OBS, start Virtual Camera, and play your soccer video.



In a PowerShell:



powershell

Copy code

cd C:\\Users\\sina\\Desktop\\soccer\_yolo\_stream

. .venv/Scripts/Activate.ps1

python capture\_obs\_zmq.py

This script:



Opens camera index 1 (OBS Virtual Camera).



Logs something like:

\[CAPTURE] Sent FPS: 30.0, raw frame size: 1920x1080



PUSHes frames to tcp://127.0.0.1:5555.



If your OBS virtual cam is on a different index, you can change CAMERA\_INDEX in capture\_obs\_zmq.py.



3.4 Terminal 2 – Run YOLO worker

Open another PowerShell window:



powershell

Copy code

cd C:\\Users\\sina\\Desktop\\soccer\_yolo\_stream

. .venv/Scripts/Activate.ps1

python yolo\_worker\_zmq.py

You should see logs similar to:



\[WORKER] MODEL\_NAME = yolov8l-seg.pt, INFER\_WIDTH = 960



\[WORKER] Torch CUDA available: True



\[WORKER] ZMQ PULL connected to tcp://127.0.0.1:5555



\[WORKER] ZMQ PUSH bound to tcp://127.0.0.1:5556



\[WORKER] HUMAN\_BBOXES JSON: {...}



\[WORKER] FPS: XX.X | Inference: YY.Y ms | BBoxes: N | frame\_id: M



A window should pop up showing:



Left: original raw frame.



Right: binary mask.



Press Esc in that window to stop.



3.5 Terminal 3 – Debug listener (simulate BroadTrack consumer)

Optional, but useful for testing the ZMQ output.



Open a third PowerShell:



powershell

Copy code

cd C:\\Users\\sina\\Desktop\\soccer\_yolo\_stream

. .venv/Scripts/Activate.ps1

python debug\_downstream\_zmq.py

This will:



PULL messages from tcp://127.0.0.1:5556.



Print the JSON meta.



Show a preview of the mask.



4\. How DeckLink C++ App Should Talk to This

This section is for the C++ developer who will integrate the DeckLink capture.



4.1 C++ → Python (input contract recap)

From C++:



Open a ZMQ PUSH socket to ZMQ\_IN\_ENDPOINT, default tcp://127.0.0.1:5555.



For each captured frame:



Get a BGR buffer of size width \* height \* 3 (or convert from whatever DeckLink gives you).



Build the JSON meta (see the 1.1 section).



Send as multipart \[meta\_json, raw\_frame\_bytes].



Pseudo-code sketch:



cpp

Copy code

// PSEUDOCODE (not full C++):



zmq::context\_t ctx(1);

zmq::socket\_t socket(ctx, zmq::socket\_type::push);

socket.connect("tcp://127.0.0.1:5555");



// Inside your frame capture loop:

nlohmann::json meta;

meta\["schema"] = "moonalt.raw\_frame\_v1";

meta\["frame\_id"] = frame\_id++;

meta\["timestamp\_ms"] = current\_time\_ms();

meta\["width"] = width;

meta\["height"] = height;

meta\["pixel\_format"] = "BGR24";



std::string meta\_str = meta.dump();

zmq::message\_t meta\_msg(meta\_str.begin(), meta\_str.end());



// raw\_bgr is e.g. std::vector<uint8\_t> or uint8\_t\*

zmq::message\_t frame\_msg(raw\_bgr\_size);

memcpy(frame\_msg.data(), raw\_bgr\_data, raw\_bgr\_size);



std::vector<zmq::message\_t> parts;

parts.push\_back(std::move(meta\_msg));

parts.push\_back(std::move(frame\_msg));



socket.send(parts.begin(), parts.end());

As long as this matches the contract in Section 1.1, the Python worker will happily consume it.



5\. Docker Usage (for studio / 2×4090 setup)

5.1 Build the image

On a machine with Docker (and ideally NVIDIA Container Toolkit installed):



bash

Copy code

cd /path/to/soccer\_yolo\_stream

docker build -t moonalt-yolo-worker .

5.2 Run the container with GPU + host network (Linux)

Example (Linux):



bash

Copy code

docker run --gpus all --network host \\

&nbsp; -e ZMQ\_IN\_ENDPOINT=tcp://127.0.0.1:5555 \\

&nbsp; -e ZMQ\_OUT\_ENDPOINT=tcp://0.0.0.0:5556 \\

&nbsp; moonalt-yolo-worker

--gpus all makes both RTX 4090s visible.



--network host lets the DeckLink C++ app on the same machine use 127.0.0.1:5555.



You can change endpoints by changing the environment variables.



In production:



DeckLink C++ app runs on the host.



YOLO worker runs inside Docker.



For the C++ app, nothing changes: it still just pushes frames to ZMQ\_IN\_ENDPOINT.



6\. Git / GitLab Notes

Typical workflow to commit \& push:



bash

Copy code

git init

git add .

git commit -m "Initial YOLO ZMQ worker: DeckLink-ready, Dockerized, docs"

git remote add origin <GITLAB\_SSH\_OR\_HTTPS\_URL>

git branch -M main

git push -u origin main

Make sure INSTRUCTIONS.md, Dockerfile, requirements.txt, and all .py files are committed so teammates can clone and run with minimal friction.



7\. Tuning \& Future Extensions

Accuracy vs speed:



Currently using yolov8l-seg.pt and INFER\_WIDTH = 960.



You can:



Decrease INFER\_WIDTH for more speed (less accuracy).



Switch to yolov8s-seg.pt for faster but less accurate results.



Switch to yolov8x-seg.pt for more accuracy but heavier compute (4090 might still handle it fine).



Threshold \& classes:



CONF\_THRES (in yolo\_worker\_zmq.py) controls detection confidence.



Currently limited to:



person (class 0)



sports ball (class 32)



Ball-only mask or different labels:



You can extend build\_binary\_mask() to treat ball vs players differently (e.g., different gray values or separate channel) if BroadTrack later needs that.



If you are reading this and something is unclear:



Start with Section 3 (Local Dev) using OBS.



Once you see the worker processing frames and sending masks, move on to integrating DeckLink C++ using Section 4.



yaml

Copy code



---



If you want, next we can also tighten up your existing `README.md` to be a short “marketing/summary” and keep `INSTRUCTIONS.md` as the big, detailed how-to.

::contentReference\[oaicite:0]{index=0}



















