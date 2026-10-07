# Turing — Dev 1 workspace

**Owner:** Dev 1 (Perception, AI & Vision)  
**Authority:** [`architecture.md`](../architecture.md) wins over [`dev.md`](../dev.md) on any conflict.  
**Product path:** live outdoor camera → Perception Port. Not a demo, not a bag player, not a model.

This folder splits Dev 1 into **independently buildable, independently testable modules**. Build in order. Do not skip a module by stubbing it with fake vision.

## What Dev 1 is

Dev 1 is **not** the brain. The brain is RTAB-Map + Nav2 (Dev 2 / Dev 4).

Dev 1 owns:

1. **The Perception Port** — the only semantic contract the rest of the product is allowed to see.
2. **Adapters** that implement that port (RUGD SegFormer-B5 outdoor default; YOLOE selectable; tutorial ONNX is eval-only).
3. **Optional geometry side-channel** (Depth Anything) that is **not** the port.

Downstream (Dev 3 costmaps, Dev 5 safety) bind only to port outputs. They never import YOLOE classes, ONNX labels, or raw model scores.

## Build order

```
T01 types + validators     pure contract, no camera
T02 consume Image+CameraInfo  decode + ros_bridge **shipped**; outdoor stream still Dev 5
T03 remap engine           YAML, no model
T04 confidence gates       YAML, no model
T05 freshness / degraded   time + flags, no model
T12 backend seam           OpenVINO GPU on Arc B580; CUDA PyTorch when Intel GPU is missing
T06 YOLOE adapter          selectable masks; live outdoor adapter is RUGD SegFormer (T12)
T07 port node              compose + publish canonical mask + degraded
T10 contract tests         T01–T07 wired: encoding, {0,1,2}, stamp, frame, degraded
T11 performance / latency  capture→mask, FPS, bounded queue  ← prefer real camera
T08 Depth Anything         optional geometry, parallel to port  ← needs weights
T09 tutorial ONNX          eval scaffold only, not product
```

**Implemented (2026-09-18):** T01–T05 kernels, T02 decode+ros_bridge, T06 pack, T07 `compose_tick` + Lyrical `adapter_node`, T12 seam.  
**On disk:** RUGD SegFormer-B5 OpenVINO IR (`weights/rugd-segformer.xml`, 25 RUGD classes, Arc `device=GPU`) is the live adapter. YOLOE-26s IR remains at `weights/yoloe-26s-seg.xml`. **No DummySource.** Outdoor product `infer` still needs a Dev 5 Image+CameraInfo stream.  
T10 uses header/label fixtures. T11 policy (latest-only queue + starve watchdog) **shipped**; live p95 waits on Dev 5. Outdoor camera is Dev 5.

Runtime: Intel Arc B580 **now** (OpenVINO 2026.4.0, `device=GPU`, CPU fallback). ROS 2 **Lyrical** on this RHEL 10 box. NVIDIA: CUDA + PyTorch on the same HuggingFace checkpoints, no OpenVINO. See [HARDWARE.md](HARDWARE.md).

**Weights (gitignored).** Intel: HuggingFace download, then export IR, then run.

```bash
cd turing
bash scripts/fetch_rugd_segformer.sh
.venv/bin/python scripts/export_rugd_segformer_openvino.py
bash scripts/fetch_da3metric_large.sh
.venv/bin/python scripts/export_da3metric_openvino.py --height 336 --width 504
```

NVIDIA: HuggingFace download only (`weights/rugd-segformer/` and `weights/da3metric-large/`), plus PyTorch with CUDA. Skip the export. The node picks CUDA when OpenVINO reports no Intel `GPU`.

## Live throughput on Intel Arc B580

Measured on this desktop, `thetestimage1.jpg` **408×612**, RUGD SegFormer-B5 + DA3METRIC-LARGE, OpenVINO GPU, **plugin-default main-net compile**, FP32 IR. GPU preprocess and GPU decode stay f32. One frame in flight.

**Original** = CPU resize / softmax / argmax. **GPU decode** = upsample → softmax → argmax on GPU. **GPU decode + preprocess** = that plus GPU resize + ImageNet. **+ multiprocess** = same GPU path, DA3 `maps()` on a worker while RUGD compose runs, then join for the same `stamp_ns` / `frame_id`.

| Metric | Original | GPU decode | GPU decode + preprocess | + multiprocess | Orig. → decode | Orig. → pre | Pre → multi |
|---|---:|---:|---:|---:|---:|---:|---:|
| RUGD preprocess | ~56 ms | ~56 ms | **1.8 ms** | **1.8 ms** | 1.0× | **31.1×** | 1.0× |
| RUGD inference | ~63 ms | ~57 ms | **~57 ms** | **~57 ms** | 1.1× | 1.1× | 1.0× |
| RUGD decode | ~292 ms | **~41 ms** | **~41 ms** | **~41 ms** | **7.1×** | **7.1×** | 1.0× |
| RUGD total (mask compose) | ~415 ms | ~98 ms | **61.2 ms** | **61.2 ms** | **4.23×** | **6.78×** | 1.0× |
| RUGD FPS | ~2.4 | ~10.2 | **16.3** | **16.3** | **4.25×** | **6.8×** | 1.0× |
| DA3 preprocess | ~21 ms | ~21 ms | **1.1 ms** | **1.1 ms** | 1.0× | **19.1×** | 1.0× |
| DA3 total (`maps()`) | ~63.4 ms | ~63.4 ms | **28.5 ms** | **28.5 ms** | 1.0× | **2.22×** | 1.0× |
| DA3 FPS | ~15.8 | ~15.8 | **35.1** | **35.1** | 1.0× | **2.22×** | 1.0× |
| RUGD + DA3 live tick | **~465 ms / 2.15 FPS** | — | **87.8 ms / 11.4 FPS** | **76.0 ms / 13.2 FPS** | — | **5.3×** | **1.16×** |
| Dual-resident GPU peak | — | — | **1.72 GiB** | **1.72 GiB** | — | — | 1.0× |
| Host RSS (both nets) | — | — | **~455 MiB** | **~455 MiB** | — | — | 1.0× |

Multiprocess does **not** run the two GPU nets at the same time on this Arc (OpenVINO still serializes those kernels). It hides DA3 CPU (hole-safe / remap) under RUGD GPU time. CUDA uses the same pool with a private stream per net; NVIDIA fps is unmeasured here.

Do not compile the main nets with `INFERENCE_PRECISION_HINT=f32` or `LATENCY`: that dropped this box from ~16 FPS mask / ~11 FPS sequential to ~7 / ~3.6. Pre and post graphs stay f32.

### T07 port node (explicit)

- Compose ingestion + RUGD SegFormer + remap + confidence + freshness
- Publish the canonical product mask
- Publish `perception_degraded`
- Preserve original sensor timestamp
- Preserve optical `frame_id`
- No model-specific logic downstream (no YOLOE/OpenVINO/TensorRT in this node)

## Module graph

```
                    [T02 Source]
                    ImageFrame (real)
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   [T06 YOLOE adapter]     [T08 Depth Anything]
   RawSemOutput            DepthFrame
          │                 (NOT the port)
          ▼
   [T03 Remap] ── YAML ontologies
          │
          ▼
   [T04 Conf gates] ── YAML profile
          │
          ▼
   [T05 Freshness]
          │
          ▼
   [T07 Perception Port]          [T01 validators on every publish]
          │                       [T10 contract tests on the wire]
          │                       [T11 latency / backpressure]
          ├─ /segmentation/mask            {0,1,2} mono8
          ├─ /segmentation/confidence      [0,1] 32FC1
          ├─ /segmentation/port_meta       valid, age (informational), scale
          └─ /ugv/perception_degraded      Bool → Dev 5
```

T06 calls `InferenceBackend` (T12). T07 never does.  
T09 plugs in at the same adapter seam as T06. It must not ship as the live_cam default.

## Laws that every task inherits

Copied from architecture; not optional:

| Law | Meaning for Dev 1 |
|---|---|
| Adapters are sources, not the brain | YOLOE never talks to Nav2 or `/cmd_vel` |
| 3-class port only | pixels ∈ `{0,1,2}` — no `cautious` |
| No remap → no publish | adapter without YAML is a hard error |
| Normalize then gate | τ_* apply in `[0,1]`, never on raw multi-model scores |
| Stale ≠ current | age > `perception_max_age` → degraded, do not present mask as live |
| Unknown ≠ free | class 0 inflates; never “fill unknown with traversable” |
| Geometry is a side-channel | Depth Anything does not write `/segmentation/mask` |
| Front ROI lethal on fail | Dev 1 publishes degraded; **Dev 3** inflates the costmap. We do not write costmaps. |

## Data policy (product)

**Forbidden:** synthetic images, simulator cameras as product perception, random masks, constant “all traversable” stubs, fake `CameraInfo`, recycled stamps, downloaded demo clips presented as this UGV’s camera.

**Allowed:** live calibrated camera, outdoor recordings from that (or the target) camera, pretrained weights, YAML we author, numeric tables for remap/gate unit tests (label→id, score→class). Numeric tables are contract tests, not perception data.

If a task cannot be tested without a fake camera, the task is not done — get a real source (T02) first.

## Files

| Path | What |
|---|---|
| [00-role-and-laws.md](00-role-and-laws.md) | Dev 1’s place in the product |
| [interfaces.md](interfaces.md) | Internal APIs between modules |
| [DATASETS.md](DATASETS.md) | Weights / datasets — **read this, action required** |
| [HARDWARE.md](HARDWARE.md) | Arc B580 OpenVINO GPU; ROS Lyrical; YOLOE-26s; no camera device |
| [CONFLICTS.md](CONFLICTS.md) | Where `dev.md` is ignored |
| [tasks/](tasks/) | T01–T12 build checklists |
| [subarchs/subarch1.md](subarchs/subarch1.md) | T01 port-kernel architecture (wins over the T01 checklist) |
| [subarchs/subarch2.md](subarchs/subarch2.md) | T02 consume-camera-data architecture (not the device; wins over V4L2 T02) |
| [subarchs/subarch3.md](subarchs/subarch3.md) | T03 remap-kernel architecture (wins over the T03 checklist) |
| [subarchs/subarch4.md](subarchs/subarch4.md) | T04 confidence-gate architecture (wins over the T04 checklist) |
| [subarchs/subarch5.md](subarchs/subarch5.md) | T05 freshness/degraded-policy architecture (wins over the T05 checklist) |
| [subarchs/subarch6.md](subarchs/subarch6.md) | T06 YOLOE adapter architecture (wins over the T06 checklist) |
| [subarchs/subarch7.md](subarchs/subarch7.md) | T07 port-composition architecture (wins over the T07 checklist) |
| [subarchs/subarch8.md](subarchs/subarch8.md) | T08 depth side-channel (DA3METRIC-LARGE; frozen; head-only export) |
| [subarchs/subarch9.md](subarchs/subarch9.md) | T09 tutorial ONNX eval scaffold (not live; not frozen) |
| [subarchs/subarch10.md](subarchs/subarch10.md) | T10 wired port-contract tests (wins over the T10 checklist; T08 still deferred) |
| [subarchs/subarch11.md](subarchs/subarch11.md) | T11 latency / latest-only queue (wins over the T11 checklist; T08 still deferred) |
| [subarchs/subarch12.md](subarchs/subarch12.md) | T12 OpenVINO GPU seam (matches shipped T06 Protocol; wins over old T12 checklist) |

Code for each task lands under `turing/src/` when we implement. Do not start T06/T08 until weights are on disk (see DATASETS.md).
