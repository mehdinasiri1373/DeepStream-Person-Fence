# Stage5 Live WebRTC Notes

## Why not direct `/deepstream/`?

DeepStream RTSP output worked, but MediaMTX WebRTC could get stuck on browser loading because the H264 RTP packetization from DeepStream produced repeated `FU-A packet in packetization mode 0` errors.

The stable solution is:

```text
DeepStream RTSP -> ffmpeg decode/re-encode -> MediaMTX path /out/ -> WebRTC browser
```

## Final live URL

```text
http://localhost:8889/out/
```

## Runtime ports

- MediaMTX RTSP input/output: `8554`
- MediaMTX WebRTC HTTP: `8889`
- MediaMTX WebRTC ICE UDP: `8189/udp`
- DeepStream internal RTSP exposed to host: `8555 -> container 8554`

## Why keep `stage5-deepstream` persistent?

The DeepStream image needs build dependencies, DeepStream-Yolo parser build, Python packages, model export and TensorRT engine creation. Keeping the container persistent avoids repeating apt installs on every run. Project files such as model, ONNX and engine are still stored in the project folder.
