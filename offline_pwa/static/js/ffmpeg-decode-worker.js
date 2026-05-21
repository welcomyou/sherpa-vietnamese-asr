import { FFmpeg } from "/vendor/ffmpeg/ffmpeg/classes.js";

const CORE_URL = "/vendor/ffmpeg/core/ffmpeg-core.js";
const WASM_URL = "/vendor/ffmpeg/core/ffmpeg-core.wasm";
const TARGET_RATE = 16000;
const DEFAULT_DECODE_TIMEOUT_MS = 180000;
const LONG_DECODE_SECONDS = 30 * 60;
const LONG_DECODE_CHUNK_SECONDS = 5 * 60;
const LONG_DECODE_MIN_BYTES = 96 * 1024 * 1024;

let ffmpeg = null;
let ffmpegLoadPromise = null;
let lastLogLines = [];

function post(type, payload = {}, transfer = []) {
  self.postMessage({ type, ...payload }, transfer);
}

function errorMessage(error) {
  return error?.message || String(error);
}

function sanitizeName(name) {
  const clean = String(name || "input")
    .replace(/\\/g, "/")
    .split("/")
    .pop()
    .replace(/[^a-zA-Z0-9_.-]/g, "_");
  return clean || "input";
}

async function ensureFfmpeg() {
  if (ffmpegLoadPromise) return ffmpegLoadPromise;
  ffmpeg = new FFmpeg();
  ffmpeg.on("log", ({ type, message }) => {
    const line = `${type || "ffmpeg"}: ${message || ""}`.trim();
    if (line) {
      lastLogLines.push(line);
      if (lastLogLines.length > 60) lastLogLines = lastLogLines.slice(-60);
    }
  });
  ffmpeg.on("progress", ({ progress }) => {
    if (Number.isFinite(progress)) {
      post("progress", { progress: Math.max(0, Math.min(1, progress)) });
    }
  });
  ffmpegLoadPromise = ffmpeg.load({ coreURL: CORE_URL, wasmURL: WASM_URL });
  await ffmpegLoadPromise;
  post("log", { message: "FFmpeg WASM decoder loaded." });
  return ffmpegLoadPromise;
}

async function cleanup(paths) {
  for (const path of paths) {
    try {
      await ffmpeg.deleteFile(path);
    } catch (_) {
      // Best effort cleanup in MEMFS.
    }
  }
}

async function probeAudio(inputPath) {
  const probePath = "probe.json";
  try {
    const ret = await ffmpeg.ffprobe([
      "-v", "error",
      "-select_streams", "a:0",
      "-show_entries", "stream=sample_rate,channels,duration,bit_rate:format=duration,bit_rate,size",
      "-of", "json",
      inputPath,
      "-o", probePath,
    ]);
    if (ret !== 0) return null;
    const data = await ffmpeg.readFile(probePath, "utf8");
    return JSON.parse(data);
  } catch (_) {
    return null;
  } finally {
    await cleanup([probePath]);
  }
}

function summarizeProbe(probe, byteLength = 0) {
  const stream = probe?.streams?.[0] || {};
  const format = probe?.format || {};
  const sampleRate = Number.parseInt(stream.sample_rate, 10);
  const channels = Number.parseInt(stream.channels, 10);
  let duration = Number.parseFloat(stream.duration);
  let durationEstimated = false;
  if (!Number.isFinite(duration)) {
    duration = Number.parseFloat(format.duration);
  }
  const bitRate = Number.parseFloat(stream.bit_rate) || Number.parseFloat(format.bit_rate);
  const formatSize = Number.parseFloat(format.size);
  const inputBytes = Number.isFinite(byteLength) && byteLength > 0
    ? byteLength
    : (Number.isFinite(formatSize) && formatSize > 0 ? formatSize : 0);
  if (!Number.isFinite(duration) && bitRate > 0 && inputBytes > 0) {
    duration = inputBytes * 8 / bitRate;
    durationEstimated = true;
  }
  return {
    sampleRate: Number.isFinite(sampleRate) ? sampleRate : null,
    channels: Number.isFinite(channels) ? channels : null,
    duration: Number.isFinite(duration) ? duration : null,
    durationEstimated,
    bitRate: Number.isFinite(bitRate) ? bitRate : null,
    inputBytes,
  };
}

function shouldUseChunkedDecode(probeSummary, byteLength) {
  return (Number.isFinite(probeSummary?.duration) && probeSummary.duration >= LONG_DECODE_SECONDS)
    || (Number.isFinite(byteLength) && byteLength >= LONG_DECODE_MIN_BYTES);
}

function ensureByteCapacity(buffer, needed) {
  if (buffer && buffer.length >= needed) return buffer;
  const nextLength = Math.max(needed, Math.ceil((buffer?.length || 0 || 1024) * 1.5));
  const next = new Uint8Array(nextLength);
  if (buffer?.length) next.set(buffer);
  return next;
}

function readFloat32Array(raw) {
  const buffer = raw.byteOffset === 0 && raw.byteLength === raw.buffer.byteLength
    ? raw.buffer
    : raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  if (buffer.byteLength % 4 !== 0) {
    throw new Error(`FFmpeg returned invalid f32 byte count: ${buffer.byteLength}`);
  }
  return new Float32Array(buffer);
}

function monoResampleTo16k(samples, sourceRate, channels) {
  const srcRate = Number.isFinite(sourceRate) && sourceRate > 0 ? sourceRate : TARGET_RATE;
  const srcChannels = Number.isFinite(channels) && channels > 0 ? Math.floor(channels) : 1;
  const frames = Math.floor((samples?.length || 0) / srcChannels);
  const mono = new Float32Array(frames);
  if (srcChannels === 1) {
    mono.set(samples.subarray(0, frames));
  } else {
    for (let frame = 0; frame < frames; frame += 1) {
      let sum = 0;
      const offset = frame * srcChannels;
      for (let ch = 0; ch < srcChannels; ch += 1) sum += samples[offset + ch] || 0;
      mono[frame] = sum / srcChannels;
    }
  }

  if (srcRate === TARGET_RATE) return mono;
  const outLength = Math.max(1, Math.round(mono.length * TARGET_RATE / srcRate));
  const out = new Float32Array(outLength);
  const ratio = srcRate / TARGET_RATE;
  for (let i = 0; i < outLength; i += 1) {
    const pos = i * ratio;
    const left = Math.min(mono.length - 1, Math.floor(pos));
    const right = Math.min(mono.length - 1, left + 1);
    const frac = pos - left;
    out[i] = mono[left] * (1 - frac) + mono[right] * frac;
  }
  return out;
}

async function runDecodeAttempt(args, outputPath, timeoutMs = DEFAULT_DECODE_TIMEOUT_MS) {
  lastLogLines = [];
  const ret = await ffmpeg.exec(args, timeoutMs);
  if (ret !== 0) {
    const detail = lastLogLines.slice(-10).join("\n");
    const timeoutNote = ret === 1 ? ` after ${Math.round(timeoutMs / 1000)}s` : "";
    throw new Error(`ffmpeg decode failed with code ${ret}${timeoutNote}${detail ? `\n${detail}` : ""}`);
  }
  return ffmpeg.readFile(outputPath);
}

async function decodeWithFfmpegChunked(inputPath, probeSummary, totalTimeoutMs) {
  const duration = Number(probeSummary?.duration || 0);
  if (!(duration > 0)) {
    throw new Error(
      "FFmpeg chunked decode requires a known audio duration " +
      `(bitrate=${probeSummary?.bitRate || "n/a"}, bytes=${probeSummary?.inputBytes || 0}).`
    );
  }

  const chunkCount = Math.ceil(duration / LONG_DECODE_CHUNK_SECONDS);
  const expectedBytes = Math.ceil(duration * TARGET_RATE) * 4;
  let pcmBytes = new Uint8Array(Math.max(4, expectedBytes + TARGET_RATE * 4));
  let writeOffset = 0;
  const perChunkTimeoutMs = Math.min(
    Math.max(120000, Math.ceil(totalTimeoutMs / Math.max(1, chunkCount))),
    300000
  );

  post("log", {
    message: `FFmpeg chunked decode: ${chunkCount} chunk(s), ${LONG_DECODE_CHUNK_SECONDS}s each.`,
  });

  for (let index = 0; index < chunkCount; index += 1) {
    const start = index * LONG_DECODE_CHUNK_SECONDS;
    const length = Math.min(LONG_DECODE_CHUNK_SECONDS, duration - start);
    if (!(length > 0)) break;
    const outputPath = `chunk_${index}.f32`;
    const args = [
      "-y",
      "-hide_banner",
      "-nostdin",
      "-loglevel", "error",
      "-ss", start.toFixed(3),
      "-t", length.toFixed(3),
      "-i", inputPath,
      "-map", "0:a:0",
      "-vn",
      "-ac", "1",
      "-ar", String(TARGET_RATE),
      "-f", "f32le",
      "-acodec", "pcm_f32le",
      outputPath,
    ];
    let raw;
    try {
      raw = await runDecodeAttempt(args, outputPath, perChunkTimeoutMs);
    } catch (error) {
      if (probeSummary.durationEstimated && index > 0) {
        post("log", { message: `FFmpeg chunk ${index + 1} ended early from estimated duration.` });
        break;
      }
      throw error;
    }
    const bytes = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
    if (!bytes.byteLength && probeSummary.durationEstimated && index > 0) {
      break;
    }
    pcmBytes = ensureByteCapacity(pcmBytes, writeOffset + bytes.byteLength);
    pcmBytes.set(bytes, writeOffset);
    writeOffset += bytes.byteLength;
    await cleanup([outputPath]);
    post("progress", { progress: Math.min(0.995, (index + 1) / chunkCount) });
  }

  const finalBuffer = pcmBytes.buffer.slice(0, writeOffset);
  if (finalBuffer.byteLength % 4 !== 0) {
    throw new Error(`FFmpeg chunked decode returned invalid f32 byte count: ${finalBuffer.byteLength}`);
  }
  return {
    pcmBuffer: finalBuffer,
    sampleRate: TARGET_RATE,
    originalSampleRate: probeSummary.sampleRate || null,
    channels: 1,
    duration: finalBuffer.byteLength / 4 / TARGET_RATE,
    decoder: "ffmpeg-wasm-chunked",
    resampler: "ffmpeg-default",
  };
}

async function decodeWithFfmpegChunkedToMessages(id, inputPath, probeSummary, totalTimeoutMs) {
  const duration = Number(probeSummary?.duration || 0);
  if (!(duration > 0)) {
    throw new Error(
      "FFmpeg streamed chunk decode requires a known audio duration " +
      `(bitrate=${probeSummary?.bitRate || "n/a"}, bytes=${probeSummary?.inputBytes || 0}).`
    );
  }

  const chunkCount = Math.ceil(duration / LONG_DECODE_CHUNK_SECONDS);
  const perChunkTimeoutMs = Math.min(
    Math.max(120000, Math.ceil(totalTimeoutMs / Math.max(1, chunkCount))),
    300000
  );
  let sampleOffset = 0;

  post("decoded-chunks-start", {
    id,
    chunkCount,
    sampleRate: TARGET_RATE,
    originalSampleRate: probeSummary.sampleRate || null,
    channels: 1,
    duration,
    decoder: "ffmpeg-wasm-chunked-stream",
    resampler: "ffmpeg-default",
  });

  for (let index = 0; index < chunkCount; index += 1) {
    const start = index * LONG_DECODE_CHUNK_SECONDS;
    const length = Math.min(LONG_DECODE_CHUNK_SECONDS, duration - start);
    if (!(length > 0)) break;
    const outputPath = `chunk_${index}.f32`;
    const args = [
      "-y",
      "-hide_banner",
      "-nostdin",
      "-loglevel", "error",
      "-ss", start.toFixed(3),
      "-t", length.toFixed(3),
      "-i", inputPath,
      "-map", "0:a:0",
      "-vn",
      "-ac", "1",
      "-ar", String(TARGET_RATE),
      "-f", "f32le",
      "-acodec", "pcm_f32le",
      outputPath,
    ];
    const raw = await runDecodeAttempt(args, outputPath, perChunkTimeoutMs);
    const bytes = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    if (buffer.byteLength % 4 !== 0) {
      throw new Error(`FFmpeg chunk ${index + 1} returned invalid f32 byte count: ${buffer.byteLength}`);
    }
    const samples = buffer.byteLength / 4;
    post("decoded-chunk", {
      id,
      chunkIndex: index,
      chunkCount,
      sampleOffset,
      samples,
      pcmBuffer: buffer,
    }, [buffer]);
    sampleOffset += samples;
    await cleanup([outputPath]);
    post("progress", { progress: Math.min(0.995, (index + 1) / chunkCount) });
  }

  post("decoded-chunks-complete", {
    id,
    sampleRate: TARGET_RATE,
    originalSampleRate: probeSummary.sampleRate || null,
    channels: 1,
    sampleCount: sampleOffset,
    duration: sampleOffset / TARGET_RATE,
    decoder: "ffmpeg-wasm-chunked-stream",
    resampler: "ffmpeg-default",
  });
}

async function decodeWithFfmpeg({ fileName, bytes, timeoutMs }) {
  await ensureFfmpeg();
  lastLogLines = [];

  const inputPath = sanitizeName(fileName);
  const outputPath = "output.f32";
  const rawOutputPath = "output.raw.f32";
  await ffmpeg.writeFile(inputPath, new Uint8Array(bytes));
  try {
    const probe = await probeAudio(inputPath);
    const sourceByteLength = bytes?.byteLength || bytes?.length || 0;
    const probeSummary = summarizeProbe(probe, sourceByteLength);
    if (shouldUseChunkedDecode(probeSummary, bytes?.byteLength || bytes?.length || 0)) {
      return await decodeWithFfmpegChunked(inputPath, probeSummary, timeoutMs || DEFAULT_DECODE_TIMEOUT_MS);
    }

    const primaryArgs = [
      "-y",
      "-hide_banner",
      "-nostdin",
      "-loglevel", "error",
      "-i", inputPath,
      "-map", "0:a:0",
      "-vn",
      "-ac", "1",
      "-ar", String(TARGET_RATE),
      "-f", "f32le",
      "-acodec", "pcm_f32le",
      outputPath,
    ];
    const raw = await runDecodeAttempt(primaryArgs, outputPath, timeoutMs);
    const out = readFloat32Array(raw);
    const buffer = out.byteOffset === 0 && out.byteLength === out.buffer.byteLength
      ? out.buffer
      : out.buffer.slice(out.byteOffset, out.byteOffset + out.byteLength);

    return {
      pcmBuffer: buffer,
      sampleRate: TARGET_RATE,
      originalSampleRate: null,
      channels: 1,
      duration: out.length / TARGET_RATE,
      decoder: "ffmpeg-wasm-default",
      resampler: "ffmpeg-default",
    };
  } catch (primaryError) {
    throw primaryError;
  } finally {
    await cleanup([inputPath, outputPath, rawOutputPath]);
  }
}

async function decodeWithFfmpegToChunks({ id, fileName, file, bytes, timeoutMs, durationHint }) {
  await ensureFfmpeg();
  lastLogLines = [];

  const inputName = fileName || file?.name || "input";
  const inputPath = sanitizeName(inputName);
  const sourceBytes = bytes || (file && await file.arrayBuffer());
  if (!sourceBytes) throw new Error("No input bytes for FFmpeg streamed chunk decode.");
  const sourceByteLength = sourceBytes?.byteLength || sourceBytes?.length || 0;
  post("log", { message: `FFmpeg streamed input bytes: ${sourceByteLength}.` });
  await ffmpeg.writeFile(inputPath, new Uint8Array(sourceBytes));
  try {
    const probe = await probeAudio(inputPath);
    const probeSummary = summarizeProbe(probe, sourceByteLength);
    const hintedDuration = Number(durationHint || 0);
    if (!(probeSummary.duration > 0) && hintedDuration > 0) {
      probeSummary.duration = hintedDuration;
      probeSummary.durationEstimated = true;
      post("log", { message: `FFmpeg using browser duration hint: ${hintedDuration.toFixed(3)}s.` });
    }
    await decodeWithFfmpegChunkedToMessages(
      id,
      inputPath,
      probeSummary,
      timeoutMs || DEFAULT_DECODE_TIMEOUT_MS
    );
  } finally {
    await cleanup([inputPath]);
  }
}

async function extractWithFfmpeg({ fileName, bytes, outputFormat, timeoutMs }) {
  await ensureFfmpeg();
  lastLogLines = [];

  const inputPath = sanitizeName(fileName);
  const format = outputFormat === "aac" ? "aac" : "mp3";
  const outputPath = `extracted.${format}`;
  await ffmpeg.writeFile(inputPath, new Uint8Array(bytes));

  const formatArgs = format === "aac"
    ? ["-c:a", "copy", "-f", "adts"]
    : ["-c:a", "copy", "-f", "mp3"];
  try {
    const args = [
      "-y",
      "-hide_banner",
      "-nostdin",
      "-loglevel", "error",
      "-i", inputPath,
      "-map", "0:a:0",
      "-vn",
      ...formatArgs,
      outputPath,
    ];
    const raw = await runDecodeAttempt(args, outputPath, timeoutMs);
    const buffer = raw.byteOffset === 0 && raw.byteLength === raw.buffer.byteLength
      ? raw.buffer
      : raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    return { audioBuffer: buffer, outputFormat: format };
  } finally {
    await cleanup([inputPath, outputPath]);
  }
}

self.onmessage = async (event) => {
  const { id, type } = event.data || {};
  try {
    if (type === "decode") {
      const result = await decodeWithFfmpeg(event.data);
      post("decoded", { id, ...result }, [result.pcmBuffer]);
      return;
    }
    if (type === "decode-chunks") {
      await decodeWithFfmpegToChunks(event.data);
      return;
    }
    if (type === "extract") {
      const result = await extractWithFfmpeg(event.data);
      post("extracted", { id, ...result }, [result.audioBuffer]);
      return;
    }
    throw new Error(`Unknown FFmpeg decoder message: ${type}`);
  } catch (error) {
    post("error", {
      id,
      message: errorMessage(error),
      stack: error?.stack || "",
    });
  }
};
