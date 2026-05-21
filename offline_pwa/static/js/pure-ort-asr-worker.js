const ORT_BASE = "/vendor/onnxruntime-web/";

const SAMPLE_RATE = 16000;
const FRAME_LENGTH = 400;
const FRAME_SHIFT = 160;
const N_FFT = 512;
const NUM_MEL_BINS = 80;
const LOW_FREQ = 20.0;
const HIGH_FREQ = 7600.0;
const PREEMPHASIS = 0.97;
const LOG_FLOOR = 1.1920928955078125e-7;
const BLANK_ID = 0;
const UNK_ID = 2;
const CONTEXT_SIZE = 2;
const WORD_ASSIGN_MAX_DURATION_SECONDS = 0.40;

let runtimeReady = false;
let webgpuRuntimeLoaded = false;
let fftTables = null;
let asrWindow = null;
let asrMelBank = null;
const recognizers = new Map();

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function reportError(id, error) {
  post("error", {
    id,
    message: error?.message || String(error),
    stack: error?.stack || "",
  });
}

try {
  importScripts(`${ORT_BASE}ort.wasm.min.js`);
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.proxy = false;
  runtimeReady = true;
  post("runtime-ready");
} catch (error) {
  reportError(null, error);
}

function ensureWebGpuRuntimeLoaded() {
  if (webgpuRuntimeLoaded) return true;
  if (!self.navigator?.gpu) return false;
  try {
    importScripts(`${ORT_BASE}ort.webgpu.min.js`);
    ort.env.wasm.wasmPaths = ORT_BASE;
    ort.env.wasm.proxy = false;
    webgpuRuntimeLoaded = true;
    return true;
  } catch (error) {
    post("log", { message: `WebGPU ORT runtime load failed; using WASM fallback: ${error.message || String(error)}` });
    return false;
  }
}

function boundedNumber(value, fallback, min, max) {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function safeName(value) {
  return String(value || "model").replace(/[^a-zA-Z0-9_.-]/g, "_");
}

function decodeUtf8(buffer) {
  if (typeof buffer === "string") return buffer;
  return new TextDecoder("utf-8").decode(buffer);
}

function parseTokens(buffer) {
  const id2token = [];
  const pieceToId = new Map();
  for (const line of decodeUtf8(buffer).split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const parts = trimmed.split(/\s+/);
    if (parts.length < 2) continue;
    const id = Number.parseInt(parts[parts.length - 1], 10);
    if (!Number.isFinite(id)) continue;
    const piece = parts.slice(0, -1).join(" ");
    id2token[id] = piece;
    pieceToId.set(piece, id);
  }
  return { id2token, pieceToId };
}

function parseBpeVocab(text, pieceToId) {
  const pieces = new Map();
  let maxPieceLength = 1;
  for (const rawLine of String(text || "").split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) continue;
    const tab = line.lastIndexOf("\t");
    if (tab <= 0) continue;
    const piece = line.slice(0, tab);
    const score = Number.parseFloat(line.slice(tab + 1));
    if (!Number.isFinite(score)) continue;
    const id = pieceToId.get(piece);
    if (id === undefined) continue;
    pieces.set(piece, { id, score });
    maxPieceLength = Math.max(maxPieceLength, piece.length);
  }
  return { pieces, maxPieceLength };
}

function normalizeHotwordText(value) {
  return String(value || "")
    .normalize("NFC")
    .replace(/\s+/g, " ")
    .trim()
    .toLocaleUpperCase("vi-VN");
}

function parseHotwordsText(text, defaultScore = 1.5) {
  const items = [];
  for (const rawLine of String(text || "").split(/\r?\n/)) {
    let line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;

    let score = defaultScore;
    const scoreMatch = line.match(/^(.*?)(?:\s*:([+-]?(?:\d+(?:\.\d*)?|\.\d+)))$/);
    if (scoreMatch) {
      line = scoreMatch[1].trim();
      const parsed = Number.parseFloat(scoreMatch[2]);
      if (Number.isFinite(parsed)) score = parsed;
    }

    const phrase = normalizeHotwordText(line);
    if (phrase) items.push({ phrase, score });
  }
  return items;
}

function sentencePieceEncode(text, tokenizer) {
  const normalized = normalizeHotwordText(text)
    .normalize("NFKC")
    .replace(/\s+/g, " ")
    .trim();
  if (!normalized) return [];

  const input = `\u2581${normalized.replace(/ /g, "\u2581")}`;
  const n = input.length;
  const best = new Float64Array(n + 1);
  const prev = new Int32Array(n + 1);
  const prevId = new Int32Array(n + 1);
  best.fill(Number.NEGATIVE_INFINITY);
  prev.fill(-1);
  prevId.fill(UNK_ID);
  best[0] = 0;

  for (let i = 0; i < n; i += 1) {
    if (!Number.isFinite(best[i])) continue;
    let matched = false;
    const maxLen = Math.min(tokenizer.maxPieceLength, n - i);
    for (let len = 1; len <= maxLen; len += 1) {
      const piece = input.slice(i, i + len);
      const entry = tokenizer.pieces.get(piece);
      if (!entry) continue;
      matched = true;
      const j = i + len;
      const candidate = best[i] + entry.score;
      if (candidate > best[j]) {
        best[j] = candidate;
        prev[j] = i;
        prevId[j] = entry.id;
      }
    }

    if (!matched) {
      const codePoint = input.codePointAt(i);
      const len = codePoint > 0xffff ? 2 : 1;
      const j = Math.min(n, i + len);
      const candidate = best[i] - 20.0;
      if (candidate > best[j]) {
        best[j] = candidate;
        prev[j] = i;
        prevId[j] = UNK_ID;
      }
    }
  }

  if (!Number.isFinite(best[n])) return [];
  const ids = [];
  for (let pos = n; pos > 0;) {
    ids.push(prevId[pos]);
    pos = prev[pos];
    if (pos < 0) return [];
  }
  ids.reverse();
  return ids;
}

class ContextState {
  constructor(token = -1, tokenScore = 0.0, nodeScore = 0.0, outputScore = 0.0, isEnd = false) {
    this.token = token;
    this.tokenScore = tokenScore;
    this.nodeScore = nodeScore;
    this.outputScore = outputScore;
    this.isEnd = isEnd;
    this.next = new Map();
    this.fail = null;
    this.output = null;
  }
}

class ContextGraph {
  constructor() {
    this.root = new ContextState(-1);
    this.root.fail = this.root;
    this.nPhrases = 0;
  }

  build(tokenSequences, scores) {
    for (let s = 0; s < tokenSequences.length; s += 1) {
      const seq = tokenSequences[s];
      const score = scores[s];
      if (!seq?.length) continue;
      let node = this.root;
      for (let j = 0; j < seq.length; j += 1) {
        const tid = seq[j];
        const isLast = j === seq.length - 1;
        if (!node.next.has(tid)) {
          node.next.set(
            tid,
            new ContextState(
              tid,
              score,
              node.nodeScore + score,
              isLast ? node.nodeScore + score : 0.0,
              isLast
            )
          );
        } else {
          const existing = node.next.get(tid);
          existing.tokenScore = Math.max(score, existing.tokenScore);
          existing.nodeScore = node.nodeScore + existing.tokenScore;
          if (isLast) {
            existing.isEnd = true;
            existing.outputScore = existing.nodeScore;
          } else if (existing.isEnd) {
            existing.outputScore = existing.nodeScore;
          }
        }
        node = node.next.get(tid);
      }
      this.nPhrases += 1;
    }
    this.fillFailOutput();
  }

  fillFailOutput() {
    const queue = [];
    for (const child of this.root.next.values()) {
      child.fail = this.root;
      queue.push(child);
    }

    for (let head = 0; head < queue.length; head += 1) {
      const current = queue[head];
      for (const [tid, child] of current.next.entries()) {
        let fail = current.fail;
        if (fail.next.has(tid)) {
          fail = fail.next.get(tid);
        } else {
          fail = fail.fail;
          while (!fail.next.has(tid)) {
            fail = fail.fail;
            if (fail.token === -1) break;
          }
          if (fail.next.has(tid)) fail = fail.next.get(tid);
        }

        child.fail = fail;
        let output = fail;
        while (!output.isEnd) {
          output = output.fail;
          if (output.token === -1) {
            output = null;
            break;
          }
        }
        child.output = output;
        if (output !== null) child.outputScore += output.outputScore;
        queue.push(child);
      }
    }
  }

  forwardOneStep(state, tokenId) {
    let node = null;
    let score = 0.0;
    if (state.next.has(tokenId)) {
      node = state.next.get(tokenId);
      score = node.tokenScore;
    } else {
      node = state.fail;
      while (!node.next.has(tokenId)) {
        node = node.fail;
        if (node.token === -1) break;
      }
      if (node.next.has(tokenId)) node = node.next.get(tokenId);
      score = node.nodeScore - state.nodeScore;
    }

    if (node.outputScore !== 0) {
      let outputScore = node.nodeScore;
      if (!node.isEnd && node.output !== null) outputScore = node.output.nodeScore;
      return [score + outputScore - node.nodeScore, this.root];
    }

    return [score, node];
  }

  finalize(state) {
    return -state.nodeScore;
  }
}

function buildContextGraph(hotwordsText, tokenizer, defaultScore) {
  const hotwords = parseHotwordsText(hotwordsText, defaultScore);
  if (!hotwords.length) return null;

  const tokenSequences = [];
  const scores = [];
  let skipped = 0;
  for (const item of hotwords) {
    const ids = sentencePieceEncode(item.phrase, tokenizer);
    if (!ids.length) {
      skipped += 1;
      continue;
    }
    tokenSequences.push(ids);
    scores.push(item.score);
  }

  if (!tokenSequences.length) return null;
  const graph = new ContextGraph();
  graph.build(tokenSequences, scores);
  post("log", {
    message: `PureORT hotword context graph: ${graph.nPhrases} phrases (skipped ${skipped}).`,
  });
  return graph;
}

function hzToMel(freq) {
  return 1127.0 * Math.log(1.0 + freq / 700.0);
}

function melToHz(mel) {
  return 700.0 * (Math.exp(mel / 1127.0) - 1.0);
}

function getAsrWindow() {
  if (asrWindow) return asrWindow;
  asrWindow = new Float32Array(FRAME_LENGTH);
  for (let i = 0; i < FRAME_LENGTH; i += 1) {
    const hann = 0.5 - 0.5 * Math.cos((2.0 * Math.PI * i) / (FRAME_LENGTH - 1));
    asrWindow[i] = Math.pow(hann, 0.85);
  }
  return asrWindow;
}

function getAsrMelBank() {
  if (asrMelBank) return asrMelBank;
  const lowMel = hzToMel(LOW_FREQ);
  const highMel = hzToMel(HIGH_FREQ);
  const melDelta = (highMel - lowMel) / (NUM_MEL_BINS + 1);
  const centers = new Float64Array(NUM_MEL_BINS + 2);
  for (let i = 0; i < centers.length; i += 1) {
    centers[i] = melToHz(lowMel + i * melDelta);
  }

  const bins = N_FFT / 2 + 1;
  asrMelBank = Array.from({ length: NUM_MEL_BINS }, () => new Float32Array(bins));
  for (let bin = 0; bin < bins; bin += 1) {
    const freq = (bin * SAMPLE_RATE) / N_FFT;
    for (let mel = 0; mel < NUM_MEL_BINS; mel += 1) {
      const left = centers[mel];
      const center = centers[mel + 1];
      const right = centers[mel + 2];
      let weight = 0;
      if (freq > left && freq <= center) {
        weight = (freq - left) / Math.max(center - left, 1e-12);
      } else if (freq > center && freq < right) {
        weight = (right - freq) / Math.max(right - center, 1e-12);
      }
      asrMelBank[mel][bin] = weight;
    }
  }
  return asrMelBank;
}

function getFftTables(size) {
  if (fftTables && fftTables.size === size) return fftTables;
  const levels = Math.log2(size);
  const reverse = new Uint16Array(size);
  for (let i = 0; i < size; i += 1) {
    let value = i;
    let result = 0;
    for (let bit = 0; bit < levels; bit += 1) {
      result = (result << 1) | (value & 1);
      value >>= 1;
    }
    reverse[i] = result;
  }

  const cos = new Float64Array(size / 2);
  const sin = new Float64Array(size / 2);
  for (let i = 0; i < size / 2; i += 1) {
    const angle = (-2.0 * Math.PI * i) / size;
    cos[i] = Math.cos(angle);
    sin[i] = Math.sin(angle);
  }

  fftTables = { size, reverse, cos, sin };
  return fftTables;
}

function fftInPlace(real, imag) {
  const n = real.length;
  const tables = getFftTables(n);
  for (let i = 0; i < n; i += 1) {
    const j = tables.reverse[i];
    if (j <= i) continue;
    const tr = real[i];
    const ti = imag[i];
    real[i] = real[j];
    imag[i] = imag[j];
    real[j] = tr;
    imag[j] = ti;
  }

  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const tableStep = n / size;
    for (let start = 0; start < n; start += size) {
      for (let j = 0; j < half; j += 1) {
        const k = j * tableStep;
        const wr = tables.cos[k];
        const wi = tables.sin[k];
        const even = start + j;
        const odd = even + half;
        const tr = wr * real[odd] - wi * imag[odd];
        const ti = wr * imag[odd] + wi * real[odd];
        real[odd] = real[even] - tr;
        imag[odd] = imag[even] - ti;
        real[even] += tr;
        imag[even] += ti;
      }
    }
  }
}

function reflectIndex(index, length) {
  if (length <= 1) return 0;
  let value = index;
  while (value < 0 || value >= length) {
    if (value < 0) value = -value - 1;
    else value = 2 * length - 1 - value;
  }
  return value;
}

function computeFbank(samples) {
  const n = samples?.length || 0;
  if (n <= 0) return { data: new Float32Array(0), frames: 0 };
  const numFrames = Math.floor((n + FRAME_SHIFT / 2) / FRAME_SHIFT);
  if (numFrames <= 0) return { data: new Float32Array(0), frames: 0 };

  const window = getAsrWindow();
  const melBank = getAsrMelBank();
  const frame = new Float64Array(FRAME_LENGTH);
  const real = new Float64Array(N_FFT);
  const imag = new Float64Array(N_FFT);
  const power = new Float64Array(N_FFT / 2 + 1);
  const features = new Float32Array(numFrames * NUM_MEL_BINS);

  for (let f = 0; f < numFrames; f += 1) {
    const start = f * FRAME_SHIFT + Math.floor(FRAME_SHIFT / 2) - Math.floor(FRAME_LENGTH / 2);
    let mean = 0;
    for (let i = 0; i < FRAME_LENGTH; i += 1) {
      const sample = samples[reflectIndex(start + i, n)] || 0;
      frame[i] = sample;
      mean += sample;
    }
    mean /= FRAME_LENGTH;

    real.fill(0);
    imag.fill(0);
    for (let i = 0; i < FRAME_LENGTH; i += 1) {
      const current = frame[i] - mean;
      const previous = i === 0 ? frame[0] - mean : frame[i - 1] - mean;
      real[i] = (current - PREEMPHASIS * previous) * window[i];
    }

    fftInPlace(real, imag);
    for (let i = 0; i < power.length; i += 1) {
      power[i] = real[i] * real[i] + imag[i] * imag[i];
    }

    const outOffset = f * NUM_MEL_BINS;
    for (let mel = 0; mel < NUM_MEL_BINS; mel += 1) {
      const weights = melBank[mel];
      let energy = 0;
      for (let bin = 0; bin < weights.length; bin += 1) {
        energy += power[bin] * weights[bin];
      }
      features[outOffset + mel] = Math.log(Math.max(energy, LOG_FLOOR));
    }
  }

  return { data: features, frames: numFrames };
}

function makeInt64Tensor(values, dims) {
  const data = new BigInt64Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    data[i] = BigInt(values[i]);
  }
  return new ort.Tensor("int64", data, dims);
}

function logAdd(a, b) {
  let x = a;
  let y = b;
  if (x < y) {
    x = b;
    y = a;
  }
  const diff = y - x;
  return diff < -36.0 ? x : x + Math.log1p(Math.exp(diff));
}

function insertTopK(scores, indexes, score, index) {
  const k = scores.length;
  if (score <= scores[k - 1]) return;
  let pos = k - 1;
  while (pos > 0 && score > scores[pos - 1]) {
    scores[pos] = scores[pos - 1];
    indexes[pos] = indexes[pos - 1];
    pos -= 1;
  }
  scores[pos] = score;
  indexes[pos] = index;
}

function computeTokenEntropy(logits, base, count) {
  const maxEntropy = count > 1 ? Math.log(count) : 1.0;
  const alpha = 1.0 / 3.0;
  const tsallisMax = count > 1
    ? (1.0 / (alpha - 1.0)) * (1.0 - Math.pow(count, 1.0 - alpha))
    : 1.0;
  let maxLogit = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < count; i += 1) {
    const value = logits[base + i];
    if (value > maxLogit) maxLogit = value;
  }

  let sum = 0.0;
  for (let i = 0; i < count; i += 1) {
    sum += Math.exp(logits[base + i] - maxLogit);
  }
  const denom = Math.max(sum, 1e-30);
  let entropy = 0.0;
  let tsallisPowSum = 0.0;
  let top1 = 0.0;
  let top2 = 0.0;
  for (let i = 0; i < count; i += 1) {
    const p = Math.exp(logits[base + i] - maxLogit) / denom;
    entropy -= p * Math.log(p + 1e-30);
    tsallisPowSum += Math.pow(p, alpha);
    if (p > top1) {
      top2 = top1;
      top1 = p;
    } else if (p > top2) {
      top2 = p;
    }
  }

  const tsallis = (1.0 / (alpha - 1.0)) * (1.0 - tsallisPowSum);
  return {
    tsallis_norm: Number((tsallis / Math.max(tsallisMax, 1e-12)).toFixed(4)),
    margin: Number((top1 - top2).toFixed(4)),
    entropy_norm: Number((entropy / Math.max(maxEntropy, 1e-12)).toFixed(4)),
    top1_prob: top1,
  };
}

async function runDecoder(recognizer, ctxs, timings = null) {
  const count = ctxs.length;
  const input = new BigInt64Array(count * CONTEXT_SIZE);
  for (let i = 0; i < count; i += 1) {
    input[i * CONTEXT_SIZE] = BigInt(ctxs[i][0]);
    input[i * CONTEXT_SIZE + 1] = BigInt(ctxs[i][1]);
  }
  const started = performance.now();
  const outputs = await recognizer.decoder.run({
    [recognizer.decoder.inputNames[0]]: new ort.Tensor("int64", input, [count, CONTEXT_SIZE]),
  });
  if (timings) timings.decoderMs += performance.now() - started;
  return outputs[recognizer.decoder.outputNames[0]];
}

async function ortBeamSearch(recognizer, features, beamSize = 8) {
  const timings = {
    encoderMs: 0,
    decoderMs: 0,
    joinerMs: 0,
    searchMs: 0,
  };
  const searchStarted = performance.now();
  const x = new ort.Tensor("float32", features.data, [1, features.frames, NUM_MEL_BINS]);
  const xLens = makeInt64Tensor([features.frames], [1]);
  const encoderStarted = performance.now();
  const encOutputs = await recognizer.encoder.run({
    [recognizer.encoder.inputNames[0]]: x,
    [recognizer.encoder.inputNames[1]]: xLens,
  });
  timings.encoderMs += performance.now() - encoderStarted;
  const encOut = encOutputs[recognizer.encoder.outputNames[0]];
  const encLens = encOutputs[recognizer.encoder.outputNames[1]];
  const T = Number(encLens.data[0]);
  if (!T) return { tokenIds: [], frames: [], tokenLogProbs: [], T: 0, timings };

  const encData = encOut.data;
  const encDims = encOut.dims;
  const encDim = encDims[2];
  const initYs = [-1, BLANK_ID];
  const initCtx = [BLANK_ID, BLANK_ID];
  if (!recognizer.decCache.has("0,0")) {
    const dec = await runDecoder(recognizer, [initCtx], timings);
    recognizer.decCache.set("0,0", dec.data.slice(0, dec.dims[1]));
  }

  let hyps = new Map();
  hyps.set(initYs.join(","), {
    ys: initYs,
    lp: 0.0,
    frames: [],
    tokenLogProbs: [],
    tokenEntropies: [],
    contextState: recognizer.contextGraph ? recognizer.contextGraph.root : null,
  });

  for (let t = 0; t < T; t += 1) {
    const prev = Array.from(hyps.values());
    const B = prev.length;
    const ctxs = [];
    const missingIdx = [];
    const decRows = new Array(B);

    for (let i = 0; i < B; i += 1) {
      const ys = prev[i].ys;
      const ctx = [
        Math.max(0, ys[ys.length - 2]),
        Math.max(0, ys[ys.length - 1]),
      ];
      const key = ctx.join(",");
      const cached = recognizer.decCache.get(key);
      if (cached) {
        decRows[i] = cached;
      } else {
        ctxs.push(ctx);
        missingIdx.push(i);
      }
    }

    if (ctxs.length) {
      const dec = await runDecoder(recognizer, ctxs, timings);
      const decDim = dec.dims[1];
      for (let j = 0; j < ctxs.length; j += 1) {
        const row = dec.data.slice(j * decDim, (j + 1) * decDim);
        recognizer.decCache.set(ctxs[j].join(","), row);
        decRows[missingIdx[j]] = row;
      }
    }

    const decDim = decRows[0].length;
    const encBuf = new Float32Array(B * encDim);
    const decBuf = new Float32Array(B * decDim);
    const encOffset = t * encDim;
    const encFrame = encData.subarray(encOffset, encOffset + encDim);
    for (let i = 0; i < B; i += 1) {
      encBuf.set(encFrame, i * encDim);
      decBuf.set(decRows[i], i * decDim);
    }

    const joinerStarted = performance.now();
    const joinerOutputs = await recognizer.joiner.run({
      [recognizer.joiner.inputNames[0]]: new ort.Tensor("float32", encBuf, [B, encDim]),
      [recognizer.joiner.inputNames[1]]: new ort.Tensor("float32", decBuf, [B, decDim]),
    });
    timings.joinerMs += performance.now() - joinerStarted;
    const logits = joinerOutputs[recognizer.joiner.outputNames[0]];
    const V = recognizer.vocabSize || logits.dims[1];
    const beam = Math.min(beamSize, B * V);
    const topScores = new Float64Array(beam);
    const topIndexes = new Int32Array(beam);
    const rowLogDen = new Float64Array(B);
    topScores.fill(Number.NEGATIVE_INFINITY);
    topIndexes.fill(-1);

    for (let i = 0; i < B; i += 1) {
      const rowOffset = i * V;
      let maxLogit = Number.NEGATIVE_INFINITY;
      for (let v = 0; v < V; v += 1) {
        const value = logits.data[rowOffset + v];
        if (value > maxLogit) maxLogit = value;
      }
      let sumExp = 0.0;
      for (let v = 0; v < V; v += 1) {
        sumExp += Math.exp(logits.data[rowOffset + v] - maxLogit);
      }
      rowLogDen[i] = maxLogit + Math.log(sumExp);
      const prevLp = prev[i].lp;
      for (let v = 0; v < V; v += 1) {
        insertTopK(topScores, topIndexes, logits.data[rowOffset + v] - rowLogDen[i] + prevLp, i * V + v);
      }
    }

    const newHyps = new Map();
    for (let k = 0; k < beam; k += 1) {
      const flatIdx = topIndexes[k];
      if (flatIdx < 0) continue;
      const hi = Math.floor(flatIdx / V);
      const token = flatIdx % V;
      let score = topScores[k];
      const source = prev[hi];
      let ys = source.ys;
      let frames = source.frames;
      let tokenLogProbs = source.tokenLogProbs;
      let tokenEntropies = source.tokenEntropies;
      let contextState = source.contextState;

      if (token !== BLANK_ID) {
        const rowBase = hi * V;
        ys = source.ys.concat([token]);
        frames = source.frames.concat([t]);
        tokenLogProbs = source.tokenLogProbs.concat([logits.data[rowBase + token] - rowLogDen[hi]]);
        tokenEntropies = source.tokenEntropies.concat([computeTokenEntropy(logits.data, rowBase, V)]);
        if (recognizer.contextGraph && contextState && token !== UNK_ID) {
          const advanced = recognizer.contextGraph.forwardOneStep(contextState, token);
          score += advanced[0];
          contextState = advanced[1];
        }
      }

      const key = ys.join(",");
      const old = newHyps.get(key);
      if (old) {
        old.lp = logAdd(old.lp, score);
      } else {
        newHyps.set(key, { ys, lp: score, frames, tokenLogProbs, tokenEntropies, contextState });
      }
    }
    hyps = newHyps;
  }

  if (recognizer.contextGraph) {
    for (const hyp of hyps.values()) {
      if (hyp.contextState) hyp.lp += recognizer.contextGraph.finalize(hyp.contextState);
    }
  }

  let best = null;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const hyp of hyps.values()) {
    const score = hyp.lp / Math.max(hyp.ys.length, 1);
    if (score > bestScore) {
      bestScore = score;
      best = hyp;
    }
  }

  const tokenIds = (best?.ys || []).slice(CONTEXT_SIZE).filter((token) => token > 0);
  timings.searchMs = Math.max(0, performance.now() - searchStarted - timings.encoderMs - timings.decoderMs - timings.joinerMs);
  return {
    tokenIds,
    frames: best?.frames || [],
    tokenLogProbs: best?.tokenLogProbs || [],
    tokenEntropies: best?.tokenEntropies || [],
    T,
    timings,
  };
}

function finalizeWordEntropy(word) {
  word.prob = word.probs.reduce((sum, value) => sum + value, 0) / word.probs.length;
  delete word.probs;
  const ents = word._ents || [];
  delete word._ents;
  if (!ents.length) {
    word.tsallis_max = null;
    word.margin_min = null;
    word.entropy_norm = null;
    word._conf = null;
    return;
  }
  word.tsallis_max = Number(Math.max(...ents.map((item) => item.tsallis_norm)).toFixed(4));
  word.margin_min = Number(Math.min(...ents.map((item) => item.margin)).toFixed(4));
  word.entropy_norm = Number((ents.reduce((sum, item) => sum + item.entropy_norm, 0) / ents.length).toFixed(4));
  word._conf = Number((ents.reduce((sum, item) => sum + item.margin * (1.0 - item.tsallis_norm), 0) / ents.length).toFixed(4));
}

function tokensToWords(recognizer, search, sampleCount, timeOffset = 0) {
  const { tokenIds, frames, tokenLogProbs, tokenEntropies, T } = search;
  if (!tokenIds.length || !frames.length || !T) return [];

  const chunkDur = sampleCount / SAMPLE_RATE;
  const ts = frames.map((frame) => (frame / T) * chunkDur);
  if (!ts.length) return [];
  const avgBpeDur = ts.length >= 2 ? (ts[ts.length - 1] - ts[0]) / (ts.length - 1) : 0.08;
  const bpeWords = [];
  for (let j = 0; j < tokenIds.length; j += 1) {
    const token = (recognizer.id2token[tokenIds[j]] || "").toLocaleLowerCase("vi-VN");
    const localStart = ts[j] || 0;
    const localEnd = j < ts.length - 1 ? ts[j + 1] : localStart + avgBpeDur;
    bpeWords.push({
      text: token,
      start: localStart + timeOffset,
      end: localEnd + timeOffset,
      local_start: localStart,
      local_end: localEnd,
      prob: Math.exp(tokenLogProbs[j] || 0),
      _ent: tokenEntropies[j] || null,
    });
  }

  const merged = [];
  let current = null;
  for (const tokenInfo of bpeWords) {
    const token = tokenInfo.text;
    if (token.startsWith(" ") || token.startsWith("\u2581")) {
      if (current) merged.push(current);
      current = {
        text: token.replace(/^[ \u2581]+/u, ""),
        start: tokenInfo.start,
        end: tokenInfo.end,
        local_start: tokenInfo.local_start,
        local_end: tokenInfo.local_end,
        last_bpe_start: tokenInfo.start,
        probs: [tokenInfo.prob],
        _ents: tokenInfo._ent ? [tokenInfo._ent] : [],
      };
    } else if (current) {
      current.text += token;
      current.end = tokenInfo.end;
      current.local_end = tokenInfo.local_end;
      current.last_bpe_start = tokenInfo.start;
      current.probs.push(tokenInfo.prob);
      if (tokenInfo._ent) current._ents.push(tokenInfo._ent);
    } else {
      current = {
        text: token,
        start: tokenInfo.start,
        end: tokenInfo.end,
        local_start: tokenInfo.local_start,
        local_end: tokenInfo.local_end,
        last_bpe_start: tokenInfo.start,
        probs: [tokenInfo.prob],
        _ents: tokenInfo._ent ? [tokenInfo._ent] : [],
      };
    }
  }
  if (current) merged.push(current);

  for (let i = 0; i < merged.length; i += 1) {
    const word = merged[i];
    let estimatedEnd = word.last_bpe_start + avgBpeDur;
    if (i < merged.length - 1) estimatedEnd = Math.min(estimatedEnd, merged[i + 1].start);
    estimatedEnd = Math.min(estimatedEnd, word.start + WORD_ASSIGN_MAX_DURATION_SECONDS);
    if (estimatedEnd <= word.start) estimatedEnd = word.start + Math.min(0.01, WORD_ASSIGN_MAX_DURATION_SECONDS);
    word.end = estimatedEnd;
    word.local_end = estimatedEnd - timeOffset;
    finalizeWordEntropy(word);
    delete word.last_bpe_start;
  }

  return merged.filter((word) => word.text);
}

async function initRecognizer(files, options = {}) {
  if (!runtimeReady) throw new Error("ONNX Runtime Web did not load.");
  const modelId = options.modelId || "unknown";
  if (recognizers.has(modelId)) return;

  const numThreads = boundedNumber(options.numThreads, 1, 1, 8);
  ort.env.wasm.numThreads = numThreads;
  const providerMode = options.providerMode === "webgpu" ? "webgpu" : "wasm";
  const webgpuAvailable = providerMode === "webgpu" && ensureWebGpuRuntimeLoaded();
  ort.env.wasm.numThreads = numThreads;
  const providerForStage = (stage) => {
    const requested = options.stageProviders?.[stage] || providerMode;
    return requested === "webgpu" && webgpuAvailable ? "webgpu" : "wasm";
  };
  const sessionOptionsFor = (stage) => {
    const provider = providerForStage(stage);
    return {
      executionProviders: provider === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
      graphOptimizationLevel: "all",
      executionMode: "sequential",
    };
  };

  let encoder = null;
  let decoder = null;
  let joiner = null;
  if (providerMode === "webgpu") {
    // ORT WebGPU can reject concurrent session creation in the same browser process.
    // Create ASR sessions one by one so benchmark WebGPU runs are repeatable.
    encoder = await ort.InferenceSession.create(files.encoder, sessionOptionsFor("encoder"));
    decoder = await ort.InferenceSession.create(files.decoder, sessionOptionsFor("decoder"));
    joiner = await ort.InferenceSession.create(files.joiner, sessionOptionsFor("joiner"));
  } else {
    [encoder, decoder, joiner] = await Promise.all([
      ort.InferenceSession.create(files.encoder, sessionOptionsFor("encoder")),
      ort.InferenceSession.create(files.decoder, sessionOptionsFor("decoder")),
      ort.InferenceSession.create(files.joiner, sessionOptionsFor("joiner")),
    ]);
  }
  const providers = {
    encoder: providerForStage("encoder"),
    decoder: providerForStage("decoder"),
    joiner: providerForStage("joiner"),
  };

  const parsedTokens = parseTokens(files.tokens);
  const tokenizer = files.bpeVocab
    ? parseBpeVocab(decodeUtf8(files.bpeVocab), parsedTokens.pieceToId)
    : null;
  const contextGraph = tokenizer
    ? buildContextGraph(options.hotwordsText || "", tokenizer, Number(options.hotwordsScore) || 1.5)
    : null;
  recognizers.set(modelId, {
    id: safeName(modelId),
    encoder,
    decoder,
    joiner,
    id2token: parsedTokens.id2token,
    vocabSize: parsedTokens.id2token.length,
    decCache: new Map(),
    contextGraph,
    maxActivePaths: boundedNumber(options.maxActivePaths, 8, 1, 16),
    providers,
  });

  post("log", {
    message: `Initialized PureORT ${options.modelLabel || modelId} with ${numThreads} thread(s), providers encoder=${providers.encoder}, decoder=${providers.decoder}, joiner=${providers.joiner}.`,
  });
}

function getRecognizerForDecode(modelId = null) {
  if (modelId) {
    const recognizer = recognizers.get(modelId);
    if (!recognizer) {
      throw new Error(`PureORT recognizer is not initialized for ${modelId}.`);
    }
    return recognizer;
  } else if (recognizers.size === 1) {
    return recognizers.values().next().value;
  }
  throw new Error(`PureORT recognizer is not initialized${modelId ? ` for ${modelId}` : ""}.`);
}

async function decodeWithFeatures(recognizer, features, sampleCount, timeOffset = 0, fbankMs = 0) {
  if (!features.frames) {
    return { text: "", words: [], backend: "pure_ort", frames: 0 };
  }
  const search = await ortBeamSearch(recognizer, features, recognizer.maxActivePaths);
  const words = tokensToWords(recognizer, search, sampleCount, Number(timeOffset) || 0);
  return {
    text: words.map((word) => word.text).join(" "),
    words,
    backend: "pure_ort",
    frames: search.T,
    tokens: search.tokenIds.length,
    providers: recognizer.providers,
    timings: {
      fbankMs,
      ...(search.timings || {}),
    },
  };
}

async function decode(samples, modelId, timeOffset = 0) {
  const recognizer = getRecognizerForDecode(modelId);
  const fbankStarted = performance.now();
  const features = computeFbank(samples);
  const fbankMs = performance.now() - fbankStarted;
  return decodeWithFeatures(recognizer, features, samples.length, timeOffset, fbankMs);
}

async function decodePair(samples, primaryModelId, secondaryModelId, timeOffset = 0) {
  const primary = getRecognizerForDecode(primaryModelId);
  const secondary = getRecognizerForDecode(secondaryModelId);
  const fbankStarted = performance.now();
  const features = computeFbank(samples);
  const fbankMs = performance.now() - fbankStarted;
  const primaryResult = await decodeWithFeatures(primary, features, samples.length, timeOffset, fbankMs);
  const secondaryResult = await decodeWithFeatures(secondary, features, samples.length, timeOffset, 0);
  return {
    primary: primaryResult,
    secondary: secondaryResult,
    backend: "pure_ort",
    sharedFbank: true,
    timings: { fbankMs },
  };
}

self.onmessage = async (event) => {
  const { id, type } = event.data || {};
  try {
    if (type === "init") {
      await initRecognizer(event.data.files, {
        modelId: event.data.modelId,
        modelLabel: event.data.modelLabel,
        numThreads: event.data.numThreads,
        maxActivePaths: event.data.maxActivePaths,
        hotwordsText: event.data.hotwordsText,
        hotwordsScore: event.data.hotwordsScore,
        providerMode: event.data.providerMode,
        stageProviders: event.data.stageProviders,
      });
      post("ready", { id });
      return;
    }

    if (type === "decode") {
      const result = await decode(event.data.samples, event.data.modelId, event.data.timeOffset);
      post("decoded", { id, result });
      return;
    }

    if (type === "decode_pair") {
      const result = await decodePair(
        event.data.samples,
        event.data.primaryModelId,
        event.data.secondaryModelId,
        event.data.timeOffset
      );
      post("decoded", { id, result });
      return;
    }

    throw new Error(`Unknown PureORT ASR worker message: ${type}`);
  } catch (error) {
    reportError(id, error);
  }
};
