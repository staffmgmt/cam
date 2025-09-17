class PCMChunker extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // samplesPerChunk is injected from main thread (B8 sets 160ms @16kHz = 2560 samples)
    this.samplesPerChunk = (options && options.processorOptions && options.processorOptions.samplesPerChunk) || 16000;
    this.buffer = new Float32Array(this.samplesPerChunk);
    this.offset = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const data = input[0];
      let i = 0;
      while (i < data.length) {
        const space = this.samplesPerChunk - this.offset;
        const toCopy = Math.min(space, data.length - i);
        this.buffer.set(data.subarray(i, i + toCopy), this.offset);
        this.offset += toCopy;
        i += toCopy;
        if (this.offset >= this.samplesPerChunk) {
          const out = new Int16Array(this.samplesPerChunk);
          for (let j = 0; j < this.samplesPerChunk; j++) {
            let s = this.buffer[j];
            if (s > 1) s = 1; else if (s < -1) s = -1;
            out[j] = s < 0 ? s * 32768 : s * 32767;
          }
          const buf = out.buffer;
          this.port.postMessage(buf, [buf]);
          this.offset = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor('pcm-chunker', PCMChunker);

// PCM player pulls Int16 buffers from a queue pushed via port messages and outputs Float32 samples.
class PCMPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.current = null;
    this.offset = 0;
    this.samplesPerBuffer = 0;
    this.port.onmessage = (e) => {
      const d = e.data;
      if (d instanceof ArrayBuffer) {
        this.queue.push(new Int16Array(d));
      } else if (d instanceof Int16Array) {
        this.queue.push(d);
      }
    };
  }
  process(_inputs, outputs) {
    const output = outputs[0][0];
    if (!output) return true;
    let i = 0;
    while (i < output.length) {
      if (!this.current) {
        this.current = this.queue.shift();
        this.offset = 0;
        if (!this.current) {
          // Fill rest with silence
          while (i < output.length) output[i++] = 0;
          break;
        }
      }
      const remain = this.current.length - this.offset;
      const needed = output.length - i;
      const toCopy = Math.min(remain, needed);
      for (let j = 0; j < toCopy; j++) {
        output[i + j] = this.current[this.offset + j] / 32768;
      }
      i += toCopy;
      this.offset += toCopy;
      if (this.offset >= this.current.length) {
        this.current = null;
      }
    }
    return true;
  }
}

registerProcessor('pcm-player', PCMPlayer);
