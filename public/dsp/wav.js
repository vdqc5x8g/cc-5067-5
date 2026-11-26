// 16-bit PCM WAV 编码（浏览器与 Node 同构，零依赖）。
// 导出的文件从样本 0 开始，包含 FIR 全部尾部，
// 不依赖任何播放器起点偏移来“隐藏”群延迟。

export function encodeWavBytes(channels, sampleRate) {
  const chs =
    Array.isArray(channels[0]) || channels[0] instanceof Float32Array || channels[0] instanceof Float64Array
      ? channels
      : [channels];
  const numChannels = chs.length;
  const numFrames = chs[0].length;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const dataSize = numFrames * blockAlign;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeAscii = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeAscii(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(8, 'WAVE');
  writeAscii(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeAscii(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < numFrames; i++) {
    for (let c = 0; c < numChannels; c++) {
      let s = chs[c][i];
      if (!Number.isFinite(s)) s = 0;
      s = Math.max(-1, Math.min(1, s));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }
  return new Uint8Array(buffer);
}

// Node 便捷封装
export function encodeWavBuffer(channels, sampleRate) {
  return Buffer.from(encodeWavBytes(channels, sampleRate));
}
