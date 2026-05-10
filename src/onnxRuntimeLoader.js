const LOCAL_ORT_SCRIPT = "./assets/onnxruntime-web/ort.min.js";
const CDN_ORT_SCRIPT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js";

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-ort-loader=\"${src}\"]`);
    if (existing) {
      existing.addEventListener("load", resolve, { once: true });
      existing.addEventListener("error", reject, { once: true });
      return;
    }

    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.dataset.ortLoader = src;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Failed to load ONNX Runtime script: ${src}`));
    document.head.appendChild(script);
  });
}

export async function loadOnnxRuntime() {
  if (window.ort) {
    return window.ort;
  }

  try {
    await loadScript(LOCAL_ORT_SCRIPT);
  } catch (localError) {
    console.warn(localError);
    await loadScript(CDN_ORT_SCRIPT);
  }

  if (!window.ort) {
    throw new Error("ONNX Runtime did not expose window.ort.");
  }

  window.ort.env.wasm.numThreads = 1;
  return window.ort;
}
