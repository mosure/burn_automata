let bindingsPromise;
let modulePromise;

async function workerBindings() {
  if (!bindingsPromise) {
    bindingsPromise = (async () => {
      const module = await import("./worker_pkg/burn_automata_web_worker.js");
      await module.default();
      return module;
    })();
  }
  return bindingsPromise;
}

async function workerModule() {
  if (!modulePromise) {
    modulePromise = (async () => {
      const module = await workerBindings();
      await module.initialize_worker_webgpu();
      return module;
    })();
  }
  return modulePromise;
}

self.onmessage = async ({ data }) => {
  if (data?.type === "probe") {
    try {
      await workerBindings();
      self.postMessage({ type: "probe-ready" });
    } catch (error) {
      self.postMessage({
        type: "probe-failed",
        error: error?.stack ?? error?.message ?? String(error),
      });
    }
    return;
  }
  if (data?.type !== "train") {
    return;
  }
  try {
    const module = await workerModule();
    await module.train_target_image(
      data.jobId,
      data.targetId,
      data.mode,
      data.targetBytes,
      data.modelBytes,
      data.configJson,
      data.snapshotIntervalSteps,
      data.snapshotIntervalMs,
    );
  } catch (error) {
    console.error(error);
    self.postMessage({
      type: "failed",
      jobId: data.jobId,
      targetId: data.targetId,
      error: error?.stack ?? error?.message ?? String(error),
    });
  }
};
