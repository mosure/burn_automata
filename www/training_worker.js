let modulePromise;

async function workerModule() {
  if (!modulePromise) {
    modulePromise = (async () => {
      const module = await import("./worker_pkg/burn_automata_web_worker.js");
      await module.default();
      await module.initialize_worker_webgpu();
      return module;
    })();
  }
  return modulePromise;
}

self.onmessage = async ({ data }) => {
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
