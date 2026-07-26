import fs from "node:fs";
import process from "node:process";
import { chromium } from "playwright";

const baseUrl = process.env.WEB_BASE_URL ?? "http://127.0.0.1:4173/";
const BASE_URL = new URL(baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`);
const RUN_MODE = process.argv[2] ?? "--all";
const TARGET_PNG_BASE64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACAQMAAABIeJ9nAAAAA1BMVEUg0GBTvd99AAAADElEQVQI12NgYGAAAAAEAAEnNCcKAAAAAElFTkSuQmCC";

function chromeExecutable() {
  const configured = process.env.CHROME_PATH;
  const candidates = [
    configured,
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    chromium.executablePath(),
  ].filter(Boolean);
  const executable = candidates.find((path) => fs.existsSync(path));
  if (!executable) {
    throw new Error(
      `Chrome executable not found; checked ${candidates.join(", ")}`,
    );
  }
  return executable;
}

async function launchBrowser() {
  const args = ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"];
  if (process.env.WEB_SMOKE_SOFTWARE_GPU === "1") {
    args.push(
      "--no-sandbox",
      "--use-angle=vulkan",
      "--enable-features=Vulkan",
      "--disable-vulkan-surface",
    );
  }
  return chromium.launch({
    headless: process.env.WEB_SMOKE_HEADED !== "1",
    executablePath: chromeExecutable(),
    args,
  });
}

function fixedConfig() {
  return {
    epochs: 1,
    repetitions: 1,
    report_interval: 1,
    batch_size: 1,
    pool_size: 1,
    particle_count: 64,
    step_min: 2,
    step_max: 2,
    tbptt_chunk_steps: 2,
    inject_seed_interval: 1,
    update_prob: 0.5,
    seed: 42,
    seed_scale: 0.2,
    seed_mode: "UniformCircle",
    brush_size: 0.1,
    per_parameter_grad_normalization: true,
    optimizer: {
      learning_rate: 0.0005,
      weight_decay: 0,
      grad_clip_norm: 0,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
    },
    scheduler_milestones: [],
    scheduler_gamma: 0.3,
  };
}

function adaptiveConfig() {
  return {
    target2d: {
      ...fixedConfig(),
      pool_size: 2,
      particle_count: 61,
      step_min: 4,
      step_max: 4,
      tbptt_chunk_steps: 2,
      brush_size: 0,
    },
    material: {
      reference_particle_count: 64,
      total_measure: 0.12566371,
      fine_bandwidth: 0.1,
      bandwidth_exponent: 0.5,
      max_initial_fine_units: 4,
    },
    topology: {
      enabled: true,
      start_step: 2,
      split_radius_scale: 1,
      merge_detail_scale: 0.01,
      interval_steps: 2,
      events_per_interval: 1,
    },
  };
}

async function runWorkerSmokes() {
  const browser = await launchBrowser();
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.stack ?? String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(message.text());
    }
  });
  await page.goto(new URL("smoke.html", BASE_URL).href, {
    waitUntil: "domcontentloaded",
    timeout: 120_000,
  });
  const result = await page.evaluate(
    async ({ fixed, adaptive, targetBase64 }) => {
      if (!navigator.gpu) {
        throw new Error("navigator.gpu is unavailable");
      }
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error("WebGPU adapter request failed");
      }
      const modelResponse = await fetch("./models/catalog/growing/lizard.bpk");
      if (!modelResponse.ok) {
        throw new Error(`catalog model fetch failed: ${modelResponse.status}`);
      }
      const modelBytes = new Uint8Array(await modelResponse.arrayBuffer());
      const targetBytes = Uint8Array.from(atob(targetBase64), (value) =>
        value.charCodeAt(0),
      );

      const run = (mode, config, jobId) =>
        new Promise((resolve, reject) => {
          const worker = new Worker("./training_worker.js", { type: "module" });
          const updates = [];
          const timeout = setTimeout(() => {
            worker.terminate();
            reject(new Error(`${mode} browser training timed out`));
          }, 240_000);
          worker.onerror = (event) => {
            clearTimeout(timeout);
            worker.terminate();
            reject(new Error(`${mode} worker failed: ${event.message}`));
          };
          worker.onmessage = ({ data }) => {
            if (data.type === "failed") {
              clearTimeout(timeout);
              worker.terminate();
              reject(new Error(`${mode} training failed: ${data.error}`));
              return;
            }
            if (data.type === "progress") {
              updates.push({
                step: data.step,
                loss: data.loss,
                gradNorm: data.gradNorm,
                modelBytes: data.modelBytes?.byteLength ?? 0,
              });
            }
            if (data.type === "finished") {
              clearTimeout(timeout);
              worker.terminate();
              resolve({
                mode,
                updates,
                finalModelBytes: data.modelBytes?.byteLength ?? 0,
              });
            }
          };
          worker.postMessage({
            type: "train",
            jobId,
            targetId: jobId,
            mode,
            targetBytes,
            modelBytes,
            configJson: JSON.stringify(config),
            snapshotIntervalSteps: 1,
            snapshotIntervalMs: 0,
          });
        });

      return {
        fixed: await run("fixed", fixed, 1),
        adaptive: await run("adaptive", adaptive, 2),
      };
    },
    {
      fixed: fixedConfig(),
      adaptive: adaptiveConfig(),
      targetBase64: TARGET_PNG_BASE64,
    },
  );
  await browser.close();
  if (errors.length > 0) {
    throw new Error(`browser worker console errors:\n${errors.join("\n")}`);
  }
  for (const run of [result.fixed, result.adaptive]) {
    if (run.updates.length < 2 || run.finalModelBytes < 100_000) {
      throw new Error(`${run.mode} browser training returned an incomplete run`);
    }
    for (const update of run.updates) {
      if (
        !Number.isFinite(update.loss) ||
        !Number.isFinite(update.gradNorm) ||
        update.modelBytes < 100_000
      ) {
        throw new Error(
          `${run.mode} browser training returned invalid progress: ${JSON.stringify(update)}`,
        );
      }
    }
  }
  console.log(JSON.stringify({ workerTraining: result }, null, 2));
}

async function runAppSmoke() {
  const browser = await launchBrowser();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  const fetched = new Map();
  page.on("pageerror", (error) => errors.push(error.stack ?? String(error)));
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(message.text());
    }
  });
  page.on("response", (response) => {
    const url = response.url();
    if (
      url.includes("/models/") ||
      url.includes("/artifacts/") ||
      url.endsWith("/training_worker.js") ||
      url.includes("/worker_pkg/")
    ) {
      fetched.set(url, response.status());
    }
  });
  const catalogResponse = page.waitForResponse(
    (response) =>
      response.url().endsWith("/models/catalog/growing/lizard.bpk") &&
      response.status() === 200,
    { timeout: 120_000 },
  );
  await page.goto(BASE_URL.href, {
    waitUntil: "domcontentloaded",
    timeout: 120_000,
  });
  await page.locator("#boot-status").waitFor({
    state: "detached",
    timeout: 120_000,
  });
  await catalogResponse;
  await page.waitForTimeout(5_000);

  const runInference = process.env.WEB_SMOKE_INFERENCE !== "0";
  const runUiTraining = process.env.WEB_SMOKE_UI_TRAINING !== "0";
  if (runInference || runUiTraining) {
    const chooserPromise = page.waitForEvent("filechooser", { timeout: 20_000 });
    await page.locator("canvas").click({ position: { x: 90, y: 225 } });
    const chooser = await chooserPromise;
    await chooser.setFiles("assets/reference_targets/lizard_upstream_120.png");
    await page.getByRole("button", { name: "Ok", exact: true }).click();
    await page.waitForTimeout(3_000);
  }

  if (runInference) {
    const dinoResponse = page.waitForResponse(
      (response) =>
        response.url().endsWith("/models/dino/dino_vits.mpk") &&
        response.status() === 200,
      { timeout: 240_000 },
    );
    await page.locator("canvas").click({ position: { x: 250, y: 240 } });
    await dinoResponse;
    await page.waitForTimeout(75_000);
  }

  if (runUiTraining) {
    const workerWasmResponse = page.waitForResponse(
      (response) =>
        response
          .url()
          .endsWith("/worker_pkg/burn_automata_web_worker_bg.wasm") &&
        response.status() === 200,
      { timeout: 120_000 },
    );
    await page.locator("canvas").click({ position: { x: 420, y: 240 } });
    await workerWasmResponse;
    await page.waitForTimeout(500);
    await page.locator("canvas").click({ position: { x: 420, y: 240 } });
    await page.waitForTimeout(500);
  }

  if (process.env.WEB_SMOKE_SCREENSHOT) {
    await page.screenshot({
      path: process.env.WEB_SMOKE_SCREENSHOT,
    });
  }
  await browser.close();
  if (errors.length > 0) {
    throw new Error(`browser app console errors:\n${errors.join("\n")}`);
  }
  console.log(
    JSON.stringify(
      { app: { baseUrl: BASE_URL.href, fetched: Object.fromEntries(fetched) } },
      null,
      2,
    ),
  );
}

if (!["--all", "--workers", "--app"].includes(RUN_MODE)) {
  throw new Error(`unsupported mode ${RUN_MODE}`);
}
if (RUN_MODE === "--all" || RUN_MODE === "--workers") {
  await runWorkerSmokes();
}
if (RUN_MODE === "--all" || RUN_MODE === "--app") {
  await runAppSmoke();
}
