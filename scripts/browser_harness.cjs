const { spawn } = require('node:child_process');
const path = require('node:path');

const root = path.resolve(__dirname, '..');

function spawnTracked(command, args, options) {
  const child = spawn(command, args, {
    ...options,
    detached: process.platform !== 'win32',
    stdio: ['ignore', 'pipe', 'pipe']
  });
  let output = '';
  const append = chunk => { output = (output + chunk.toString()).slice(-20000); };
  child.stdout.on('data', append);
  child.stderr.on('data', append);
  return { child, getOutput: () => output };
}

async function stopProcess(child) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  const target = process.platform === 'win32' ? child.pid : -child.pid;
  try { process.kill(target, 'SIGTERM'); } catch (_) { return; }
  await Promise.race([
    new Promise(resolve => child.once('exit', resolve)),
    new Promise(resolve => setTimeout(resolve, 2000))
  ]);
  if (child.exitCode === null && child.signalCode === null) {
    try { process.kill(target, 'SIGKILL'); } catch (_) {}
  }
}

async function waitForDemo(frontend, backend) {
  for (let attempt = 0; attempt < 200; attempt += 1) {
    if (frontend.child.exitCode !== null || backend.child.exitCode !== null) {
      throw new Error(`A demo server exited early.\nFrontend:\n${frontend.getOutput()}\nBackend:\n${backend.getOutput()}`);
    }
    try {
      const [ui, api] = await Promise.all([
        fetch('http://127.0.0.1:5188/', { signal: AbortSignal.timeout(1000) }),
        fetch('http://127.0.0.1:8000/api/ping', { signal: AbortSignal.timeout(1000) })
      ]);
      if (ui.ok && api.ok) return;
    } catch (_) {}
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`Demo servers did not become ready.\nFrontend:\n${frontend.getOutput()}\nBackend:\n${backend.getOutput()}`);
}

async function startDemoServers() {
  const backend = spawnTracked(process.execPath, ['-r', 'ts-node/register/transpile-only', 'src/index.ts'], {
    cwd: path.join(root, 'backend'),
    env: { ...process.env, PORT: '8000', JWT_SECRET: 'browser-fixture-secret', SKIP_DB: '1' }
  });
  const frontend = spawnTracked(process.execPath, [path.join(root, 'frontend/node_modules/vite/bin/vite.js'), 'preview', '--host', '127.0.0.1', '--port', '5188'], {
    cwd: path.join(root, 'frontend'),
    env: process.env
  });
  try {
    await waitForDemo(frontend, backend);
  } catch (error) {
    await Promise.all([stopProcess(frontend.child), stopProcess(backend.child)]);
    throw error;
  }
  return {
    async stop() {
      await Promise.all([stopProcess(frontend.child), stopProcess(backend.child)]);
    }
  };
}

module.exports = { root, startDemoServers };
