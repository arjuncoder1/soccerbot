<script setup lang="ts">
type Backend = 'replay' | 'local' | 'remote'
type OrchestratorState = 'idle' | 'running' | 'succeeded' | 'failed' | 'stopped'

interface StatusResponse {
  state: OrchestratorState
  pid: number | null
  exit_code: number | null
  started_at: number | null
  finished_at: number | null
  command: string[]
  error: string | null
  log_tail: string[]
}

const config = useRuntimeConfig()
const apiBase = computed(() => String(config.public.apiBase).replace(/\/$/, ''))

const backend = ref<Backend>('replay')
const iface = ref('eth0')
const pickupDuration = ref(30)
const remoteServer = ref('')
const busy = ref(false)
const actionError = ref<string | null>(null)
const status = ref<StatusResponse | null>(null)

const isRunning = computed(() => status.value?.state === 'running')

const stateLabel = computed(() => {
  switch (status.value?.state) {
    case 'running':
      return 'Running'
    case 'succeeded':
      return 'Succeeded'
    case 'failed':
      return 'Failed'
    case 'stopped':
      return 'Stopped'
    case 'idle':
      return 'Idle'
    case undefined:
      return 'Unknown'
    default: {
      const _exhaustive: never = status.value.state
      return _exhaustive
    }
  }
})

async function refreshStatus() {
  try {
    status.value = await $fetch<StatusResponse>(`${apiBase.value}/api/orchestrator/status`)
    if (!busy.value) {
      actionError.value = null
    }
  } catch (err) {
    actionError.value = err instanceof Error ? err.message : 'Failed to reach API'
  }
}

async function startOrchestrator() {
  busy.value = true
  actionError.value = null
  try {
    status.value = await $fetch<StatusResponse>(`${apiBase.value}/api/orchestrator/start`, {
      method: 'POST',
      body: {
        backend: backend.value,
        iface: iface.value || null,
        pickup_duration_s: pickupDuration.value,
        remote_server: backend.value === 'remote' ? remoteServer.value || null : null,
      },
    })
  } catch (err: unknown) {
    const detail =
      err && typeof err === 'object' && 'data' in err
        ? String((err as { data?: { detail?: string } }).data?.detail ?? '')
        : ''
    actionError.value = detail || (err instanceof Error ? err.message : 'Start failed')
  } finally {
    busy.value = false
    await refreshStatus()
  }
}

async function stopOrchestrator() {
  busy.value = true
  actionError.value = null
  try {
    status.value = await $fetch<StatusResponse>(`${apiBase.value}/api/orchestrator/stop`, {
      method: 'POST',
    })
  } catch (err) {
    actionError.value = err instanceof Error ? err.message : 'Stop failed'
  } finally {
    busy.value = false
  }
}

let timer: ReturnType<typeof setInterval> | undefined
onMounted(async () => {
  await refreshStatus()
  timer = setInterval(refreshStatus, 1500)
})
onBeforeUnmount(() => {
  if (timer) clearInterval(timer)
})
</script>

<template>
  <div class="page">
    <div class="atmosphere" aria-hidden="true" />
    <main class="shell">
      <header class="brand-block">
        <p class="brand">Soccerbot</p>
        <h1>Orchestrator</h1>
        <p class="lede">
          Launch the pickup → turn → avoid → throw demo on the G1 from one control surface.
        </p>
      </header>

      <section class="controls" aria-label="Orchestrator controls">
        <div class="field-grid">
          <label class="field">
            <span>Backend</span>
            <select v-model="backend" :disabled="isRunning || busy">
              <option value="replay">replay</option>
              <option value="local">local</option>
              <option value="remote">remote</option>
            </select>
          </label>

          <label class="field">
            <span>Interface</span>
            <input v-model="iface" type="text" placeholder="eth0" :disabled="isRunning || busy" />
          </label>

          <label class="field">
            <span>Pickup duration (s)</span>
            <input
              v-model.number="pickupDuration"
              type="number"
              min="1"
              max="600"
              step="1"
              :disabled="isRunning || busy"
            />
          </label>

          <label v-if="backend === 'remote'" class="field field-wide">
            <span>Remote server</span>
            <input
              v-model="remoteServer"
              type="text"
              placeholder="192.168.1.42:8000"
              :disabled="isRunning || busy"
            />
          </label>
        </div>

        <div class="actions">
          <button
            class="btn primary"
            type="button"
            :disabled="isRunning || busy || (backend === 'remote' && !remoteServer)"
            @click="startOrchestrator"
          >
            Start orchestrator
          </button>
          <button
            class="btn ghost"
            type="button"
            :disabled="!isRunning || busy"
            @click="stopOrchestrator"
          >
            Stop
          </button>
        </div>

        <p v-if="actionError" class="error" role="alert">{{ actionError }}</p>
      </section>

      <section class="status" aria-live="polite">
        <div class="status-row">
          <span class="status-pill" :data-state="status?.state ?? 'idle'">{{ stateLabel }}</span>
          <span v-if="status?.pid" class="meta">pid {{ status.pid }}</span>
          <span v-if="status?.exit_code != null" class="meta">exit {{ status.exit_code }}</span>
        </div>
        <p v-if="status?.command?.length" class="command">
          <code>{{ status.command.join(' ') }}</code>
        </p>
        <pre class="log">{{ status?.log_tail?.length ? status.log_tail.join('\n') : 'Waiting for logs…' }}</pre>
      </section>
    </main>
  </div>
</template>

<style>
:root {
  --ink: #142018;
  --muted: #4a5c50;
  --paper: #e7efe6;
  --panel: rgba(255, 252, 246, 0.82);
  --line: rgba(20, 32, 24, 0.14);
  --accent: #0f6a4c;
  --accent-deep: #0a4633;
  --warn: #9a3412;
  --ok: #166534;
  --run: #1d4ed8;
  --font-sans: 'DM Sans', 'Segoe UI', sans-serif;
  --font-display: 'Instrument Serif', Georgia, serif;
}

* {
  box-sizing: border-box;
}

html,
body,
#__nuxt {
  margin: 0;
  min-height: 100%;
}

body {
  color: var(--ink);
  font-family: var(--font-sans);
  background: var(--paper);
}

.page {
  position: relative;
  min-height: 100vh;
  overflow: hidden;
}

.atmosphere {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(1200px 700px at 10% -10%, rgba(15, 106, 76, 0.28), transparent 60%),
    radial-gradient(900px 600px at 100% 0%, rgba(26, 74, 120, 0.18), transparent 55%),
    linear-gradient(160deg, #dce8dc 0%, #eef3ea 45%, #f4efe6 100%);
  z-index: 0;
}

.shell {
  position: relative;
  z-index: 1;
  width: min(920px, calc(100% - 2rem));
  margin: 0 auto;
  padding: 4.5rem 0 3rem;
  display: grid;
  gap: 2rem;
}

.brand-block h1 {
  margin: 0.15rem 0 0.6rem;
  font-family: var(--font-display);
  font-size: clamp(2.8rem, 7vw, 4.4rem);
  font-weight: 400;
  letter-spacing: -0.02em;
  line-height: 0.95;
}

.brand {
  margin: 0;
  font-size: 0.95rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--accent-deep);
}

.lede {
  margin: 0;
  max-width: 36rem;
  color: var(--muted);
  font-size: 1.05rem;
  line-height: 1.5;
}

.controls,
.status {
  background: var(--panel);
  border: 1px solid var(--line);
  backdrop-filter: blur(10px);
  padding: 1.35rem 1.4rem 1.45rem;
}

.field-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.9rem 1rem;
}

.field {
  display: grid;
  gap: 0.35rem;
  font-size: 0.82rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--muted);
}

.field-wide {
  grid-column: 1 / -1;
}

.field input,
.field select {
  width: 100%;
  border: 1px solid var(--line);
  background: #fffef9;
  color: var(--ink);
  border-radius: 0;
  padding: 0.7rem 0.75rem;
  font: inherit;
  font-weight: 500;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-top: 1.2rem;
}

.btn {
  border: 1px solid transparent;
  border-radius: 0;
  padding: 0.85rem 1.25rem;
  font: inherit;
  font-weight: 650;
  letter-spacing: 0.01em;
  cursor: pointer;
  transition: transform 160ms ease, background 160ms ease, opacity 160ms ease;
}

.btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.btn:not(:disabled):hover {
  transform: translateY(-1px);
}

.btn.primary {
  background: var(--accent);
  color: #f7fff9;
}

.btn.primary:not(:disabled):hover {
  background: var(--accent-deep);
}

.btn.ghost {
  background: transparent;
  border-color: var(--line);
  color: var(--ink);
}

.error {
  margin: 0.9rem 0 0;
  color: var(--warn);
  font-size: 0.92rem;
}

.status-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.75rem 1rem;
  margin-bottom: 0.85rem;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  min-height: 1.8rem;
  padding: 0.2rem 0.65rem;
  border: 1px solid var(--line);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.status-pill[data-state='running'] {
  color: var(--run);
  border-color: color-mix(in srgb, var(--run) 40%, var(--line));
}

.status-pill[data-state='succeeded'] {
  color: var(--ok);
}

.status-pill[data-state='failed'],
.status-pill[data-state='stopped'] {
  color: var(--warn);
}

.meta {
  color: var(--muted);
  font-size: 0.85rem;
}

.command {
  margin: 0 0 0.8rem;
}

.command code {
  display: block;
  overflow-x: auto;
  white-space: nowrap;
  font-size: 0.78rem;
  color: var(--muted);
}

.log {
  margin: 0;
  min-height: 12rem;
  max-height: 22rem;
  overflow: auto;
  padding: 0.9rem 1rem;
  background: #101612;
  color: #d7e7d9;
  font-size: 0.8rem;
  line-height: 1.45;
}

@media (max-width: 760px) {
  .shell {
    padding-top: 2.5rem;
  }

  .field-grid {
    grid-template-columns: 1fr;
  }
}
</style>
