// Tests for durable-source monotonic cost behaviour (PR #477 / copilot-otel).
// Five scenarios:
//   (a) file-purge monotonic  — copilot JSONL file deleted → total unchanged
//   (b) OTel-prune monotonic  — OTel DB rows pruned      → total unchanged
//   (c) no double-count       — same source parsed twice  → counted once
//   (d) non-durable evicts    — deleted source for non-durable provider IS removed
//   (e) 90-day age-out        — ≥ 91d pruned (orphans AND unflagged discovered
//                               sources); retainWhilePresent + discovered kept

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { mkdtemp, mkdir, writeFile, rm, unlink } from 'fs/promises'
import { tmpdir } from 'os'
import { join } from 'path'
import { createRequire } from 'node:module'

import { isSqliteAvailable } from '../src/sqlite.js'
import { calculateCost } from '../src/models.js'
import { aggregateProjectsIntoDays } from '../src/day-aggregator.js'
import { DAILY_CACHE_VERSION, currentTzKey, ensureCacheHydrated, saveDailyCache } from '../src/daily-cache.js'
import { clearSessionCache, isSessionHydrationComplete, parseAllSessions, setParseReuseValidator } from '../src/parser.js'
import { loadCache, saveCache, sessionCachePath } from '../src/session-cache.js'
import type { SessionSource, SessionParser, ParsedProviderCall } from '../src/providers/types.js'

// ── Synthetic provider state ───────────────────────────────────────────────
// Module-level so the vi.mock factory closure captures them by reference and
// tests can mutate them freely without re-creating the mock.
let _synthSources: SessionSource[] = []
let _synthDurable = false
let _synthYields: ParsedProviderCall[] = []

vi.mock('../src/providers/index.js', async (importOriginal) => {
  type Mod = typeof import('../src/providers/index.js')
  const actual = await importOriginal<Mod>()
  return {
    ...actual,
    async discoverAllSessions(filter?: string) {
      // Pass through for specific non-synthetic providers; inject synthetic
      // sources only when filter is undefined/'all'/'test-synthetic'.
      if (filter && filter !== 'all' && filter !== 'test-synthetic') {
        return actual.discoverAllSessions(filter)
      }
      const base = filter === 'test-synthetic'
        ? []
        : await actual.discoverAllSessions(filter)
      return [..._synthSources, ...base]
    },
    async getProvider(name: string) {
      if (name === 'test-synthetic') {
        return {
          name: 'test-synthetic',
          displayName: 'Test Synthetic',
          durableSources: _synthDurable,
          modelDisplayName: (m: string) => m,
          toolDisplayName: (t: string) => t,
          async discoverSessions() { return _synthSources },
          createSessionParser(_s: SessionSource, _k: Set<string>): SessionParser {
            return {
              async *parse(): AsyncGenerator<ParsedProviderCall> {
                for (const call of _synthYields) {
                  // Respect seenKeys so that when multiple sources share the same
                  // dedup key, only the first source yields it (mirrors real parsers).
                  if (_k.has(call.deduplicationKey)) continue
                  _k.add(call.deduplicationKey)
                  yield call
                }
              },
            }
          },
        }
      }
      return actual.getProvider(name)
    },
  }
})

// ── OTel DB helpers ───────────────────────────────────────────────────────
const requireForTest = createRequire(import.meta.url)
type TestDb = {
  exec(sql: string): void
  prepare(sql: string): { run(...p: unknown[]): void }
  close(): void
}

function createOtelDb(dbPath: string): void {
  const { DatabaseSync } = requireForTest('node:sqlite') as {
    DatabaseSync: new (path: string) => TestDb
  }
  const db = new DatabaseSync(dbPath)
  db.exec(`
    CREATE TABLE spans (
      span_id        TEXT    PRIMARY KEY NOT NULL,
      trace_id       TEXT    NOT NULL,
      operation_name TEXT,
      start_time_ms  INTEGER NOT NULL DEFAULT 0,
      response_model TEXT
    );
    CREATE TABLE span_attributes (
      id      INTEGER PRIMARY KEY AUTOINCREMENT,
      span_id TEXT    NOT NULL,
      key     TEXT    NOT NULL,
      value   TEXT
    );
  `)
  db.close()
}

interface OtelConvSpec {
  spanId: string
  traceId: string
  convId: string
  model: string
  input: number
  output: number
  startTimeMs?: number
}

function insertOtelConv(dbPath: string, spec: OtelConvSpec): void {
  const { DatabaseSync } = requireForTest('node:sqlite') as {
    DatabaseSync: new (path: string) => TestDb
  }
  const db = new DatabaseSync(dbPath)
  db.prepare(
    `INSERT INTO spans (span_id, trace_id, operation_name, start_time_ms, response_model)
     VALUES (?, ?, ?, ?, ?)`
  ).run(spec.spanId, spec.traceId, 'chat', spec.startTimeMs ?? Date.now(), spec.model)
  const attr = db.prepare(
    `INSERT INTO span_attributes (span_id, key, value) VALUES (?, ?, ?)`
  )
  const attrs: Record<string, string | number> = {
    'gen_ai.conversation.id':               spec.convId,
    'gen_ai.response.model':                spec.model,
    'gen_ai.usage.input_tokens':            spec.input,
    'gen_ai.usage.output_tokens':           spec.output,
    'gen_ai.usage.cache_read.input_tokens': 0,
    'gen_ai.usage.cache_creation.input_tokens': 0,
  }
  for (const [k, v] of Object.entries(attrs)) attr.run(spec.spanId, k, String(v))
  db.close()
}

// ── Copilot JSONL helpers ─────────────────────────────────────────────────
async function createJsonlSession(
  sessionStateDir: string,
  sessionId: string,
  outputTokens: number,
): Promise<string> {
  const dir = join(sessionStateDir, sessionId)
  await mkdir(dir, { recursive: true })
  await writeFile(join(dir, 'workspace.yaml'), `id: ${sessionId}\ncwd: /home/user/testproj\n`)
  // Relative timestamps: fixed calendar dates rot. The original '2026-05-01'
  // crossed copilot's durable 90-day age-out on 2026-07-30, at which point the
  // very first parse pruned the freshly-cached session and both durable tests
  // started failing everywhere with "expected +0 to be 200".
  const base = Date.now() - 5 * 24 * 60 * 60 * 1000
  const at = (offsetSec: number): string => new Date(base + offsetSec * 1000).toISOString()
  const lines = [
    JSON.stringify({ type: 'session.model_change', timestamp: at(0), data: { newModel: 'gpt-4.1' } }),
    JSON.stringify({ type: 'user.message', timestamp: at(5), data: { content: 'hello', interactionId: 'int-1' } }),
    JSON.stringify({ type: 'assistant.message', timestamp: at(10), data: { messageId: 'msg-1', outputTokens, interactionId: 'int-1', toolRequests: [] } }),
  ]
  await writeFile(join(dir, 'events.jsonl'), lines.join('\n') + '\n')
  return join(dir, 'events.jsonl')
}

// ── Helpers ───────────────────────────────────────────────────────────────
function totalCost(projects: Awaited<ReturnType<typeof parseAllSessions>>): number {
  return projects
    .flatMap(p => p.sessions)
    .flatMap(s => s.turns)
    .flatMap(t => t.assistantCalls)
    .reduce((s, c) => s + c.costUSD, 0)
}

function totalOutput(projects: Awaited<ReturnType<typeof parseAllSessions>>): number {
  return projects
    .flatMap(p => p.sessions)
    .flatMap(s => s.turns)
    .flatMap(t => t.assistantCalls)
    .reduce((s, c) => s + c.usage.outputTokens, 0)
}

// ── Common env setup ──────────────────────────────────────────────────────
let tmpHome: string
let tmpCache: string

beforeEach(async () => {
  tmpHome  = await mkdtemp(join(tmpdir(), 'cb-parser-test-home-'))
  tmpCache = await mkdtemp(join(tmpdir(), 'cb-parser-test-cache-'))

  process.env['HOME']               = tmpHome
  process.env['CODEBURN_CACHE_DIR'] = tmpCache

  // Reset synthetic provider state
  _synthSources = []
  _synthDurable = false
  _synthYields  = []
})

afterEach(async () => {
  clearSessionCache()
  vi.unstubAllEnvs()

  _synthSources = []

  await rm(tmpHome,  { recursive: true, force: true })
  await rm(tmpCache, { recursive: true, force: true })
})

// ═══════════════════════════════════════════════════════════════════════════
// (a) File-purge monotonic: copilot JSONL file deleted → total unchanged
// ═══════════════════════════════════════════════════════════════════════════
describe('(a) copilot JSONL file-purge monotonic', () => {
  it('preserves monthly total after events.jsonl is deleted', async () => {
    const sessionStateDir = join(tmpHome, 'session-state')
    await mkdir(sessionStateDir, { recursive: true })

    vi.stubEnv('CODEBURN_COPILOT_SESSION_STATE_DIR', sessionStateDir)
    vi.stubEnv('CODEBURN_COPILOT_DISABLE_OTEL', '1')
    vi.stubEnv('CODEBURN_COPILOT_WS_STORAGE_DIR', join(tmpHome, 'no-ws'))

    const eventsPath = await createJsonlSession(sessionStateDir, 'sess-del', 200)

    // First parse: file exists → cached
    const proj1 = await parseAllSessions(undefined, 'copilot')
    const out1 = totalOutput(proj1)
    expect(out1).toBe(200)

    // Delete the source file (simulates VS Code / CLI pruning it)
    await unlink(eventsPath)
    clearSessionCache()

    // Second parse: file gone but copilot is durable → total must not drop
    const proj2 = await parseAllSessions(undefined, 'copilot')
    const out2 = totalOutput(proj2)
    expect(out2).toBeGreaterThanOrEqual(out1)
    expect(out2).toBe(out1)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (b) OTel-prune monotonic: OTel DB rows pruned → total unchanged
// ═══════════════════════════════════════════════════════════════════════════
describe.skipIf(!isSqliteAvailable())(
  '(b) OTel DB-prune monotonic',
  () => {
    it('preserves total after one conversation is pruned from the OTel DB', async () => {
      const dbPath = join(tmpHome, 'agent-traces.db')
      vi.stubEnv('CODEBURN_COPILOT_OTEL_DB', dbPath)
      vi.stubEnv('CODEBURN_COPILOT_DISABLE_OTEL', '')
      vi.stubEnv('CODEBURN_COPILOT_SESSION_STATE_DIR', join(tmpHome, 'no-jsonl'))
      vi.stubEnv('CODEBURN_COPILOT_WS_STORAGE_DIR',   join(tmpHome, 'no-ws'))

      // DB with two conversations
      createOtelDb(dbPath)
      insertOtelConv(dbPath, { spanId: 's1', traceId: 't1', convId: 'prune-c1', model: 'gpt-4.1', input: 500,  output: 50 })
      insertOtelConv(dbPath, { spanId: 's2', traceId: 't2', convId: 'prune-c2', model: 'gpt-4.1', input: 1000, output: 100 })

      const proj1 = await parseAllSessions(undefined, 'copilot')
      const out1 = totalOutput(proj1)
      expect(out1).toBe(150)  // 50 + 100

      // Simulate OTel pruning conv-1 from the DB: rebuild DB with only conv-2
      clearSessionCache()
      await rm(dbPath)
      createOtelDb(dbPath)
      insertOtelConv(dbPath, { spanId: 's2', traceId: 't2', convId: 'prune-c2', model: 'gpt-4.1', input: 1000, output: 100 })

      // Second parse: DB was rebuilt without conv-1. The union-merge in
      // parseProviderSources keeps conv-1's turns in the cache (since its
      // dedup keys are not re-emitted by the re-parse) → total must not drop.
      const proj2 = await parseAllSessions(undefined, 'copilot')
      const out2 = totalOutput(proj2)
      expect(out2).toBeGreaterThanOrEqual(out1)
      expect(out2).toBe(out1)
    })
  }
)

// ═══════════════════════════════════════════════════════════════════════════
// (c) No double-count: same fully-present source parsed twice → counted once
// ═══════════════════════════════════════════════════════════════════════════
describe.skipIf(!isSqliteAvailable())(
  '(c) OTel source parsed twice is counted once',
  () => {
    it('second parse of unchanged DB yields same total, not double', async () => {
      const dbPath = join(tmpHome, 'agent-traces.db')
      vi.stubEnv('CODEBURN_COPILOT_OTEL_DB', dbPath)
      vi.stubEnv('CODEBURN_COPILOT_DISABLE_OTEL', '')
      vi.stubEnv('CODEBURN_COPILOT_SESSION_STATE_DIR', join(tmpHome, 'no-jsonl'))
      vi.stubEnv('CODEBURN_COPILOT_WS_STORAGE_DIR',   join(tmpHome, 'no-ws'))

      createOtelDb(dbPath)
      insertOtelConv(dbPath, { spanId: 'dedup-s1', traceId: 'dedup-t1', convId: 'dedup-c1', model: 'gpt-4.1', input: 300, output: 30 })

      const proj1 = await parseAllSessions(undefined, 'copilot')
      expect(totalOutput(proj1)).toBe(30)

      clearSessionCache()

      // Second parse — disk cache is populated, fingerprint unchanged
      const proj2 = await parseAllSessions(undefined, 'copilot')
      expect(totalOutput(proj2)).toBe(30)  // NOT 60
    })
  }
)

// ═══════════════════════════════════════════════════════════════════════════
// (d) Non-durable evicts: deleted source for non-durable provider is removed
// ═══════════════════════════════════════════════════════════════════════════
describe('(d) non-durable provider evicts deleted sources', () => {
  it('removes cache entry for a path that leaves discoverSessions()', async () => {
    // Two real temp files as source paths (fingerprintFile needs them to exist)
    const fileA = join(tmpHome, 'synth-a.txt')
    const fileB = join(tmpHome, 'synth-b.txt')
    await writeFile(fileA, 'placeholder-a')
    await writeFile(fileB, 'placeholder-b')

    const dedupA = 'synth-dedup-evict-a'
    const dedupB = 'synth-dedup-evict-b'

    const makeCall = (deduplicationKey: string): ParsedProviderCall => ({
      provider: 'test-synthetic', model: 'gpt-4o',
      inputTokens: 10, outputTokens: 5,
      cacheCreationInputTokens: 0, cacheReadInputTokens: 0,
      cachedInputTokens: 0, reasoningTokens: 0, webSearchRequests: 0,
      costUSD: 0.001, tools: [], bashCommands: [],
      timestamp: new Date().toISOString(),
      speed: 'standard',
      deduplicationKey,
      userMessage: 'test', sessionId: 'synth-sess',
    })

    _synthDurable = false
    _synthSources = [
      { path: fileA, project: 'test', provider: 'test-synthetic' },
      { path: fileB, project: 'test', provider: 'test-synthetic' },
    ]
    _synthYields = [makeCall(dedupA)]

    // First parse: both sources present → data for A cached
    const proj1 = await parseAllSessions(undefined, 'test-synthetic')
    expect(totalOutput(proj1)).toBeGreaterThan(0)

    clearSessionCache()

    // Remove A from discovered sources (simulates file-gone + discoverSessions skips it).
    // B stays so sources.length > 0 → eviction loop fires.
    _synthSources = [{ path: fileB, project: 'test', provider: 'test-synthetic' }]
    _synthYields  = []  // B yields nothing (empty file)

    const proj2 = await parseAllSessions(undefined, 'test-synthetic')
    // A's cache entry must be evicted → total should be 0
    expect(totalOutput(proj2)).toBe(0)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (e) 90-day age-out: orphan ≥ 91d old is pruned; ≤ 89d is retained
// ═══════════════════════════════════════════════════════════════════════════
describe('(e) 90-day age-out for durable providers', () => {
  it('prunes a 91-day-old entry even while its unflagged source is still discovered', async () => {
    const synthFile = join(tmpHome, 'synth-age.txt')
    await writeFile(synthFile, 'placeholder')

    const ts91dAgo = new Date(Date.now() - 91 * 24 * 60 * 60 * 1000).toISOString()

    _synthDurable = true
    _synthSources = [{ path: synthFile, project: 'test', provider: 'test-synthetic' }]
    _synthYields  = [{
      provider: 'test-synthetic', model: 'gpt-4o',
      inputTokens: 10, outputTokens: 8,
      cacheCreationInputTokens: 0, cacheReadInputTokens: 0,
      cachedInputTokens: 0, reasoningTokens: 0, webSearchRequests: 0,
      costUSD: 0.002, tools: [], bashCommands: [],
      timestamp: ts91dAgo,
      speed: 'standard',
      deduplicationKey: 'synth-age-out-91d',
      userMessage: 'old', sessionId: 'synth-old',
    }]

    // Unflagged durable sources age out on the ordinary schedule even while
    // the file remains on disk (the pre-existing cap on cache growth for
    // provider-pruned journals). Only a retainWhilePresent source — the
    // copilot session-store, which IS the durable record — is exempt.
    const proj1 = await parseAllSessions(undefined, 'test-synthetic')
    expect(totalOutput(proj1)).toBe(0)

    // Stable on the next pass too (re-parsed, then aged out again).
    clearSessionCache()
    const proj2 = await parseAllSessions(undefined, 'test-synthetic')
    expect(totalOutput(proj2)).toBe(0)
  })

  it('retains a 91-day-old entry whose still-discovered source declares retainWhilePresent', async () => {
    const synthFile = join(tmpHome, 'synth-retain-flag.txt')
    await writeFile(synthFile, 'placeholder')

    const ts91dAgo = new Date(Date.now() - 91 * 24 * 60 * 60 * 1000).toISOString()

    _synthDurable = true
    _synthSources = [{ path: synthFile, project: 'test', provider: 'test-synthetic', retainWhilePresent: true }]
    _synthYields  = [{
      provider: 'test-synthetic', model: 'gpt-4o',
      inputTokens: 10, outputTokens: 8,
      cacheCreationInputTokens: 0, cacheReadInputTokens: 0,
      cachedInputTokens: 0, reasoningTokens: 0, webSearchRequests: 0,
      costUSD: 0.002, tools: [], bashCommands: [],
      timestamp: ts91dAgo,
      speed: 'standard',
      deduplicationKey: 'synth-retain-flag-91d',
      userMessage: 'old', sessionId: 'synth-flag',
    }]

    // Flagged + discovered: the file is the durable record; pruning it would
    // drop data only it holds. Served and retained across passes.
    const proj1 = await parseAllSessions(undefined, 'test-synthetic')
    expect(totalOutput(proj1)).toBe(8)
    clearSessionCache()
    const proj2 = await parseAllSessions(undefined, 'test-synthetic')
    expect(totalOutput(proj2)).toBe(8)

    // Once ORPHANED the flag no longer applies — orphan age-out prunes.
    clearSessionCache()
    _synthSources = []  // no longer discovered
    const proj3 = await parseAllSessions(undefined, 'test-synthetic')
    expect(totalOutput(proj3)).toBe(0)
  })

  it('retains an orphaned cache entry whose newest call is 89 days old', async () => {
    const synthFile = join(tmpHome, 'synth-retain.txt')
    await writeFile(synthFile, 'placeholder')

    const ts89dAgo = new Date(Date.now() - 89 * 24 * 60 * 60 * 1000).toISOString()

    _synthDurable = true
    _synthSources = [{ path: synthFile, project: 'test', provider: 'test-synthetic' }]
    _synthYields  = [{
      provider: 'test-synthetic', model: 'gpt-4o',
      inputTokens: 10, outputTokens: 7,
      cacheCreationInputTokens: 0, cacheReadInputTokens: 0,
      cachedInputTokens: 0, reasoningTokens: 0, webSearchRequests: 0,
      costUSD: 0.002, tools: [], bashCommands: [],
      timestamp: ts89dAgo,
      speed: 'standard',
      deduplicationKey: 'synth-retain-89d',
      userMessage: 'recent-ish', sessionId: 'synth-recent',
    }]

    // First parse: cached with 89d-old timestamp → NOT pruned (within 90d window)
    const proj1 = await parseAllSessions(undefined, 'test-synthetic')
    expect(totalOutput(proj1)).toBe(7)

    // Remove source (simulate it being orphaned)
    clearSessionCache()
    _synthSources = []  // no longer discovered → orphan pass handles it

    // Second parse: orphan with 89d timestamp → retained + counted via orphan pass
    const proj2 = await parseAllSessions(undefined, 'test-synthetic')
    expect(totalOutput(proj2)).toBe(7)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// Emission gate: a reasoning-only copilot session still emits
// ═══════════════════════════════════════════════════════════════════════════
// A rollup can carry ONLY reasoning tokens (zero input/cache/output, zero
// cost — copilot reasoning is never priced). The gate admits any
// usage-bearing session; dropping this one would silently lose the only
// record of its usage.
describe('reasoning-only copilot session emission', () => {
  it('emits a session whose sole rollup carries only reasoning tokens', async () => {
    const sessionStateDir = join(tmpHome, 'session-state-reason-only')
    await mkdir(sessionStateDir, { recursive: true })
    vi.stubEnv('CODEBURN_COPILOT_SESSION_STATE_DIR', sessionStateDir)
    vi.stubEnv('CODEBURN_COPILOT_SESSION_STORE_DB', join(tmpHome, 'no-store.db'))
    vi.stubEnv('CODEBURN_COPILOT_DISABLE_OTEL', '1')
    vi.stubEnv('CODEBURN_COPILOT_WS_STORAGE_DIR', join(tmpHome, 'no-ws'))
    vi.stubEnv('CODEBURN_COPILOT_GLOBAL_STORAGE_DIR', join(tmpHome, 'no-global'))
    vi.stubEnv('CODEBURN_COPILOT_JETBRAINS_DIR', join(tmpHome, 'no-jb'))

    const ts = new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString()
    const dir = join(sessionStateDir, 'sess-reason')
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, 'workspace.yaml'), 'id: sess-reason\ncwd: /home/user/testproj\n')
    await writeFile(join(dir, 'events.jsonl'), [
      JSON.stringify({ type: 'session.model_change', timestamp: ts, data: { newModel: 'gpt-5' } }),
      JSON.stringify({
        type: 'session.shutdown',
        timestamp: ts,
        data: {
          shutdownType: 'routine',
          modelMetrics: {
            'gpt-5': {
              requests: { count: 1, cost: 0 },
              usage: { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0, reasoningTokens: 800 },
            },
          },
        },
      }),
    ].join('\n') + '\n')

    const projects = await parseAllSessions(undefined, 'copilot')
    const sessions = projects.flatMap(p => p.sessions).filter(s => s.sessionId === 'sess-reason')
    expect(sessions).toHaveLength(1)
    expect(sessions[0]!.totalReasoningTokens).toBe(800)
    // The rollup is supplementary accounting: usage served, zero call weight.
    expect(sessions[0]!.apiCalls).toBe(0)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (f) Version-bump survival: a PROVIDER_PARSE_VERSIONS bump (or any env
//     fingerprint change) must NOT erase durable orphans. The cache is the
//     only remaining record of usage whose source was pruned; discarding the
//     section wholesale on fingerprint mismatch permanently lost that history
//     (caught in the #684 re-review).
// ═══════════════════════════════════════════════════════════════════════════
describe('(f) durable orphans survive a parse-version bump', () => {
  it('keeps counting a pruned-source orphan after the provider fingerprint changes', async () => {
    const sessionStateDir = join(tmpHome, 'session-state')
    await mkdir(sessionStateDir, { recursive: true })
    vi.stubEnv('CODEBURN_COPILOT_SESSION_STATE_DIR', sessionStateDir)
    vi.stubEnv('CODEBURN_COPILOT_DISABLE_OTEL', '1')
    vi.stubEnv('CODEBURN_COPILOT_WS_STORAGE_DIR', join(tmpHome, 'no-ws'))

    // Parse once so the session is cached, then prune the source: the cache
    // entry becomes a durable orphan (its only record).
    const eventsPath = await createJsonlSession(sessionStateDir, 'sess-bump', 200)
    const before = totalOutput(await parseAllSessions(undefined, 'copilot'))
    expect(before).toBe(200)
    await unlink(eventsPath)
    clearSessionCache()

    // Simulate the fingerprint a PREVIOUS release computed (any mismatching
    // value takes the same code path as a real parse-version bump).
    const { readFile, writeFile: writeFileFs } = await import('fs/promises')
    const cachePath = sessionCachePath()
    const disk = JSON.parse(await readFile(cachePath, 'utf-8')) as { providers: Record<string, { envFingerprint: string }> }
    expect(disk.providers['copilot']).toBeDefined()
    disk.providers['copilot']!.envFingerprint = '0000000000000000'
    await writeFileFs(cachePath, JSON.stringify(disk), 'utf-8')

    // First parse after the "upgrade": the orphan must still be counted and
    // must survive in the rewritten cache, not be erased with the section.
    const after = totalOutput(await parseAllSessions(undefined, 'copilot'))
    expect(after).toBe(200)

    clearSessionCache()
    const again = totalOutput(await parseAllSessions(undefined, 'copilot'))
    expect(again).toBe(200)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (g) Skill attribution is independent of turn category
// ═══════════════════════════════════════════════════════════════════════════
describe('(g) skill attribution is independent of turn category', () => {
  it('puts a Skill + Edit turn in skillBreakdown while preserving coding category', async () => {
    const synthFile = join(tmpHome, 'synth-skill.txt')
    await writeFile(synthFile, 'placeholder')

    _synthSources = [{ path: synthFile, project: 'test', provider: 'test-synthetic' }]
    _synthYields = [{
      provider: 'test-synthetic', model: 'gpt-4o',
      inputTokens: 10, outputTokens: 5,
      cacheCreationInputTokens: 0, cacheReadInputTokens: 0,
      cachedInputTokens: 0, reasoningTokens: 0, webSearchRequests: 0,
      costUSD: 0.001, tools: ['Skill', 'Edit'], bashCommands: [],
      skills: ['telemetry-review'],
      timestamp: '2026-07-18T12:00:00.000Z',
      speed: 'standard',
      deduplicationKey: 'synth-skill-edit',
      userMessage: '', sessionId: 'synth-skill-session',
    }]

    const projects = await parseAllSessions(undefined, 'test-synthetic')
    const session = projects.flatMap(project => project.sessions)[0]

    expect(session).toBeDefined()
    expect(session!.turns[0]!.category).toBe('coding')
    expect(session!.turns[0]!.subCategory).toBe('telemetry-review')
    expect(session!.categoryBreakdown.coding.turns).toBe(1)
    expect(session!.skillBreakdown['telemetry-review']?.turns).toBe(1)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (h) Provider filter isolates claude: a --provider <other> run must not
//     re-surface cached claude sessions through the orphan pass, while a run
//     that DOES include claude still preserves PR-bearing orphans.
// ═══════════════════════════════════════════════════════════════════════════
describe('(h) provider filter excludes claude from the orphan pass', () => {
  const SYNTH_SOURCE = (path: string): SessionSource[] =>
    [{ path, project: 'synth-proj', provider: 'test-synthetic' }]

  // The provider lives on each parsed call, not on SessionSummary.
  const providersOf = (projects: Awaited<ReturnType<typeof parseAllSessions>>): Set<string> =>
    new Set(projects
      .flatMap(p => p.sessions)
      .flatMap(s => s.turns)
      .flatMap(t => t.assistantCalls)
      .map(c => c.provider))

  const SYNTH_CALL: ParsedProviderCall = {
    provider: 'test-synthetic', model: 'gpt-4o',
    inputTokens: 10, outputTokens: 5,
    cacheCreationInputTokens: 0, cacheReadInputTokens: 0,
    cachedInputTokens: 0, reasoningTokens: 0, webSearchRequests: 0,
    costUSD: 0.25, tools: [], bashCommands: [],
    skills: [],
    timestamp: '2026-07-18T12:00:00.000Z',
    speed: 'standard',
    deduplicationKey: 'synth-isolation-call',
    userMessage: '', sessionId: 'synth-isolation-session',
  }

  // A claude transcript carrying a pr-link: `prLinks` is exactly what lets a
  // cached entry survive the write-mode orphan gate, so it is the shape that
  // leaks. Cost is deliberately far larger than the synthetic call's, so a leak
  // is unmistakable rather than a rounding difference.
  async function writeClaudeSessionWithPrLink(): Promise<string> {
    const projectDir = join(tmpHome, '.claude', 'projects', 'leaky-app')
    await mkdir(projectDir, { recursive: true })
    const filePath = join(projectDir, 'session.jsonl')
    await writeFile(filePath, [
      JSON.stringify({
        type: 'user', sessionId: 'claude-leak-1', timestamp: '2026-07-18T12:00:00.000Z',
        message: { role: 'user', content: 'ship it' },
      }),
      JSON.stringify({
        type: 'assistant', sessionId: 'claude-leak-1', timestamp: '2026-07-18T12:00:10.000Z',
        message: {
          id: 'msg-leak-1', type: 'message', role: 'assistant', model: 'claude-sonnet-4-5',
          content: [{ type: 'text', text: 'done' }],
          usage: { input_tokens: 900_000, output_tokens: 90_000 },
        },
      }),
      JSON.stringify({
        type: 'pr-link', sessionId: 'claude-leak-1', timestamp: '2026-07-18T12:00:20.000Z',
        prUrl: 'https://github.com/getagentseal/codeburn/pull/1',
      }),
    ].join('\n') + '\n')
    return filePath
  }

  it('does not surface cached claude sessions when filtering to another provider', async () => {
    const synthFile = join(tmpHome, 'synth-isolation.txt')
    await writeFile(synthFile, 'placeholder')
    await writeClaudeSessionWithPrLink()

    _synthSources = SYNTH_SOURCE(synthFile)
    _synthYields = [SYNTH_CALL]

    // Baseline: what the synthetic provider costs on its own, before anything
    // claude-shaped has ever entered the session cache. Self-calibrating, since
    // cost is re-derived from tokens by the pricing engine.
    const baseline = await parseAllSessions(undefined, 'test-synthetic')
    const synthOnlyCost = totalCost(baseline)
    expect([...providersOf(baseline)]).toEqual(['test-synthetic'])
    clearSessionCache()

    // Warm the session cache so the claude file is persisted WITH its prLinks.
    const all = await parseAllSessions(undefined, 'all')
    expect(providersOf(all)).toContain('claude')
    expect(totalCost(all)).toBeGreaterThan(synthOnlyCost)

    clearSessionCache()

    // Filtering to the synthetic provider must yield ONLY its own spend. Before
    // the fix, claudeDirs was empty yet scanProjectDirs still ran, so every
    // cached PR-bearing claude file was treated as a pruned orphan and re-added.
    const filtered = await parseAllSessions(undefined, 'test-synthetic')

    expect([...providersOf(filtered)]).toEqual(['test-synthetic'])
    expect(totalCost(filtered)).toBeCloseTo(synthOnlyCost, 10)
  })

  it('still preserves a PR-bearing claude orphan when claude IS in scope', async () => {
    const filePath = await writeClaudeSessionWithPrLink()
    _synthSources = []
    _synthYields = []

    const before = await parseAllSessions(undefined, 'all')
    const costBefore = totalCost(before)
    expect(costBefore).toBeGreaterThan(0)

    // Every claude transcript disappears from disk. Claude is still in scope, so
    // the orphan pass must keep the PR-attributed spend alive — this is the case
    // a naive `claudeDirs.length > 0` guard would silently break.
    await unlink(filePath)
    clearSessionCache()

    const after = await parseAllSessions(undefined, 'all')
    expect(totalCost(after)).toBeCloseTo(costBefore, 10)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (f) Growing resumed CLI session: durable merge appends only the new leg
// ═══════════════════════════════════════════════════════════════════════════
// Resumed Copilot CLI sessions append one CUMULATIVE session.shutdown per leg
// (#944). The parser emits per-leg deltas keyed by occurrence; this exercises
// the PRODUCTION merge path — the durable union-by-dedup-key merge against the
// on-disk cache when the file grows between parses — which the unit tests
// (which pre-seed seenKeys) cannot reach.
describe('(f) growing resumed CLI session durable merge', () => {
  it('totals equal the final cumulative rollup after the file grows a leg', async () => {
    const sessionStateDir = join(tmpHome, 'session-state')
    await mkdir(sessionStateDir, { recursive: true })
    vi.stubEnv('CODEBURN_COPILOT_SESSION_STATE_DIR', sessionStateDir)
    vi.stubEnv('CODEBURN_COPILOT_DISABLE_OTEL', '1')
    vi.stubEnv('CODEBURN_COPILOT_WS_STORAGE_DIR', join(tmpHome, 'no-ws'))
    vi.stubEnv('CODEBURN_COPILOT_GLOBAL_STORAGE_DIR', join(tmpHome, 'no-global'))
    vi.stubEnv('CODEBURN_COPILOT_JETBRAINS_DIR', join(tmpHome, 'no-jb'))

    const base = Date.now() - 5 * 24 * 60 * 60 * 1000
    const at = (offsetSec: number): string => new Date(base + offsetSec * 1000).toISOString()
    const dir = join(sessionStateDir, 'sess-grow')
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, 'workspace.yaml'), 'id: sess-grow\ncwd: /home/user/testproj\n')
    const eventsPath = join(dir, 'events.jsonl')

    // Cumulative rollups from a real resumed CLI 1.0.78 session.
    const shutdown = (ts: string, inputTokens: number, cacheReadTokens: number, cacheWriteTokens: number, outputTokens: number) =>
      JSON.stringify({
        type: 'session.shutdown',
        timestamp: ts,
        data: {
          shutdownType: 'routine',
          modelMetrics: {
            'claude-sonnet-4-5': {
              requests: { count: 1, cost: 1 },
              usage: { inputTokens, outputTokens, cacheReadTokens, cacheWriteTokens, reasoningTokens: 0 },
            },
          },
        },
      })
    const leg1 = [
      JSON.stringify({ type: 'session.model_change', timestamp: at(0), data: { newModel: 'claude-sonnet-4-5' } }),
      JSON.stringify({ type: 'assistant.message', timestamp: at(10), data: { messageId: 'msg-1', outputTokens: 17, toolRequests: [] } }),
      shutdown(at(20), 24672, 0, 24670, 17),
    ]
    await writeFile(eventsPath, leg1.join('\n') + '\n')

    const sumUsage = (projects: Awaited<ReturnType<typeof parseAllSessions>>) => {
      const calls = projects.flatMap(p => p.sessions).flatMap(s => s.turns).flatMap(t => t.assistantCalls)
      return {
        input: calls.reduce((s, c) => s + c.usage.inputTokens, 0),
        cacheRead: calls.reduce((s, c) => s + c.usage.cacheReadInputTokens, 0),
        cacheWrite: calls.reduce((s, c) => s + c.usage.cacheCreationInputTokens, 0),
      }
    }

    const first = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(first).toEqual({ input: 2, cacheRead: 0, cacheWrite: 24670 })

    // The session resumes: leg 2 appends per-turn events plus a CUMULATIVE
    // rollup. The cached leg-1 delta must be kept once and only the leg-2
    // delta appended — totals equal the final cumulative rollup exactly.
    clearSessionCache()
    await writeFile(eventsPath, [
      ...leg1,
      JSON.stringify({ type: 'assistant.message', timestamp: at(100), data: { messageId: 'msg-2', outputTokens: 132, toolRequests: [] } }),
      shutdown(at(120), 74463, 49489, 24968, 149),
    ].join('\n') + '\n')

    const second = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(second).toEqual({ input: 74463 - 49489 - 24968, cacheRead: 49489, cacheWrite: 24968 })
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (q) Burst reuse: a through-now range re-anchored seconds later reuses the
//     previous parse instead of re-running discovery (serve fast-path)
// ═══════════════════════════════════════════════════════════════════════════
describe('(q) parse burst reuse (CODEBURN_PARSE_BURST_MS)', () => {
  it('serves a re-anchored range from the previous parse inside the window, never outside it', async () => {
    vi.stubEnv('CODEBURN_PARSE_BURST_MS', '10000')
    clearSessionCache()
    const start = new Date(Date.now() - 60 * 60 * 1000)
    const ts = new Date(Date.now() - 10 * 60 * 1000).toISOString()
    const synthFile = join(tmpHome, 'synth-burst.txt')
    await writeFile(synthFile, 'placeholder')
    _synthSources = [{ path: synthFile, project: 'p', provider: 'test-synthetic' }]
    _synthYields = [{
      provider: 'test-synthetic', model: 'synth-model',
      inputTokens: 1, outputTokens: 5, cacheCreationInputTokens: 0, cacheReadInputTokens: 0,
      cachedInputTokens: 0, reasoningTokens: 0, webSearchRequests: 0,
      costUSD: 0, costIsEstimated: false, tools: [], bashCommands: [], skills: [],
      timestamp: ts, speed: 'standard', deduplicationKey: 'synth-burst-1', userMessage: 'hi', sessionId: 'sb-1',
    }] as never

    const first = await parseAllSessions({ start, end: new Date() }, 'test-synthetic')
    expect(totalOutput(first)).toBe(5)

    // The world changes (a second call appears), but a burst-window re-anchor
    // must serve the PREVIOUS parse: same data, no re-discovery.
    _synthYields = [..._synthYields, {
      ...( _synthYields[0] as object ), deduplicationKey: 'synth-burst-2', outputTokens: 7,
    }] as never
    const second = await parseAllSessions({ start, end: new Date(Date.now() + 1000) }, 'test-synthetic')
    expect(totalOutput(second)).toBe(5)

    // Outside the window (env cleared = burst disabled), the fresh parse sees
    // the new call: proof the reuse was the burst path, not staleness. The
    // source file must actually change, or the fingerprint-keyed disk cache
    // (correctly) serves the old turns.
    vi.stubEnv('CODEBURN_PARSE_BURST_MS', '0')
    await writeFile(synthFile, 'placeholder v2 with a second call')
    clearSessionCache()
    const third = await parseAllSessions({ start, end: new Date(Date.now() + 2000) }, 'test-synthetic')
    expect(totalOutput(third)).toBe(12)
    vi.unstubAllEnvs()
    _synthSources = []
    _synthYields = []
  })
})

describe('(r) validated parse reuse (setParseReuseValidator)', () => {
  it('reuses past the burst window while the validator reports quiet, never when dirty', async () => {
    vi.stubEnv('CODEBURN_PARSE_BURST_MS', '1')
    clearSessionCache()
    const start = new Date(Date.now() - 60 * 60 * 1000)
    const ts = new Date(Date.now() - 10 * 60 * 1000).toISOString()
    const synthFile = join(tmpHome, 'synth-validated.txt')
    await writeFile(synthFile, 'placeholder')
    _synthSources = [{ path: synthFile, project: 'p', provider: 'test-synthetic' }]
    _synthYields = [{
      provider: 'test-synthetic', model: 'synth-model',
      inputTokens: 1, outputTokens: 5, cacheCreationInputTokens: 0, cacheReadInputTokens: 0,
      cachedInputTokens: 0, reasoningTokens: 0, webSearchRequests: 0,
      costUSD: 0, costIsEstimated: false, tools: [], bashCommands: [], skills: [],
      timestamp: ts, speed: 'standard', deduplicationKey: 'synth-val-1', userMessage: 'hi', sessionId: 'sv-1',
    }] as never

    const first = await parseAllSessions({ start, end: new Date() }, 'test-synthetic')
    expect(totalOutput(first)).toBe(5)

    // 1ms burst window has certainly elapsed; with a quiet validator the
    // previous parse is still served (world changed, result must not).
    await new Promise(r => setTimeout(r, 5))
    setParseReuseValidator(() => true)
    _synthYields = [..._synthYields, { ...( _synthYields[0] as object ), deduplicationKey: 'synth-val-2', outputTokens: 7 }] as never
    await writeFile(synthFile, 'placeholder v2')
    const second = await parseAllSessions({ start, end: new Date(Date.now() + 500) }, 'test-synthetic')
    expect(totalOutput(second)).toBe(5)

    // A dirty validator ends the reuse: fresh parse sees the new call.
    setParseReuseValidator(() => false)
    const third = await parseAllSessions({ start, end: new Date(Date.now() + 1000) }, 'test-synthetic')
    expect(totalOutput(third)).toBe(12)

    setParseReuseValidator(null)
    vi.unstubAllEnvs()
    _synthSources = []
    _synthYields = []
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (i) Growing session-store DB: durable merge appends only the new rows
// ═══════════════════════════════════════════════════════════════════════════
// session-store.db records one usage row per API request; rows only ever
// append (AUTOINCREMENT ids). This exercises the PRODUCTION path end to end:
// both representations parse and cache, serve-time precedence
// (parseProviderSources) drops the covered session's shutdown rollup, and a
// re-parse after INSERTs appends exactly the new rows under the durable
// union-by-dedup-key merge — totals must equal the DB, not the rollup, and
// never double-count.
function createStoreDb(dbPath: string): void {
  const { DatabaseSync } = requireForTest('node:sqlite') as { DatabaseSync: new (path: string) => TestDb }
  const db = new DatabaseSync(dbPath)
  db.exec(`
    CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT, repository TEXT, created_at TEXT);
    CREATE TABLE assistant_usage_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT NOT NULL,
      model TEXT NOT NULL,
      input_tokens INTEGER,
      output_tokens INTEGER,
      cache_read_tokens INTEGER,
      cache_write_tokens INTEGER,
      reasoning_tokens INTEGER,
      created_at TEXT
    );
  `)
  db.close()
}

function insertStoreRow(
  dbPath: string,
  sessionId: string,
  inputTokens: number,   // cache-inclusive, as the CLI writes it
  cacheRead: number,
  cacheWrite: number,
  createdAt: string,
  reasoning = 0,
  cwd = '/home/user/testproj',
): void {
  const { DatabaseSync } = requireForTest('node:sqlite') as { DatabaseSync: new (path: string) => TestDb }
  const db = new DatabaseSync(dbPath)
  db.prepare(`INSERT OR IGNORE INTO sessions (id, cwd) VALUES (?, ?)`).run(sessionId, cwd)
  db.prepare(
    `INSERT INTO assistant_usage_events
       (session_id, model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, created_at)
     VALUES (?, 'claude-sonnet-4-5', ?, 0, ?, ?, ?, ?)`
  ).run(sessionId, inputTokens, cacheRead, cacheWrite, reasoning, createdAt)
  db.close()
}

describe.skipIf(!isSqliteAvailable())('(i) growing session-store DB durable merge', () => {
  it('totals track the store exactly as rows append, with the rollup reconciled away', async () => {
    const sessionStateDir = join(tmpHome, 'session-state')
    await mkdir(sessionStateDir, { recursive: true })
    const dbPath = join(tmpHome, 'session-store.db')
    vi.stubEnv('CODEBURN_COPILOT_SESSION_STATE_DIR', sessionStateDir)
    vi.stubEnv('CODEBURN_COPILOT_SESSION_STORE_DB', dbPath)
    vi.stubEnv('CODEBURN_COPILOT_DISABLE_OTEL', '1')
    vi.stubEnv('CODEBURN_COPILOT_WS_STORAGE_DIR', join(tmpHome, 'no-ws'))
    vi.stubEnv('CODEBURN_COPILOT_GLOBAL_STORAGE_DIR', join(tmpHome, 'no-global'))
    vi.stubEnv('CODEBURN_COPILOT_JETBRAINS_DIR', join(tmpHome, 'no-jb'))

    const base = Date.now() - 5 * 24 * 60 * 60 * 1000
    const at = (offsetSec: number): string => new Date(base + offsetSec * 1000).toISOString()

    // The session's events.jsonl carries per-turn output AND a shutdown
    // rollup summing exactly the two covered requests (the production shape:
    // rows commit before the rollup is written, cache-inclusive on both
    // sides). If reconciliation failed to replace the rollup, totals would
    // double; if it dropped usage, they would fall short of the rows.
    const dir = join(sessionStateDir, 'sess-store')
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, 'workspace.yaml'), 'id: sess-store\ncwd: /home/user/testproj\n')
    await writeFile(join(dir, 'events.jsonl'), [
      JSON.stringify({ type: 'session.model_change', timestamp: at(0), data: { newModel: 'claude-sonnet-4-5' } }),
      JSON.stringify({ type: 'assistant.message', timestamp: at(10), data: { messageId: 'msg-1', outputTokens: 17, toolRequests: [] } }),
      JSON.stringify({
        type: 'session.shutdown',
        timestamp: at(20),
        data: {
          shutdownType: 'routine',
          modelMetrics: {
            'claude-sonnet-4-5': {
              requests: { count: 2, cost: 1 },
              usage: { inputTokens: 20000, outputTokens: 17, cacheReadTokens: 17000, cacheWriteTokens: 2400, reasoningTokens: 40 },
            },
          },
        },
      }),
    ].join('\n') + '\n')

    createStoreDb(dbPath)
    // Row 1 carries reasoning tokens: they are a subset of the session's
    // per-turn output and must ride as metadata WITHOUT entering the
    // query-path cost recompute (cachedCallToApiCall discards the parser's
    // costUSD for copilot and re-derives from tokens — the assertion below
    // is the only guard that exercises that production path).
    insertStoreRow(dbPath, 'sess-store', 12000, 10000, 1500, at(12), 40) // input 500
    insertStoreRow(dbPath, 'sess-store', 8000, 7000, 900, at(15))       // input 100

    const sumUsage = (projects: Awaited<ReturnType<typeof parseAllSessions>>) => {
      const calls = projects.flatMap(p => p.sessions).flatMap(s => s.turns).flatMap(t => t.assistantCalls)
      return {
        input: calls.reduce((s, c) => s + c.usage.inputTokens, 0),
        cacheRead: calls.reduce((s, c) => s + c.usage.cacheReadInputTokens, 0),
        cacheWrite: calls.reduce((s, c) => s + c.usage.cacheCreationInputTokens, 0),
        output: calls.reduce((s, c) => s + c.usage.outputTokens, 0),
        cost: calls.reduce((s, c) => s + c.costUSD, 0),
      }
    }

    // The reasoning-free cost of everything above: per-turn output plus the
    // two store rows priced on input/cache alone. A higher observed cost
    // means the 40 reasoning tokens were billed at the output rate on a call
    // that owns no output — double-billing them against the per-turn call.
    const expectedCost =
      calculateCost('claude-sonnet-4-5', 0, 17, 0, 0, 0) +
      calculateCost('claude-sonnet-4-5', 500, 0, 1500, 10000, 0) +
      calculateCost('claude-sonnet-4-5', 100, 0, 900, 7000, 0)

    // First parse: input/cache equal the DB rows exactly (the rollup is
    // reconciled away, residual zero); output stays with the per-turn event.
    const first = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(first.cost).toBeCloseTo(expectedCost, 12)
    expect(first).toEqual({ input: 600, cacheRead: 17000, cacheWrite: 2400, output: 17, cost: first.cost })

    // The session continues: one more API request lands as one more row.
    // Re-parse against the warm disk cache — the durable merge must append
    // only the new row's key, keeping totals equal to the DB.
    clearSessionCache()
    insertStoreRow(dbPath, 'sess-store', 5000, 4600, 300, at(30))    // input 100

    const second = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(second.cost).toBeCloseTo(expectedCost + calculateCost('claude-sonnet-4-5', 100, 0, 300, 4600, 0), 12)
    expect(second).toEqual({ input: 700, cacheRead: 21600, cacheWrite: 2700, output: 17, cost: second.cost })
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (k) Serve-time precedence heals a durably double-cached session
// ═══════════════════════════════════════════════════════════════════════════
// The union merge never deletes, so a rollup cached while the store was
// unreadable (an unsupported-schema epoch, a runtime without node:sqlite,
// restored files) survives the store later becoming readable — and its
// session's rows would then be cached beside it. Serve-time precedence must
// drop the rollup calls whenever store calls exist for the session, healing
// the state instead of double-counting it forever.
describe.skipIf(!isSqliteAvailable())('(k) serve-time precedence over stale cached rollups', () => {
  it('stops counting a cached rollup once the store covers its session', async () => {
    const sessionStateDir = join(tmpHome, 'session-state')
    await mkdir(sessionStateDir, { recursive: true })
    const dbPath = join(tmpHome, 'session-store.db')
    vi.stubEnv('CODEBURN_COPILOT_SESSION_STATE_DIR', sessionStateDir)
    vi.stubEnv('CODEBURN_COPILOT_SESSION_STORE_DB', dbPath)
    vi.stubEnv('CODEBURN_COPILOT_DISABLE_OTEL', '1')
    vi.stubEnv('CODEBURN_COPILOT_WS_STORAGE_DIR', join(tmpHome, 'no-ws'))
    vi.stubEnv('CODEBURN_COPILOT_GLOBAL_STORAGE_DIR', join(tmpHome, 'no-global'))
    vi.stubEnv('CODEBURN_COPILOT_JETBRAINS_DIR', join(tmpHome, 'no-jb'))

    const base = Date.now() - 5 * 24 * 60 * 60 * 1000
    const at = (offsetSec: number): string => new Date(base + offsetSec * 1000).toISOString()
    const dir = join(sessionStateDir, 'sess-stale')
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, 'workspace.yaml'), 'id: sess-stale\ncwd: /home/user/testproj\n')
    await writeFile(join(dir, 'events.jsonl'), [
      JSON.stringify({ type: 'session.model_change', timestamp: at(0), data: { newModel: 'claude-sonnet-4-5' } }),
      JSON.stringify({ type: 'assistant.message', timestamp: at(10), data: { messageId: 'msg-1', outputTokens: 25, toolRequests: [] } }),
      JSON.stringify({
        type: 'session.shutdown',
        timestamp: at(20),
        data: {
          shutdownType: 'routine',
          modelMetrics: {
            'claude-sonnet-4-5': {
              requests: { count: 1, cost: 1 },
              usage: { inputTokens: 20000, outputTokens: 25, cacheReadTokens: 17000, cacheWriteTokens: 2400, reasoningTokens: 0 },
            },
          },
        },
      }),
    ].join('\n') + '\n')

    const sumUsage = (projects: Awaited<ReturnType<typeof parseAllSessions>>) => {
      const calls = projects.flatMap(p => p.sessions).flatMap(s => s.turns).flatMap(t => t.assistantCalls)
      return {
        input: calls.reduce((s, c) => s + c.usage.inputTokens, 0),
        cacheRead: calls.reduce((s, c) => s + c.usage.cacheReadInputTokens, 0),
        output: calls.reduce((s, c) => s + c.usage.outputTokens, 0),
      }
    }

    // Run 1: no store exists — the rollup is legitimately the only record
    // and gets durably cached.
    const first = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(first).toEqual({ input: 600, cacheRead: 17000, output: 25 })

    // The store now becomes readable WITH rows for the same session — the
    // uncovered→covered transition no writer ordering protects (schema
    // epoch ending, node:sqlite appearing, restored files). events.jsonl is
    // unchanged, so its cached rollup calls survive the merge untouched.
    clearSessionCache()
    createStoreDb(dbPath)
    insertStoreRow(dbPath, 'sess-stale', 12000, 10000, 1500, at(12)) // input 500
    insertStoreRow(dbPath, 'sess-stale', 8000, 7000, 900, at(15))    // input 100

    // Run 2: totals must equal per-turn output + store rows — the cached
    // rollup (input 600 / cacheRead 17,000) must not ALSO count.
    const second = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(second).toEqual({ input: 600, cacheRead: 17000, output: 25 })

    // Run 3 (warm disk cache, nothing changed): still healed, still once.
    clearSessionCache()
    const third = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(third).toEqual({ input: 600, cacheRead: 17000, output: 25 })
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (l) Age-out never expires the still-discovered session store
// ═══════════════════════════════════════════════════════════════════════════
// An idle machine whose session-store rows are all >90 days old: the store is
// still on disk and IS the durable record (crash-only rows have no rollup to
// fall back to), so its discovery declares retainWhilePresent and the age-out
// leaves it alone. Ordinary journal-style sources keep the pre-existing
// schedule — pruned at 90 days whether or not the file remains — and orphaned
// store entries age out normally once the DB itself is gone.
describe.skipIf(!isSqliteAvailable())('(l) age-out exempts still-discovered store data', () => {
  it('serves >90d-old store rows while the store is still on disk', async () => {
    const sessionStateDir = join(tmpHome, 'session-state')
    await mkdir(sessionStateDir, { recursive: true })
    const dbPath = join(tmpHome, 'session-store.db')
    vi.stubEnv('CODEBURN_COPILOT_SESSION_STATE_DIR', sessionStateDir)
    vi.stubEnv('CODEBURN_COPILOT_SESSION_STORE_DB', dbPath)
    vi.stubEnv('CODEBURN_COPILOT_DISABLE_OTEL', '1')
    vi.stubEnv('CODEBURN_COPILOT_WS_STORAGE_DIR', join(tmpHome, 'no-ws'))
    vi.stubEnv('CODEBURN_COPILOT_GLOBAL_STORAGE_DIR', join(tmpHome, 'no-global'))
    vi.stubEnv('CODEBURN_COPILOT_JETBRAINS_DIR', join(tmpHome, 'no-jb'))

    const base = Date.now() - 91 * 24 * 60 * 60 * 1000
    const at = (offsetSec: number): string => new Date(base + offsetSec * 1000).toISOString()
    const dir = join(sessionStateDir, 'sess-idle')
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, 'workspace.yaml'), 'id: sess-idle\ncwd: /home/user/testproj\n')
    await writeFile(join(dir, 'events.jsonl'), [
      JSON.stringify({ type: 'session.model_change', timestamp: at(0), data: { newModel: 'claude-sonnet-4-5' } }),
      JSON.stringify({ type: 'assistant.message', timestamp: at(10), data: { messageId: 'msg-1', outputTokens: 25, toolRequests: [] } }),
      JSON.stringify({
        type: 'session.shutdown',
        timestamp: at(20),
        data: {
          shutdownType: 'routine',
          modelMetrics: {
            'claude-sonnet-4-5': {
              requests: { count: 1, cost: 1 },
              usage: { inputTokens: 20000, outputTokens: 25, cacheReadTokens: 17000, cacheWriteTokens: 2400, reasoningTokens: 0 },
            },
          },
        },
      }),
    ].join('\n') + '\n')
    createStoreDb(dbPath)
    insertStoreRow(dbPath, 'sess-idle', 12000, 10000, 1500, at(12)) // input 500
    insertStoreRow(dbPath, 'sess-idle', 8000, 7000, 900, at(15))    // input 100

    const sumUsage = (projects: Awaited<ReturnType<typeof parseAllSessions>>) => {
      const calls = projects.flatMap(p => p.sessions).flatMap(s => s.turns).flatMap(t => t.assistantCalls)
      return {
        input: calls.reduce((s, c) => s + c.usage.inputTokens, 0),
        cacheRead: calls.reduce((s, c) => s + c.usage.cacheReadInputTokens, 0),
        output: calls.reduce((s, c) => s + c.usage.outputTokens, 0),
      }
    }

    // Both runs: the store rows must serve — never zero. The session's
    // events.jsonl is itself >90d old and follows the ordinary durable
    // schedule (pruned even while on disk), so its per-turn output and
    // rollup drop out; only the retainWhilePresent store keeps this
    // session's record, exactly the crash-only-rows guarantee the flag
    // exists for.
    const first = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(first).toEqual({ input: 600, cacheRead: 17000, output: 0 })
    clearSessionCache()
    const second = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(second).toEqual({ input: 600, cacheRead: 17000, output: 0 })
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// Shared setup for the serve-time precedence scenarios (m)/(n)/(o): a CLI
// session-state dir + a session-store.db, all env-pinned into tmpHome.
// ═══════════════════════════════════════════════════════════════════════════
async function setupCopilotStoreEnv(): Promise<{
  dbPath: string
  at: (offsetSec: number) => string
  writeSession: (sessionId: string, opts: { output: number; rollup?: boolean }) => Promise<string>
  sumUsage: (projects: Awaited<ReturnType<typeof parseAllSessions>>) => { input: number; cacheRead: number; cacheWrite: number; output: number }
}> {
  const sessionStateDir = join(tmpHome, 'session-state')
  await mkdir(sessionStateDir, { recursive: true })
  const dbPath = join(tmpHome, 'session-store.db')
  vi.stubEnv('CODEBURN_COPILOT_SESSION_STATE_DIR', sessionStateDir)
  vi.stubEnv('CODEBURN_COPILOT_SESSION_STORE_DB', dbPath)
  vi.stubEnv('CODEBURN_COPILOT_DISABLE_OTEL', '1')
  vi.stubEnv('CODEBURN_COPILOT_WS_STORAGE_DIR', join(tmpHome, 'no-ws'))
  vi.stubEnv('CODEBURN_COPILOT_GLOBAL_STORAGE_DIR', join(tmpHome, 'no-global'))
  vi.stubEnv('CODEBURN_COPILOT_JETBRAINS_DIR', join(tmpHome, 'no-jb'))

  const base = Date.now() - 5 * 24 * 60 * 60 * 1000
  const at = (offsetSec: number): string => new Date(base + offsetSec * 1000).toISOString()

  // The rollup always uses the maintainer's repro numbers: cache-inclusive
  // input 20,000 → uncached 600, cacheRead 17,000, cacheWrite 2,400.
  const writeSession = async (sessionId: string, opts: { output: number; rollup?: boolean }): Promise<string> => {
    const dir = join(sessionStateDir, sessionId)
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, 'workspace.yaml'), `id: ${sessionId}\ncwd: /home/user/testproj\n`)
    const lines = [
      JSON.stringify({ type: 'session.model_change', timestamp: at(0), data: { newModel: 'claude-sonnet-4-5' } }),
      JSON.stringify({ type: 'assistant.message', timestamp: at(10), data: { messageId: 'msg-1', outputTokens: opts.output, toolRequests: [] } }),
    ]
    if (opts.rollup) {
      lines.push(JSON.stringify({
        type: 'session.shutdown',
        timestamp: at(20),
        data: {
          shutdownType: 'routine',
          modelMetrics: {
            'claude-sonnet-4-5': {
              requests: { count: 1, cost: 1 },
              usage: { inputTokens: 20000, outputTokens: opts.output, cacheReadTokens: 17000, cacheWriteTokens: 2400, reasoningTokens: 0 },
            },
          },
        },
      }))
    }
    const eventsPath = join(dir, 'events.jsonl')
    await writeFile(eventsPath, lines.join('\n') + '\n')
    return eventsPath
  }

  const sumUsage = (projects: Awaited<ReturnType<typeof parseAllSessions>>) => {
    const calls = projects.flatMap(p => p.sessions).flatMap(s => s.turns).flatMap(t => t.assistantCalls)
    return {
      input: calls.reduce((s, c) => s + c.usage.inputTokens, 0),
      cacheRead: calls.reduce((s, c) => s + c.usage.cacheReadInputTokens, 0),
      cacheWrite: calls.reduce((s, c) => s + c.usage.cacheCreationInputTokens, 0),
      output: calls.reduce((s, c) => s + c.usage.outputTokens, 0),
    }
  }

  return { dbPath, at, writeSession, sumUsage }
}

// ═══════════════════════════════════════════════════════════════════════════
// (m) The probe-to-parse race, at serve level: rows commit, THEN the shutdown
//     line lands — counted once, store side wins
// ═══════════════════════════════════════════════════════════════════════════
// The #946 round-2 repro. Under parse-time suppression, a coverage snapshot
// taken at discovery went stale the moment a session ended between probe and
// parse (rows commit BEFORE the shutdown line is appended), and the rollup
// was emitted beside the rows — doubling input 100 / cacheRead 8,000 /
// cacheWrite 2,000 durably. Serve-time precedence has no snapshot to go
// stale: whatever store rows made it into the serve set drop the session's
// rollups, no matter when either side was parsed or cached.
describe.skipIf(!isSqliteAvailable())('(m) rows-then-shutdown race counted once at serve time', () => {
  it('drops the rollup cached after its session ended mid-run', async () => {
    const { dbPath, at, writeSession, sumUsage } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    const eventsPath = await writeSession('sess-race', { output: 345 })

    // Run 1 parses the live session mid-flight: no rows, no rollup yet.
    const first = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(first).toEqual({ input: 0, cacheRead: 0, cacheWrite: 0, output: 345 })

    // The session ends: rows committed FIRST (the CLI's write order), the
    // shutdown rollup appended second, both between two codeburn runs.
    clearSessionCache()
    insertStoreRow(dbPath, 'sess-race', 10100, 8000, 2000, at(15)) // input 100
    await writeFile(eventsPath, JSON.stringify({
      type: 'session.shutdown',
      timestamp: at(20),
      data: {
        shutdownType: 'routine',
        modelMetrics: {
          'claude-sonnet-4-5': {
            requests: { count: 1, cost: 1 },
            usage: { inputTokens: 10100, outputTokens: 345, cacheReadTokens: 8000, cacheWriteTokens: 2000, reasoningTokens: 0 },
          },
        },
      },
    }) + '\n', { flag: 'a' })

    // Both representations parse and cache; the serve set holds the row, so
    // the rollup is dropped: 100/8,000/2,000 once, not twice.
    const second = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(second).toEqual({ input: 100, cacheRead: 8000, cacheWrite: 2000, output: 345 })

    // Warm re-run: still once.
    clearSessionCache()
    const third = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(third).toEqual({ input: 100, cacheRead: 8000, cacheWrite: 2000, output: 345 })
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (n) A store holding no billable rows for a session never suppresses it
// ═══════════════════════════════════════════════════════════════════════════
// Two scenarios collapse into this serve-set shape. (1) Atomic replacement:
// discovery probes store A, a writer renames store B over it, the parse
// reads B — under a parse-time coverage snapshot, sessions covered by A but
// absent from B would lose their rollups AND their rows; at serve time only
// what the parse actually produced suppresses. (2) The billable predicate:
// all-zero rows emit no calls, so a session with only those must keep its
// rollup — suppression on mere row-existence would zero its input/cache.
describe.skipIf(!isSqliteAvailable())('(n) no billable store rows → the rollup still counts', () => {
  it('keeps the rollup when the served store holds only zero-usage rows for its session', async () => {
    const { dbPath, at, writeSession, sumUsage } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    // sess-r: only an all-zero row (emits nothing). sess-other: billable.
    insertStoreRow(dbPath, 'sess-r', 0, 0, 0, at(5))
    insertStoreRow(dbPath, 'sess-other', 5050, 5000, 0, at(6)) // input 50
    await writeSession('sess-r', { output: 25, rollup: true })

    const totals = sumUsage(await parseAllSessions(undefined, 'copilot'))
    // sess-r's rollup (600/17,000/2,400) + sess-other's row (50/5,000/0)
    // + sess-r's per-turn output.
    expect(totals).toEqual({ input: 650, cacheRead: 22000, cacheWrite: 2400, output: 25 })
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (o) Store-absence epoch: deleting the store changes nothing served
// ═══════════════════════════════════════════════════════════════════════════
// The round-6 finding on the previous design: gating reconciliation on the
// store being DISCOVERED made an absence epoch flip served totals — the
// orphaned rows and the returning rollup both counted, and a daily history
// finalized during the epoch kept the doubled day forever even after the
// store came back. Reconciliation therefore reads only the cached serve set:
// the rows (durable orphans included) keep replacing the rollup, totals are
// identical before and after the deletion, and sealed history can never
// flip. The orphaned rows remain the session's record until the 90-day
// age-out prunes them — at which point the rollup calls, still cached in
// events.jsonl's entry, become the record again with the same exactly-once
// guarantee.
describe.skipIf(!isSqliteAvailable())('(o) absence epoch: served totals are independent of store presence', () => {
  it('serves identical totals before and after the store file is deleted', async () => {
    const { dbPath, at, writeSession, sumUsage } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    insertStoreRow(dbPath, 'sess-e', 12000, 10000, 1500, at(12)) // input 500
    insertStoreRow(dbPath, 'sess-e', 8000, 7000, 900, at(15))    // input 100
    await writeSession('sess-e', { output: 25, rollup: true })

    // Store present: rows win, rollup reconciled away.
    const first = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(first).toEqual({ input: 600, cacheRead: 17000, cacheWrite: 2400, output: 25 })

    // The store vanishes; its cached rows become durable orphans.
    clearSessionCache()
    await rm(dbPath, { force: true })

    // Nothing changes: the cached rows are still the session's record, the
    // rollup stays reconciled away, and no epoch can double-count a day.
    const second = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(second).toEqual({ input: 600, cacheRead: 17000, cacheWrite: 2400, output: 25 })
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (p) The reverse race: a row committing after the store read never zeroes
//     the refresh
// ═══════════════════════════════════════════════════════════════════════════
// The #946 round-4 finding 3 shape. The store parses before events.jsonl; a
// row can commit after that read but before the jsonl parse reaches the
// shutdown line. Under parse-time suppression the live coverage re-check
// then saw the row and suppressed the rollup — against a store snapshot
// that had emitted nothing — losing the request's input/cache for the whole
// refresh. Serve-time reconciliation cannot outrun the serve set: the
// rollup stands until rows actually land, partially-landed rows are topped
// up by the residual to the rollup's own totals, and full coverage retires
// the residual — counted once at every step and zero at none.
describe.skipIf(!isSqliteAvailable())('(p) reconciliation never outruns the served store rows', () => {
  it('serves the rollup total at every stage while rows progressively land', async () => {
    const { dbPath, at, writeSession, sumUsage } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    await writeSession('sess-rev', { output: 25, rollup: true })

    // Refresh 1: the store was read before the rows committed — it holds
    // nothing for this session. The rollup is the only record and must
    // count; a zero here is the old design's lost refresh.
    const first = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(first).toEqual({ input: 600, cacheRead: 17000, cacheWrite: 2400, output: 25 })

    // The first row lands: it serves per-request, and the rollup usage it
    // does not yet represent serves once as the residual — the total never
    // dips below what the rollup proved happened.
    clearSessionCache()
    insertStoreRow(dbPath, 'sess-rev', 12000, 10000, 1500, at(12)) // input 500

    const second = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(second).toEqual({ input: 600, cacheRead: 17000, cacheWrite: 2400, output: 25 })

    // The second row completes coverage: totals unchanged, now served
    // entirely by rows — the residual has retired to zero.
    clearSessionCache()
    insertStoreRow(dbPath, 'sess-rev', 8000, 7000, 900, at(15))    // input 100

    const third = await parseAllSessions(undefined, 'copilot')
    expect(sumUsage(third)).toEqual({ input: 600, cacheRead: 17000, cacheWrite: 2400, output: 25 })
    const keys = third.flatMap(p => p.sessions).flatMap(s => s.turns).flatMap(t => t.assistantCalls).map(c => c.deduplicationKey)
    expect(keys.filter(k => k.startsWith('copilot-store:')).length).toBe(2)
    expect(keys.some(k => k.includes(':shutdown-residual:'))).toBe(false)
    expect(keys.some(k => k.includes(':shutdown:'))).toBe(false)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (s) Behavioral weight: supplementary accounting never fabricates activity
// ═══════════════════════════════════════════════════════════════════════════
// The round-6 finding 1 + the maintainer's weight contract. A shutdown rollup
// is aggregate accounting, never a request; a store row is a real request but
// pairs 1:1 with its served per-turn call when one exists. Blanket weight 1
// fabricated apiCalls/turns/modelCalls for every covered request; blanket
// weight 0 would hide crash-recovered requests. The pinned contract:
// JSONL + matching row = 1 call/turn; JSONL + rollup = the JSONL count only;
// store-only request = 1 call; rollup-only = 0 calls with full usage.
describe.skipIf(!isSqliteAvailable())('(s) behavioral weight of store rows and rollups', () => {
  it('pins call/turn weight across the four review scenarios', async () => {
    const { dbPath, at, writeSession } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)

    // A: one real request, both representations served.
    await writeSession('sess-paired', { output: 25 })
    insertStoreRow(dbPath, 'sess-paired', 12000, 10000, 1500, at(12))
    // B: pre-store session — per-turn call + rollup, no rows.
    await writeSession('sess-rollup', { output: 30, rollup: true })
    // C: store-only request (crash lost the per-turn call; no session-state).
    insertStoreRow(dbPath, 'sess-crash', 8000, 7000, 900, at(15))

    const sessions = (await parseAllSessions(undefined, 'copilot')).flatMap(p => p.sessions)
    const byId = new Map(sessions.map(s => [s.sessionId, s]))
    const modelCalls = (s: NonNullable<ReturnType<typeof byId.get>>): number =>
      Object.values(s.modelBreakdown).reduce((sum, m) => sum + m.calls, 0)

    const paired = byId.get('sess-paired')!
    expect(paired.apiCalls).toBe(1)
    expect(paired.turns.length).toBe(1)
    expect(modelCalls(paired)).toBe(1)
    expect(paired.totalInputTokens).toBe(500)     // the row's tokens count in full
    expect(paired.totalCacheReadTokens).toBe(10000)
    expect(paired.totalOutputTokens).toBe(25)

    const withRollup = byId.get('sess-rollup')!
    expect(withRollup.apiCalls).toBe(1)           // the per-turn call only
    expect(withRollup.turns.length).toBe(1)
    expect(modelCalls(withRollup)).toBe(1)
    expect(withRollup.totalInputTokens).toBe(600) // rollup tokens retained

    const crash = byId.get('sess-crash')!
    expect(crash.apiCalls).toBe(1)                // a real, store-only request
    expect(crash.turns.length).toBe(1)
    expect(crash.totalInputTokens).toBe(100)
  })

  it('pairs rows per model: a subagent row without its per-turn call still counts', async () => {
    const { dbPath, at, writeSession } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    // One sonnet request served both ways, plus a haiku subagent request
    // whose per-turn call was lost — the haiku row must not pair against the
    // sonnet call.
    await writeSession('sess-multi', { output: 25 })                    // sonnet per-turn call
    insertStoreRow(dbPath, 'sess-multi', 12000, 10000, 1500, at(12))    // sonnet row → paired
    const { DatabaseSync } = requireForTest('node:sqlite') as { DatabaseSync: new (path: string) => TestDb }
    const db = new DatabaseSync(dbPath)
    db.prepare(
      `INSERT INTO assistant_usage_events
         (session_id, model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, created_at)
       VALUES ('sess-multi', 'claude-haiku-4-5', 5050, 0, 5000, 0, 0, ?)`
    ).run(at(14))
    db.close()

    const session = (await parseAllSessions(undefined, 'copilot')).flatMap(p => p.sessions)
      .find(s => s.sessionId === 'sess-multi')!
    expect(session.apiCalls).toBe(2)   // the sonnet request + the haiku request
    const calls = Object.fromEntries(Object.entries(session.modelBreakdown).map(([m, b]) => [m, b.calls]))
    expect(Object.values(calls).reduce((a, b) => a + b, 0)).toBe(2)
    expect(session.totalInputTokens).toBe(500 + 50)
  })

  it('serves a rollup-only session with zero call weight but full usage', async () => {
    const { dbPath, at } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    // events.jsonl holding ONLY the shutdown rollup — no per-turn events
    // survived. Its tokens are real; its "calls" are not.
    const dir = join(tmpHome, 'session-state', 'sess-agg')
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, 'workspace.yaml'), 'id: sess-agg\ncwd: /home/user/testproj\n')
    await writeFile(join(dir, 'events.jsonl'), [
      JSON.stringify({ type: 'session.model_change', timestamp: at(0), data: { newModel: 'claude-sonnet-4-5' } }),
      JSON.stringify({
        type: 'session.shutdown',
        timestamp: at(20),
        data: {
          shutdownType: 'routine',
          modelMetrics: {
            'claude-sonnet-4-5': {
              requests: { count: 2, cost: 1 },
              usage: { inputTokens: 20000, outputTokens: 0, cacheReadTokens: 17000, cacheWriteTokens: 2400, reasoningTokens: 0 },
            },
          },
        },
      }),
    ].join('\n') + '\n')

    const sessions = (await parseAllSessions(undefined, 'copilot')).flatMap(p => p.sessions)
    const agg = sessions.find(s => s.sessionId === 'sess-agg')
    expect(agg).toBeDefined()
    expect(agg!.apiCalls).toBe(0)
    expect(agg!.totalInputTokens).toBe(600)
    expect(agg!.totalCacheReadTokens).toBe(17000)
    expect(agg!.totalCostUSD).toBeGreaterThan(0)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (t) A deferred store read marks hydration incomplete
// ═══════════════════════════════════════════════════════════════════════════
// The round-6 finding 2. A changed store whose read defers (busy, EACCES —
// every retryable shape funnels through the same busy-shaped throw) keeps
// serving its cached rows, but the data the read would have added is missing
// from the pass: reporting hydration complete would let the daily backfill
// finalize history without it. The fence must hold until the read lands.
describe.skipIf(!isSqliteAvailable())('(t) deferred store read marks hydration incomplete', () => {
  it.skipIf(process.getuid?.() === 0)('holds the fence while the changed store is unreadable, then recovers', async () => {
    const { chmod } = await import('fs/promises')
    const { dbPath, at, writeSession, sumUsage } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    insertStoreRow(dbPath, 'sess-h', 12000, 10000, 1500, at(12)) // input 500
    await writeSession('sess-h', { output: 25 })

    const first = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(first).toEqual({ input: 500, cacheRead: 10000, cacheWrite: 1500, output: 25 })
    expect(isSessionHydrationComplete()).toBe(true)

    // The store grows, then becomes unreadable before the refresh reads it.
    clearSessionCache()
    insertStoreRow(dbPath, 'sess-h', 8000, 7000, 900, at(20))    // input 100
    await chmod(dbPath, 0o000)
    try {
      const second = sumUsage(await parseAllSessions(undefined, 'copilot'))
      // Cached rows keep serving — the deferral is invisible in totals...
      expect(second).toEqual({ input: 500, cacheRead: 10000, cacheWrite: 1500, output: 25 })
      // ...but the pass must not claim full hydration: a row is missing.
      expect(isSessionHydrationComplete()).toBe(false)
    } finally {
      await chmod(dbPath, 0o644)
    }

    // Readable again: the deferred row lands and the fence lifts.
    clearSessionCache()
    const third = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(third).toEqual({ input: 600, cacheRead: 17000, cacheWrite: 2400, output: 25 })
    expect(isSessionHydrationComplete()).toBe(true)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (t2) The fence actually HOLDS THE DAILY WATERMARK, end to end
// ═══════════════════════════════════════════════════════════════════════════
// The maintainer's refinement asked for a production-path regression proving
// "busy store + unchanged cached JSONL => hydration incomplete AND daily
// watermark held" — the flag alone is not the contract. This wires the REAL
// isSessionHydrationComplete into ensureCacheHydrated (no stub) and asserts
// the daily cache refuses to finalize while the store's re-read is deferred.
describe.skipIf(!isSqliteAvailable())('(t2) deferred store holds the daily watermark', () => {
  it.skipIf(process.getuid?.() === 0)('leaves the daily cache incomplete until the store is readable', async () => {
    const { chmod } = await import('fs/promises')
    const { dbPath, at, writeSession } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    insertStoreRow(dbPath, 'sess-wm', 12000, 10000, 1500, at(12))
    await writeSession('sess-wm', { output: 25 })

    // Seed a daily cache whose watermark is deliberately BEHIND yesterday, so
    // every run below faces a real backfill gap to seal while the watermark is
    // non-null and its preservation is observable. The session cache stays
    // warm and the JSONL unchanged — the reviewer's shape, where no
    // session-state parser runs and only the store's own deferral can reach
    // the fence.
    const dayStr = (n: number): string => {
      const d = new Date()
      d.setDate(d.getDate() - n)
      return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
    }
    const seededWatermark = dayStr(5)
    const seedDailyCache = async () => {
      await saveDailyCache({
        version: DAILY_CACHE_VERSION,
        savingsConfigHash: '',
        tzKey: currentTzKey(),
        lastComputedDate: seededWatermark,
        complete: true,
        watermarkTrusted: true,
        days: [],
      })
    }
    const hydrate = async () => {
      clearSessionCache()
      return ensureCacheHydrated(
        range => parseAllSessions(range, 'copilot'),
        aggregateProjectsIntoDays,
        '',
        // The production wiring: the real flag, not a stub.
        isSessionHydrationComplete,
      )
    }

    // The store grows and turns unreadable before the refresh reads it.
    await seedDailyCache()
    insertStoreRow(dbPath, 'sess-wm', 8000, 7000, 900, at(20))
    await chmod(dbPath, 0o000)
    let degraded: Awaited<ReturnType<typeof hydrate>>
    try {
      degraded = await hydrate()
      expect(isSessionHydrationComplete()).toBe(false)
      // The contract the reviewer asked for: history does not seal, and the
      // watermark is HELD where it was — not advanced by a parse that never
      // read the new row, and not reset either.
      expect(degraded.complete).toBe(false)
      expect(degraded.lastComputedDate).toBe(seededWatermark)
    } finally {
      await chmod(dbPath, 0o644)
    }

    // Readable again — and healing must happen IN PLACE, against the same
    // persisted incomplete cache the degraded run left behind (no reseed):
    // the deferred row lands, the day seals, and the watermark advances.
    const healed = await hydrate()
    expect(isSessionHydrationComplete()).toBe(true)
    expect(healed.complete).toBe(true)
    expect(healed.lastComputedDate).not.toBe(seededWatermark)
    expect(healed.lastComputedDate! > seededWatermark).toBe(true)
    // The row deferred during the outage is in the sealed history.
    const sealed = healed.days.reduce((s, d) => s + (d.providers['copilot']?.inputTokens ?? 0), 0)
    expect(sealed).toBe(600)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (u) A project learned after rows were cached cannot split the session
// ═══════════════════════════════════════════════════════════════════════════
// The round-6 companion finding. Store rows cached before their session's
// session-state dir existed carry the store's cwd-derived project; when
// events.jsonl later appears (workspace.yaml cwd), the store file itself may
// be unchanged — its cached calls are reused verbatim. Project is part of the
// session grouping key, so without serve-time unification one real session
// splits in two. The serve pass rewrites cached store calls to the
// session-state project whenever the serve set knows it.
describe.skipIf(!isSqliteAvailable())('(u) late-learned project identity unifies cached store rows', () => {
  it('serves one session under the session-state project after it appears', async () => {
    const { dbPath, at, writeSession } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    insertStoreRow(dbPath, 'sess-late', 12000, 10000, 1500, at(12), 0, '/home/user/storeproj')

    const s1 = (await parseAllSessions(undefined, 'copilot')).flatMap(p => p.sessions)
      .find(s => s.sessionId === 'sess-late')!
    expect(s1.project).toBe('storeproj')

    // The session-state dir appears; the store file is UNCHANGED, so its
    // cached call still carries 'storeproj' until serve-time unification.
    clearSessionCache()
    await writeSession('sess-late', { output: 25 })

    const sessions = (await parseAllSessions(undefined, 'copilot')).flatMap(p => p.sessions)
      .filter(s => s.sessionId === 'sess-late')
    expect(sessions).toHaveLength(1)
    expect(sessions[0]!.project).toBe('testproj')
    expect(sessions[0]!.totalInputTokens).toBe(500)
    expect(sessions[0]!.totalOutputTokens).toBe(25)
    expect(sessions[0]!.apiCalls).toBe(1)
  })

  it('keeps one session when the session-state dir is pruned and the jsonl orphans', async () => {
    // The reverse shape (round-6.5 finding): events.jsonl becomes a durable
    // orphan, losing its source.project, while the store rows live on. Both
    // sides must adopt the surviving label — the store's own — or one real
    // request serves as two sessions with doubled call weight.
    const { dbPath, at, writeSession } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    insertStoreRow(dbPath, 'sess-prune', 12000, 10000, 1500, at(12), 0, '/home/user/storeproj')
    await writeSession('sess-prune', { output: 25 })

    const s1 = (await parseAllSessions(undefined, 'copilot')).flatMap(p => p.sessions)
      .filter(s => s.sessionId === 'sess-prune')
    expect(s1).toHaveLength(1)
    expect(s1[0]!.apiCalls).toBe(1)

    clearSessionCache()
    await rm(join(tmpHome, 'session-state', 'sess-prune'), { recursive: true, force: true })

    const s2 = (await parseAllSessions(undefined, 'copilot')).flatMap(p => p.sessions)
      .filter(s => s.sessionId === 'sess-prune')
    expect(s2).toHaveLength(1)
    // The rows were cached carrying the jsonl-derived label, so the session
    // keeps it even after the jsonl orphans — stable, and never split.
    expect(s2[0]!.project).toBe('testproj')
    expect(s2[0]!.apiCalls).toBe(1)
    expect(s2[0]!.totalInputTokens).toBe(500)
    expect(s2[0]!.totalOutputTokens).toBe(25)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (v) A same-path store reset cannot swallow new usage
// ═══════════════════════════════════════════════════════════════════════════
// The round-6 finding 4. Recreating the DB at the same path restarts the
// AUTOINCREMENT sequence; under a bare <sid>:<rowId> key the durable union
// rejected the recreated row as already-cached and its usage vanished. The
// content-discriminated key admits it; the original row remains served as
// real past usage (the durable contract: cached history never shrinks).
describe.skipIf(!isSqliteAvailable())('(v) same-path store reset', () => {
  it('serves the recreated row alongside the durable original', async () => {
    const { dbPath, at, sumUsage } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    insertStoreRow(dbPath, 'sess-reset', 12000, 10000, 1500, at(12)) // input 500
    const first = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(first).toEqual({ input: 500, cacheRead: 10000, cacheWrite: 1500, output: 0 })

    clearSessionCache()
    await rm(dbPath, { force: true })
    createStoreDb(dbPath)
    insertStoreRow(dbPath, 'sess-reset', 8000, 7000, 900, at(30))    // row id 1 again
    const second = sumUsage(await parseAllSessions(undefined, 'copilot'))
    expect(second).toEqual({ input: 600, cacheRead: 17000, cacheWrite: 2400, output: 0 })
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (w) Mixed coverage: a crash tail cannot cancel a leg's missing rows
// ═══════════════════════════════════════════════════════════════════════════
// The round-6.5 converging finding (grok + gpt-5.6-sol independently). A
// lifetime `max(0, rollup − allRows)` residual lets store rows the rollup
// never covered (a crash tail, a later reset) cancel usage the rollup DID
// cover whose rows are missing. Residuals are therefore computed per rollup
// leg over only the rows in that leg's interval — rows commit before their
// leg's shutdown line, so a row after the leg belongs to the tail, never to
// the subtraction.
describe.skipIf(!isSqliteAvailable())('(w) per-leg residual under mixed coverage', () => {
  it('serves the covered gap AND the crash tail in full', async () => {
    const { dbPath, at } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)

    const dir = join(tmpHome, 'session-state', 'sess-mix')
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, 'workspace.yaml'), 'id: sess-mix\ncwd: /home/user/testproj\n')
    await writeFile(join(dir, 'events.jsonl'), [
      JSON.stringify({ type: 'session.model_change', timestamp: at(0), data: { newModel: 'claude-sonnet-4-5' } }),
      JSON.stringify({ type: 'assistant.message', timestamp: at(5), data: { messageId: 'msg-1', outputTokens: 25, toolRequests: [] } }),
      // The rollup covers requests R1+R2 (cache-inclusive input 10,200 →
      // uncached 200, cacheRead 10,000).
      JSON.stringify({
        type: 'session.shutdown',
        timestamp: at(20),
        data: {
          shutdownType: 'routine',
          modelMetrics: {
            'claude-sonnet-4-5': {
              requests: { count: 2, cost: 1 },
              usage: { inputTokens: 10200, outputTokens: 25, cacheReadTokens: 10000, cacheWriteTokens: 0, reasoningTokens: 0 },
            },
          },
        },
      }),
    ].join('\n') + '\n')

    // R1's row exists (input 100 / cacheRead 5,000); R2's row is missing
    // (pre-store leg). R3 is a crash-tail row AFTER the shutdown (input 300)
    // that no rollup ever covered.
    insertStoreRow(dbPath, 'sess-mix', 5100, 5000, 0, at(10))  // R1
    insertStoreRow(dbPath, 'sess-mix', 300, 0, 0, at(30))      // R3, after the leg

    const session = (await parseAllSessions(undefined, 'copilot')).flatMap(p => p.sessions)
      .find(s => s.sessionId === 'sess-mix')!
    // Truth: R1 (100/5,000) + R2 (100/5,000, via the leg residual) + R3 (300).
    // A lifetime subtraction would have served input 400 — R3's crash input
    // cancelling R2's missing row.
    expect(session.totalInputTokens).toBe(500)
    expect(session.totalCacheReadTokens).toBe(10000)
    expect(session.totalOutputTokens).toBe(25)
    // Weight: the per-turn call + the unpaired crash row.
    expect(session.apiCalls).toBe(2)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (x) Per-leg residuals keep each leg's gap on that leg's own day
// ═══════════════════════════════════════════════════════════════════════════
// Round-6.5: one residual stamped at the LAST leg moved every earlier leg's
// uncovered usage onto the final day — day 1 sealed 0, day 2 sealed double.
// Each leg's residual is anchored at that leg's own timestamp.
describe.skipIf(!isSqliteAvailable())('(x) multi-leg residual day attribution', () => {
  it('serves each leg\'s uncovered usage on its own day', async () => {
    const dayN = Date.now() - 5 * 24 * 60 * 60 * 1000
    const at = (offsetHours: number): string => new Date(dayN + offsetHours * 3600 * 1000).toISOString()
    const sessionStateDir = join(tmpHome, 'session-state')
    await mkdir(sessionStateDir, { recursive: true })
    const dbPath = join(tmpHome, 'session-store.db')
    vi.stubEnv('CODEBURN_COPILOT_SESSION_STATE_DIR', sessionStateDir)
    vi.stubEnv('CODEBURN_COPILOT_SESSION_STORE_DB', dbPath)
    vi.stubEnv('CODEBURN_COPILOT_DISABLE_OTEL', '1')
    vi.stubEnv('CODEBURN_COPILOT_WS_STORAGE_DIR', join(tmpHome, 'no-ws'))
    vi.stubEnv('CODEBURN_COPILOT_GLOBAL_STORAGE_DIR', join(tmpHome, 'no-global'))
    vi.stubEnv('CODEBURN_COPILOT_JETBRAINS_DIR', join(tmpHome, 'no-jb'))

    // A resumed session: leg 1 shuts down on day 1 (cumulative input 500,
    // cache-free for arithmetic clarity), leg 2 the next day (cumulative 800
    // → delta 300). The store has one leg-2 row (100); leg 1 predates the
    // store entirely and leg 2 is only partially covered, so BOTH legs carry
    // a nonzero residual on their own day.
    const dir = join(sessionStateDir, 'sess-legs')
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, 'workspace.yaml'), 'id: sess-legs\ncwd: /home/user/testproj\n')
    const rollup = (ts: string, cumulativeInput: number): string => JSON.stringify({
      type: 'session.shutdown',
      timestamp: ts,
      data: {
        shutdownType: 'routine',
        modelMetrics: {
          'claude-sonnet-4-5': {
            requests: { count: 1, cost: 1 },
            usage: { inputTokens: cumulativeInput, outputTokens: 10, cacheReadTokens: 0, cacheWriteTokens: 0, reasoningTokens: 0 },
          },
        },
      },
    })
    await writeFile(join(dir, 'events.jsonl'), [
      JSON.stringify({ type: 'session.model_change', timestamp: at(0), data: { newModel: 'claude-sonnet-4-5' } }),
      JSON.stringify({ type: 'assistant.message', timestamp: at(1), data: { messageId: 'msg-1', outputTokens: 10, toolRequests: [] } }),
      rollup(at(2), 500),
      rollup(at(26), 800),
    ].join('\n') + '\n')

    createStoreDb(dbPath)
    insertStoreRow(dbPath, 'sess-legs', 100, 0, 0, at(25))

    const inputIn = async (startH: number, endH: number): Promise<number> => {
      const projects = await parseAllSessions(
        { start: new Date(dayN + startH * 3600 * 1000), end: new Date(dayN + endH * 3600 * 1000) }, 'copilot')
      return projects.flatMap(p => p.sessions).flatMap(s => s.turns).flatMap(t => t.assistantCalls)
        .reduce((s, c) => s + c.usage.inputTokens, 0)
    }

    expect(await inputIn(-1, 12)).toBe(500)   // day 1: leg 1's gap, at leg 1's stamp
    expect(await inputIn(12, 36)).toBe(300)   // day 2: the row (100) + leg 2's residual (200)
    expect(await inputIn(-1, 36)).toBe(800)   // both: exactly once

    // A range that excludes every behavioral turn (the per-turn call) but
    // holds residual turns from different days: they must stay SEPARATE
    // turns on their own days — an anchorless merge would re-anchor day-2
    // accounting onto day 1's turn stamp.
    const anchorless = (await parseAllSessions(
      { start: new Date(dayN + 1.5 * 3600 * 1000), end: new Date(dayN + 36 * 3600 * 1000) }, 'copilot'))
      .flatMap(p => p.sessions).filter(s => s.sessionId === 'sess-legs')
    const turnDays = anchorless.flatMap(s => s.turns).map(t => new Date(t.timestamp).toISOString().slice(0, 10))
    expect(new Set(turnDays).size).toBeGreaterThanOrEqual(2)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (w2) Residual edge shapes: equal-timestamp legs, unparseable leg stamps,
//      and crash rows outside the pairing window
// ═══════════════════════════════════════════════════════════════════════════
describe.skipIf(!isSqliteAvailable())('(w2) residual and pairing edge shapes', () => {
  it('coalesces equal-timestamp legs so their shared rows subtract once', async () => {
    const { dbPath, at } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    const dir = join(tmpHome, 'session-state', 'sess-ties')
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, 'workspace.yaml'), 'id: sess-ties\ncwd: /home/user/testproj\n')
    const rollup = (ts: string, cumulativeInput: number): string => JSON.stringify({
      type: 'session.shutdown',
      timestamp: ts,
      data: {
        shutdownType: 'routine',
        modelMetrics: {
          'claude-sonnet-4-5': {
            requests: { count: 1, cost: 1 },
            usage: { inputTokens: cumulativeInput, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0, reasoningTokens: 0 },
          },
        },
      },
    })
    // Two shutdown legs stamped the SAME second (deltas 100 and 200); the
    // store holds one 200-token row at that instant. A strict interval rule
    // hands the row to the first leg and mints a full 200 residual for the
    // second (serving 400); coalescing subtracts it once (serving 300).
    await writeFile(join(dir, 'events.jsonl'), [
      JSON.stringify({ type: 'session.model_change', timestamp: at(0), data: { newModel: 'claude-sonnet-4-5' } }),
      rollup(at(20), 100),
      rollup(at(20), 300),
    ].join('\n') + '\n')
    insertStoreRow(dbPath, 'sess-ties', 200, 0, 0, at(20))

    const input = (await parseAllSessions(undefined, 'copilot')).flatMap(p => p.sessions)
      .filter(s => s.sessionId === 'sess-ties')
      .reduce((s, x) => s + x.totalInputTokens, 0)
    expect(input).toBe(300)
  })

  it('serves a rollup with an unparseable timestamp instead of silently dropping it', async () => {
    const { dbPath, at } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    const dir = join(tmpHome, 'session-state', 'sess-badts')
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, 'workspace.yaml'), 'id: sess-badts\ncwd: /home/user/testproj\n')
    // The leg's timestamp cannot parse, so it can never enter the residual
    // sweep — dropping it because rows exist would silently lose its usage.
    // It serves unchanged (weightless); the bounded overlap with the row is
    // the documented corruption-shaped trade (over-serve, never lose).
    await writeFile(join(dir, 'events.jsonl'), [
      JSON.stringify({ type: 'session.model_change', timestamp: at(0), data: { newModel: 'claude-sonnet-4-5' } }),
      JSON.stringify({
        type: 'session.shutdown',
        timestamp: 'not-a-timestamp',
        data: {
          shutdownType: 'routine',
          modelMetrics: {
            'claude-sonnet-4-5': {
              requests: { count: 1, cost: 1 },
              usage: { inputTokens: 500, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0, reasoningTokens: 0 },
            },
          },
        },
      }),
    ].join('\n') + '\n')
    insertStoreRow(dbPath, 'sess-badts', 100, 0, 0, at(10))

    const sessions = (await parseAllSessions(undefined, 'copilot')).flatMap(p => p.sessions)
      .filter(s => s.sessionId === 'sess-badts')
    const input = sessions.reduce((s, x) => s + x.totalInputTokens, 0)
    expect(input).toBe(600)  // row 100 + un-droppable rollup 500; never 100
    expect(sessions.reduce((s, x) => s + x.apiCalls, 0)).toBe(1)  // the row; the rollup stays weightless

    // The rollup adopted a STABLE valid timestamp (its file's preceding
    // valid stamp, else the session's earliest), so a RANGED query (the
    // shape the daily backfill uses) serves the same 600 — a raw
    // unparseable stamp would be invisible to callsInRange and quietly lose
    // the 500 from every sealed day.
    const base = new Date(at(0)).getTime()
    const rangedInput = async (): Promise<number> =>
      (await parseAllSessions(
        { start: new Date(base - 3600 * 1000), end: new Date(base + 3600 * 1000) }, 'copilot'))
        .flatMap(p => p.sessions).filter(s => s.sessionId === 'sess-badts')
        .reduce((s, x) => s + x.totalInputTokens, 0)
    expect(await rangedInput()).toBe(600)

    // The session resumes a day later: the fallback must NOT move with it —
    // a day sealed with the rollup would otherwise lose it to the new day
    // and the daily union would count it twice.
    clearSessionCache()
    insertStoreRow(dbPath, 'sess-badts', 50, 0, 0, at(86400))
    expect(await rangedInput()).toBe(600)
  })

  it('anchors a timestamp-less shutdown after its rows, not at session start', async () => {
    // Audit finding: the leg's stamp anchors its interval in the residual
    // sweep. A shutdown event with no `timestamp` fell back to
    // sessionStartTime — BEFORE every row — so the leg covered nothing and
    // re-minted its whole usage beside the rows it duplicates (600/17,000
    // served as 1,200/34,000). The fallback now prefers the last event seen,
    // which is at the end of the leg where a shutdown actually happens.
    const { dbPath, at } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    const dir = join(tmpHome, 'session-state', 'sess-nots')
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, 'workspace.yaml'), 'id: sess-nots\ncwd: /home/user/testproj\n')
    await writeFile(join(dir, 'events.jsonl'), [
      JSON.stringify({ type: 'session.model_change', timestamp: at(0), data: { newModel: 'claude-sonnet-4-5' } }),
      JSON.stringify({ type: 'assistant.message', timestamp: at(18), data: { messageId: 'msg-1', outputTokens: 25, toolRequests: [] } }),
      // No `timestamp` on the shutdown; sessionStartTime precedes every row.
      JSON.stringify({
        type: 'session.shutdown',
        data: {
          shutdownType: 'routine',
          sessionStartTime: at(0),
          modelMetrics: {
            'claude-sonnet-4-5': {
              requests: { count: 2, cost: 1 },
              usage: { inputTokens: 20000, outputTokens: 25, cacheReadTokens: 17000, cacheWriteTokens: 2400, reasoningTokens: 0 },
            },
          },
        },
      }),
    ].join('\n') + '\n')
    insertStoreRow(dbPath, 'sess-nots', 12000, 10000, 1500, at(12)) // input 500
    insertStoreRow(dbPath, 'sess-nots', 8000, 7000, 900, at(15))    // input 100

    const session = (await parseAllSessions(undefined, 'copilot')).flatMap(p => p.sessions)
      .find(s => s.sessionId === 'sess-nots')!
    expect(session.totalInputTokens).toBe(600)
    expect(session.totalCacheReadTokens).toBe(17000)
    expect(session.totalCacheWriteTokens).toBe(2400)
  })

  it('keeps a crash row\'s call weight when it sits outside the pairing window', async () => {
    const { dbPath, at, writeSession } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    // The per-turn call at at(10) lost its own row; a crash-only row lands 5
    // minutes later. A wide pairing window would pair the two and hide the
    // crash request; the tight window leaves the row unpaired and counted.
    await writeSession('sess-far', { output: 25 })
    insertStoreRow(dbPath, 'sess-far', 100, 0, 0, at(310))

    const session = (await parseAllSessions(undefined, 'copilot')).flatMap(p => p.sessions)
      .find(s => s.sessionId === 'sess-far')!
    expect(session.apiCalls).toBe(2)
    expect(session.totalInputTokens).toBe(100)
    expect(session.totalOutputTokens).toBe(25)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (y) Pairing is range-invariant: adjacent day queries stay additive
// ═══════════════════════════════════════════════════════════════════════════
// Round-6.5 (gpt-5.6-sol): weights recomputed from a range slice let a store
// row and its per-turn call each count as a call on opposite sides of a day
// boundary — two calls for one request across adjacent queries. The pairing
// is computed once over the full serve set and arrives as a key set, so a
// slice cannot change a row's verdict.
describe.skipIf(!isSqliteAvailable())('(y) range-invariant behavioral pairing', () => {
  it('counts one call across two adjacent ranges that split row from per-turn call', async () => {
    const { dbPath, at, writeSession } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    // writeSession puts the per-turn call at at(10); the row lands at at(60),
    // both within the pairing window.
    await writeSession('sess-split', { output: 25 })
    insertStoreRow(dbPath, 'sess-split', 12000, 10000, 1500, at(60))

    const base = new Date(at(0)).getTime()
    const range = async (fromSec: number, toSec: number) => {
      const projects = await parseAllSessions(
        { start: new Date(base + fromSec * 1000), end: new Date(base + toSec * 1000) }, 'copilot')
      const sessions = projects.flatMap(p => p.sessions).filter(s => s.sessionId === 'sess-split')
      return {
        apiCalls: sessions.reduce((s, x) => s + x.apiCalls, 0),
        input: sessions.reduce((s, x) => s + x.totalInputTokens, 0),
        output: sessions.reduce((s, x) => s + x.totalOutputTokens, 0),
      }
    }

    // Range A holds only the per-turn call; range B only the row.
    const a = await range(0, 30)
    expect(a).toEqual({ apiCalls: 1, input: 0, output: 25 })
    const b = await range(31, 120)
    expect(b).toEqual({ apiCalls: 0, input: 500, output: 0 })

    // The full range pairs them the same way: still exactly one call.
    const full = await range(0, 120)
    expect(full).toEqual({ apiCalls: 1, input: 500, output: 25 })
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (z) The hydration verdict travels with its result
// ═══════════════════════════════════════════════════════════════════════════
describe.skipIf(!isSqliteAvailable())('(z) hydration verdict integrity', () => {
  // Round-6.5 (gpt-5.6-sol): a memoized partial parse served after a later,
  // complete parse inherited the later parse's `true`, letting the daily
  // backfill seal history around the deferred data.
  it.skipIf(process.getuid?.() === 0)('a memo hit restores the verdict its data was parsed under', async () => {
    const { chmod } = await import('fs/promises')
    const { dbPath, at, writeSession } = await setupCopilotStoreEnv()
    createStoreDb(dbPath)
    insertStoreRow(dbPath, 'sess-memo', 12000, 10000, 1500, at(12))
    await writeSession('sess-memo', { output: 25 })
    await parseAllSessions(undefined, 'copilot')
    expect(isSessionHydrationComplete()).toBe(true)

    // The store grows and becomes unreadable: the re-parse defers (false)
    // and that verdict is memoized with the result.
    clearSessionCache()
    insertStoreRow(dbPath, 'sess-memo', 8000, 7000, 900, at(20))
    await chmod(dbPath, 0o000)
    try {
      await parseAllSessions(undefined, 'copilot')
      expect(isSessionHydrationComplete()).toBe(false)

      // An unrelated provider parses completely and flips the global to true…
      await parseAllSessions(undefined, 'claude')
      expect(isSessionHydrationComplete()).toBe(true)

      // …but re-serving the deferred copilot result from the memo must
      // restore ITS verdict, not inherit the later parse's.
      await parseAllSessions(undefined, 'copilot')
      expect(isSessionHydrationComplete()).toBe(false)
    } finally {
      await chmod(dbPath, 0o644)
    }
  })

  // Round-6.5 (gpt-5.6-sol): a source whose FINGERPRINT cannot be read was
  // skipped before any parser could raise the deferral shape, leaving the
  // fence open while the cached (stale) rows served.
  it.skipIf(process.getuid?.() === 0)('an unreadable fingerprint defers instead of silently skipping', async () => {
    const { chmod } = await import('fs/promises')
    const sessionStateDir = join(tmpHome, 'session-state')
    await mkdir(sessionStateDir, { recursive: true })
    const storeDir = join(tmpHome, 'store-dir')
    await mkdir(storeDir, { recursive: true })
    const dbPath = join(storeDir, 'session-store.db')
    vi.stubEnv('CODEBURN_COPILOT_SESSION_STATE_DIR', sessionStateDir)
    vi.stubEnv('CODEBURN_COPILOT_SESSION_STORE_DB', dbPath)
    vi.stubEnv('CODEBURN_COPILOT_DISABLE_OTEL', '1')
    vi.stubEnv('CODEBURN_COPILOT_WS_STORAGE_DIR', join(tmpHome, 'no-ws'))
    vi.stubEnv('CODEBURN_COPILOT_GLOBAL_STORAGE_DIR', join(tmpHome, 'no-global'))
    vi.stubEnv('CODEBURN_COPILOT_JETBRAINS_DIR', join(tmpHome, 'no-jb'))
    const base = Date.now() - 5 * 24 * 60 * 60 * 1000
    const at = (offsetSec: number): string => new Date(base + offsetSec * 1000).toISOString()

    createStoreDb(dbPath)
    insertStoreRow(dbPath, 'sess-fp', 12000, 10000, 1500, at(12))
    const first = await parseAllSessions(undefined, 'copilot')
    expect(first.flatMap(p => p.sessions).find(s => s.sessionId === 'sess-fp')!.totalInputTokens).toBe(500)
    expect(isSessionHydrationComplete()).toBe(true)

    // The store grows, then its parent dir loses traversal: discovery still
    // emits the source (EACCES is not absence), the fingerprint read fails,
    // and the pass must report incomplete hydration — the new row is missing.
    clearSessionCache()
    insertStoreRow(dbPath, 'sess-fp', 8000, 7000, 900, at(20))
    await chmod(storeDir, 0o000)
    try {
      const second = await parseAllSessions(undefined, 'copilot')
      expect(second.flatMap(p => p.sessions).find(s => s.sessionId === 'sess-fp')!.totalInputTokens).toBe(500)
      expect(isSessionHydrationComplete()).toBe(false)
    } finally {
      await chmod(storeDir, 0o755)
    }

    clearSessionCache()
    const third = await parseAllSessions(undefined, 'copilot')
    expect(third.flatMap(p => p.sessions).find(s => s.sessionId === 'sess-fp')!.totalInputTokens).toBe(600)
    expect(isSessionHydrationComplete()).toBe(true)
  })
})

// ═══════════════════════════════════════════════════════════════════════════
// (j) Rollup-day reattribution: usage lands on the request days, not the day
//     the CLI finally shut down
// ═══════════════════════════════════════════════════════════════════════════
// Observed in the wild: a session ran entirely on day N (per-request DB rows)
// but its session.shutdown rollup was stamped the NEXT morning when the CLI
// was closed. The rollup path put the whole session's input/cache on day N+1;
// with the store covering the session, the tokens must land on day N and the
// session must contribute NOTHING to day N+1 — while still counting exactly
// once in an unfiltered (lifetime) parse. This is the per-day attribution
// change the daily-cache v18 bump re-derives for.
describe.skipIf(!isSqliteAvailable())('(j) rollup-day reattribution to request days', () => {
  it('counts a next-morning-shutdown session on its request day only', async () => {
    const sessionStateDir = join(tmpHome, 'session-state')
    await mkdir(sessionStateDir, { recursive: true })
    const dbPath = join(tmpHome, 'session-store.db')
    vi.stubEnv('CODEBURN_COPILOT_SESSION_STATE_DIR', sessionStateDir)
    vi.stubEnv('CODEBURN_COPILOT_SESSION_STORE_DB', dbPath)
    vi.stubEnv('CODEBURN_COPILOT_DISABLE_OTEL', '1')
    vi.stubEnv('CODEBURN_COPILOT_WS_STORAGE_DIR', join(tmpHome, 'no-ws'))
    vi.stubEnv('CODEBURN_COPILOT_GLOBAL_STORAGE_DIR', join(tmpHome, 'no-global'))
    vi.stubEnv('CODEBURN_COPILOT_JETBRAINS_DIR', join(tmpHome, 'no-jb'))

    // "Day N" = 5 days ago; the shutdown lands ~19h later ("next morning").
    const dayN = Date.now() - 5 * 24 * 60 * 60 * 1000
    const at = (offsetHours: number): string => new Date(dayN + offsetHours * 3600 * 1000).toISOString()

    const dir = join(sessionStateDir, 'sess-overnight')
    await mkdir(dir, { recursive: true })
    await writeFile(join(dir, 'workspace.yaml'), 'id: sess-overnight\ncwd: /home/user/testproj\n')
    await writeFile(join(dir, 'events.jsonl'), [
      JSON.stringify({ type: 'session.model_change', timestamp: at(0), data: { newModel: 'claude-sonnet-4-5' } }),
      JSON.stringify({ type: 'assistant.message', timestamp: at(1), data: { messageId: 'msg-1', outputTokens: 25, toolRequests: [] } }),
      JSON.stringify({
        type: 'session.shutdown',
        timestamp: at(19),
        data: {
          shutdownType: 'routine',
          modelMetrics: {
            'claude-sonnet-4-5': {
              requests: { count: 2, cost: 1 },
              usage: { inputTokens: 20000, outputTokens: 25, cacheReadTokens: 17000, cacheWriteTokens: 2400, reasoningTokens: 0 },
            },
          },
        },
      }),
    ].join('\n') + '\n')

    createStoreDb(dbPath)
    insertStoreRow(dbPath, 'sess-overnight', 12000, 10000, 1500, at(1)) // input 500
    insertStoreRow(dbPath, 'sess-overnight', 8000, 7000, 900, at(2))    // input 100

    const inventory = (projects: Awaited<ReturnType<typeof parseAllSessions>>) => {
      const sessions = projects.flatMap(p => p.sessions).filter(s => s.turns.some(t => t.assistantCalls.length > 0))
      const calls = sessions.flatMap(s => s.turns).flatMap(t => t.assistantCalls)
      return {
        sessions: sessions.length,
        input: calls.reduce((s, c) => s + c.usage.inputTokens, 0),
        cacheRead: calls.reduce((s, c) => s + c.usage.cacheReadInputTokens, 0),
        output: calls.reduce((s, c) => s + c.usage.outputTokens, 0),
      }
    }

    // Lifetime: exactly one session, tokens counted once, from the store.
    const lifetime = inventory(await parseAllSessions(undefined, 'copilot'))
    expect(lifetime).toEqual({ sessions: 1, input: 600, cacheRead: 17000, output: 25 })

    // A range covering only the shutdown stamp (rollup path would have put
    // 600/17000 here): the session must contribute nothing at all.
    const shutdownDay = inventory(await parseAllSessions(
      { start: new Date(dayN + 12 * 3600 * 1000), end: new Date(dayN + 36 * 3600 * 1000) }, 'copilot'))
    expect(shutdownDay).toEqual({ sessions: 0, input: 0, cacheRead: 0, output: 0 })

    // The request day carries everything.
    const requestDay = inventory(await parseAllSessions(
      { start: new Date(dayN - 1 * 3600 * 1000), end: new Date(dayN + 12 * 3600 * 1000) }, 'copilot'))
    expect(requestDay).toEqual({ sessions: 1, input: 600, cacheRead: 17000, output: 25 })
  })
})
