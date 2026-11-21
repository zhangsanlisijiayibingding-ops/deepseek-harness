import { describe, expect, it } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import SessionStore, {
  appendPluginRecord,
  buildForkSeed,
  pluginRecordOf,
  Session,
  SessionId,
  SessionLogOffset,
  SessionSeq,
} from '@deepseek-ai/dsh-session'
import type { SessionEvent } from '@deepseek-ai/dsh-session'
import type { PluginRequestFixture } from './plugin-records.type-test.ts'

/** One user turn opening, so a record lands between model-visible events. */
function openTurn(session: Session): void {
  session.append('turn/start', { turn: 1 })
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'hello' }], source: { kind: 'user' },
  }), { surfaceOp: 'append' })
}

/** Boot one isolated SessionStore and return its context. */
async function createStore(): Promise<Context> {
  const ctx = new Context()
  await ctx.plugin(SessionStore)
  return ctx
}

describe('plugin records', () => {
  it('commits an ignorable record outside the model-visible surface', () => {
    const session = Session.create(SessionId('record'))
    openTurn(session)
    const messages = session.deriveMessages()
    const nodes = session.surface.nodes
    const data = { count: 1 }

    const seq = appendPluginRecord(session, 'plugin:test/state', data)
    data.count = 2

    expect(seq).toBe(SessionSeq(2))
    const record = session.eventAt(seq)
    expect(record).toEqual({ type: 'plugin:test/state', seq: 2, time: record?.time, data: { count: 1 }, ignorable: true })
    expect(Object.isFrozen(record)).toBe(true)
    expect(Object.isFrozen(record?.data)).toBe(true)
    expect(session.deriveMessages()).toEqual(messages)
    expect(session.surface.nodes).toEqual(nodes)
  })

  it('reads records back and leaves every other event unread', () => {
    const session = Session.create(SessionId('reader'))
    openTurn(session)
    appendPluginRecord(session, 'plugin:test.bridge/entry_1', ['a', 2])

    const events = session.snapshotEvents()
    expect(events.map(pluginRecordOf)).toEqual([
      undefined,
      undefined,
      { type: 'plugin:test.bridge/entry_1', seq: 2, time: events[2]?.time, data: ['a', 2] },
    ])
  })

  it('snapshots an interface-typed request with readonly messages', () => {
    const session = Session.create(SessionId('typed-request'))
    const request: PluginRequestFixture = {
      config: { provider: 'fixture', model: 'fixture' },
      messages: [{ role: 'user', content: [{ type: 'text', text: 'Translate this fragment.' }] }],
    }
    const seq = appendPluginRecord(session, 'plugin:test/request', request)
    request.config.model = 'changed'

    const event = session.eventAt(seq)
    expect(event?.data).toEqual({
      config: { provider: 'fixture', model: 'fixture' },
      messages: [{ role: 'user', content: [{ type: 'text', text: 'Translate this fragment.' }] }],
    })
    expect(Object.isFrozen(event?.data)).toBe(true)
    expect(session.deriveMessages()).toEqual([])
  })

  it('does not read a known event, even one marked ignorable', () => {
    const events: SessionEvent[] = [
      { type: 'turn/start', seq: SessionSeq(0), time: 1, data: { turn: 1 } },
      { type: 'turn/start', seq: SessionSeq(1), time: 2, data: { turn: 2 }, ignorable: true },
    ]

    expect(events.map(pluginRecordOf)).toEqual([undefined, undefined])
  })

  it.each([
    'plugin:',
    'plugin:Test/state',
    'plugin:/state',
    'plugin:test/',
    'plugin:test//state',
    'plugin:-test',
    'plugin:test state',
    'turn/start',
  ])('rejects the record type %j without changing the log', (type) => {
    const session = Session.create(SessionId('grammar'))
    const name: string = type

    expect(() => {
      // @ts-expect-error -- runtime callers can supply names outside the declared map.
      appendPluginRecord(session, name, {})
    })
      .toThrow(`plugin record type "${type}" must be "plugin:" followed by lowercase slash-separated segments`)
    expect(session.seq).toBe(0)
  })

  it('rejects data that is not losslessly JSON-serializable without changing the log', () => {
    const session = Session.create(SessionId('lossy'))

    expect(() => appendPluginRecord(session, 'plugin:test/number', Number.NaN))
      .toThrow('plugin record "plugin:test/number" carries non-JSON-serializable data')
    expect(session.seq).toBe(0)
  })

  it('publishes a record to session/event observers and contains a reentrant record', async () => {
    const ctx = new Context()
    await ctx.plugin(SessionStore)
    const warnings: string[] = []
    ctx.logger.warn = ((message: unknown) => { warnings.push(String(message)) }) as typeof ctx.logger.warn
    const session = ctx.sessions.create(SessionId('observed-record'))
    const heard: SessionEvent[] = []
    ctx.on('session/event', (observed) => { appendPluginRecord(observed, 'plugin:test/echo', {}) })
    ctx.on('session/event', (_observed, event) => { heard.push(event) })

    appendPluginRecord(session, 'plugin:test/state', { count: 1 })

    expect(heard.map(event => event.type)).toEqual(['plugin:test/state'])
    expect(session.seq).toBe(1)
    expect(warnings).toEqual([
      'session "observed-record": session/event listener threw: Error: session append cannot reenter while another append is being published',
    ])
    await ctx.fiber.dispose()
  })

  it('carries records through a snapshot restore, a storage reopen, and a fork', () => {
    const session = Session.create(SessionId('source'))
    openTurn(session)
    appendPluginRecord(session, 'plugin:test/state', { count: 1 })
    const events = session.snapshotEvents()

    const restored = Session.create(SessionId('restored'), events)
    const reopened = Session.fromRestore(SessionId('source'), events, session.header, SessionLogOffset(0), 'shared-frozen')
    const child = Session.create(SessionId('child'), buildForkSeed(events, SessionSeq(2)))

    for (const copy of [restored, reopened, child]) {
      expect(pluginRecordOf(copy.snapshotEvents()[2] as SessionEvent))
        .toEqual({ type: 'plugin:test/state', seq: 2, time: events[2]?.time, data: { count: 1 } })
      expect(copy.deriveMessages()).toEqual(session.deriveMessages())
    }
  })
})

describe('SessionStore plugin record API', () => {
  it('appends to a live store Session and snapshots the payload', async () => {
    const ctx = await createStore()
    try {
      const session = ctx.sessions.create(SessionId('store-record'))
      const data = { count: 1 }
      const seq = ctx.sessions.appendPluginRecord(session, 'plugin:test/state', data)
      data.count = 2

      expect(seq).toBe(SessionSeq(0))
      expect(session.seq).toBe(1)
      const event = session.eventAt(seq)
      if (event === undefined) throw new Error('record was not committed')
      expect(event).toEqual({
        type: 'plugin:test/state', seq: SessionSeq(0), time: event.time, data: { count: 1 }, ignorable: true,
      })
      expect(pluginRecordOf(event)).toEqual({
        type: 'plugin:test/state', seq: SessionSeq(0), time: event.time, data: { count: 1 },
      })
      expect(session.deriveMessages()).toEqual([])
    } finally {
      await ctx.fiber.dispose()
    }
  })

  it('rejects a Session owned by another store without changing either log', async () => {
    const first = await createStore()
    const second = await createStore()
    try {
      const firstSession = first.sessions.create(SessionId('first-store'))
      const secondSession = second.sessions.create(SessionId('second-store'))

      expect(() => first.sessions.appendPluginRecord(secondSession, 'plugin:test/state', { count: 1 }))
        .toThrow('is not live in this store')
      expect(firstSession.seq).toBe(0)
      expect(secondSession.seq).toBe(0)
    } finally {
      await first.fiber.dispose()
      await second.fiber.dispose()
    }
  })

  it('rejects a cold Session and a detached Session', async () => {
    const ctx = await createStore()
    try {
      const cold = Session.create(SessionId('cold-store'))
      expect(() => ctx.sessions.appendPluginRecord(cold, 'plugin:test/state', { count: 1 }))
        .toThrow('is not live in this store')
      expect(cold.seq).toBe(0)

      const detached = ctx.sessions.prepare(SessionId('detached-store'))
      const detach = ctx.sessions.enter(detached)
      ctx.sessions.announce(detached)
      ctx.sessions.appendPluginRecord(detached, 'plugin:test/state', { count: 1 })
      detach()

      expect(() => ctx.sessions.appendPluginRecord(detached, 'plugin:test/state', { count: 2 }))
        .toThrow('is not live in this store')
      expect(detached.seq).toBe(1)
    } finally {
      await ctx.fiber.dispose()
    }
  })

  it('rejects a Session whose announcement has not completed', async () => {
    const ctx = await createStore()
    try {
      const pending = ctx.sessions.prepare(SessionId('pending-announcement'))
      ctx.sessions.enter(pending)

      expect(() => ctx.sessions.appendPluginRecord(pending, 'plugin:test/state', { count: 1 }))
        .toThrow('cannot append before session/created completes')
      expect(pending.seq).toBe(0)
    } finally {
      await ctx.fiber.dispose()
    }
  })

  it.each([
    'plugin:',
    'plugin:Test/state',
    'plugin:/state',
    'plugin:test/',
    'plugin:test//state',
    'plugin:-test',
    'plugin:test state',
    'turn/start',
  ])('rejects the record type %j without changing the log', async (type) => {
    const ctx = await createStore()
    try {
      const session = ctx.sessions.create(SessionId(`bad-type-${Buffer.from(type).toString('hex')}`))
      const name: string = type

      expect(() => {
        // @ts-expect-error -- runtime callers can supply names outside the declared map.
        ctx.sessions.appendPluginRecord(session, name, {})
      }).toThrow(`plugin record type "${type}" must be "plugin:" followed by lowercase slash-separated segments`)
      expect(session.seq).toBe(0)
    } finally {
      await ctx.fiber.dispose()
    }
  })

  it('rejects non-lossless payloads without changing the log', async () => {
    const ctx = await createStore()
    try {
      const session = ctx.sessions.create(SessionId('store-lossy'))
      const cycle: { self?: unknown } = {}
      cycle.self = cycle

      expect(() => ctx.sessions.appendPluginRecord(session, 'plugin:test/number', Number.NaN))
        .toThrow('carries non-JSON-serializable data')
      // @ts-expect-error -- runtime callers can supply values outside the declared payload type.
      expect(() => ctx.sessions.appendPluginRecord(session, 'plugin:test/number', 1n))
        .toThrow('carries non-JSON-serializable data')
      // @ts-expect-error -- runtime callers can supply values outside the declared payload type.
      expect(() => ctx.sessions.appendPluginRecord(session, 'plugin:test/number', undefined))
        .toThrow('carries non-JSON-serializable data')
      // @ts-expect-error -- runtime callers can supply values outside the declared payload type.
      expect(() => ctx.sessions.appendPluginRecord(session, 'plugin:test/number', cycle))
        .toThrow('carries non-JSON-serializable data')
      expect(session.seq).toBe(0)
    } finally {
      await ctx.fiber.dispose()
    }
  })

  it('retains the append reentrancy guard for the store method', async () => {
    const ctx = await createStore()
    try {
      const session = ctx.sessions.create(SessionId('store-reentrant'))
      const warnings: string[] = []
      ctx.logger.warn = ((message: unknown) => { warnings.push(String(message)) }) as typeof ctx.logger.warn
      ctx.on('session/event', (observed) => {
        ctx.sessions.appendPluginRecord(observed, 'plugin:test/echo', {})
      })

      ctx.sessions.appendPluginRecord(session, 'plugin:test/state', { count: 1 })

      expect(session.seq).toBe(1)
      expect(warnings).toHaveLength(1)
      expect(warnings[0]).toContain('session append cannot reenter while another append is being published')
    } finally {
      await ctx.fiber.dispose()
    }
  })

  it('commits sequential calls in order', async () => {
    const ctx = await createStore()
    try {
      const session = ctx.sessions.create(SessionId('store-sequence'))
      expect(ctx.sessions.appendPluginRecord(session, 'plugin:test/state', { count: 1 })).toBe(SessionSeq(0))
      expect(ctx.sessions.appendPluginRecord(session, 'plugin:test/state', { count: 2 })).toBe(SessionSeq(1))
      expect(session.seq).toBe(2)
    } finally {
      await ctx.fiber.dispose()
    }
  })
})
