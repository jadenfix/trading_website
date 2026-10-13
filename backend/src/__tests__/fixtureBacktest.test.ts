import request from 'supertest'
import { app } from '../app'

describe('provider-free fixture replay', () => {
  it('derives a deterministic receipt from checked-in rows', async () => {
    const response = await request(app)
      .get('/api/backtest/run')
      .query({ dataset: 'stocks_april', strategy: 'FIXTURE_BUY_HOLD', cash: '100000' })
      .expect(200)

    expect(response.body).toHaveLength(1)
    expect(response.body[0]).toMatchObject({
      strategy_on_dataset: 'FIXTURE_BUY_HOLD@stocks_april',
      mode: 'fictional-fixture',
      data_points: 8,
      fills: 2,
      quantity: 236,
      execution_status: 'completed',
      source: 'checked-in fictional OHLCV fixture'
    })
    expect(response.body[0].final_equity).toBeCloseTo(103607.8854, 4)
    expect(response.body[0].max_dd_pct).toBeCloseTo(0.1997032, 6)
  })

  it.each([
    [{ dataset: 'missing', strategy: 'FIXTURE_BUY_HOLD', cash: '100000' }, 400],
    [{ dataset: 'stocks_april', strategy: 'FIXTURE_BUY_HOLD', cash: 'NaN' }, 400],
    [{ dataset: 'stocks_april', strategy: 'MAC_5_20', cash: '100000' }, 503]
  ])('rejects unsupported requests before inventing output', async (query, status) => {
    const response = await request(app).get('/api/backtest/run').query(query).expect(status)
    expect(response.body.error).toBeTruthy()
  })

  it('exposes a provider-free health receipt from the same app', async () => {
    await request(app).get('/api/ping').expect(200, { pong: true, mode: 'provider-free' })
  })

  it('returns a truthful no-trade receipt when cash cannot buy one commission-adjusted unit', async () => {
    const response = await request(app)
      .get('/api/backtest/run')
      .query({ dataset: 'btc_2024', strategy: 'FIXTURE_BUY_HOLD', cash: '1000' })
      .expect(200)

    expect(response.body[0]).toMatchObject({
      mode: 'fictional-fixture',
      fills: 0,
      quantity: 0,
      execution_status: 'no-trade-insufficient-cash',
      return_pct: 0,
      max_dd_pct: 0,
      realized_pnl: 0,
      commission: 0,
      final_equity: 1000
    })
  })

  it('does not cross the commission-adjusted one-unit boundary early', async () => {
    const response = await request(app)
      .get('/api/backtest/run')
      .query({ dataset: 'btc_2024', strategy: 'FIXTURE_BUY_HOLD', cash: '42842.79' })
      .expect(200)

    expect(response.body[0]).toMatchObject({
      fills: 0,
      quantity: 0,
      commission: 0,
      realized_pnl: 0,
      final_equity: 42842.79,
      execution_status: 'no-trade-insufficient-cash'
    })
  })

  it('executes exactly one unit at the commission-adjusted boundary', async () => {
    const response = await request(app)
      .get('/api/backtest/run')
      .query({ dataset: 'btc_2024', strategy: 'FIXTURE_BUY_HOLD', cash: '42842.8' })
      .expect(200)

    expect(response.body[0]).toMatchObject({
      fills: 2,
      quantity: 1,
      realized_pnl: 59100,
      final_equity: 101798.1,
      execution_status: 'completed'
    })
    expect(response.body[0].commission).toBeCloseTo(144.7, 8)
  })
})
