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
      source: 'checked-in fictional OHLCV fixture'
    })
    expect(response.body[0].final_equity).toBeCloseTo(103607.8854, 4)
    expect(response.body[0].max_dd_pct).toBe(0)
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

  it('keeps the fixture deterministic when cash cannot buy one unit', async () => {
    const response = await request(app)
      .get('/api/backtest/run')
      .query({ dataset: 'btc_2024', strategy: 'FIXTURE_BUY_HOLD', cash: '1000' })
      .expect(200)

    expect(response.body[0]).toMatchObject({
      mode: 'fictional-fixture',
      fills: 2,
      realized_pnl: 0,
      commission: 0,
      final_equity: 1000
    })
  })
})
