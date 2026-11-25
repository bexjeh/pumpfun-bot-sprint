// Filtered Momentum Bot — Pump.fun Bot Sprint (pumpfun-replay/v2).
//
// Strategy (my own design): enter on a confirmed momentum breakout, but
// only on liquid coins with volume-backed demand. A coin becomes a
// candidate when its price has climbed at least 15% over its last 10
// observations (inside 7,200 slots), at least 60% of its last 20 trades
// are buys, buy SOL volume is at least twice sell SOL volume over those
// trades, and it holds at least 20 SOL of real reserves (thin coins are
// where slippage and rugs eat the edge). Entries are a fixed 0.4 SOL
// stake, one entry per mint per run. Exits: take-profit at +35%, a
// trailing stop 12% below the peak once the peak is +15% over entry, a
// hard stop-loss at -15%, and a maximum hold of 9,000 slots (~1 hour).
// Slippage is 5% on both sides: on the dev window, rejected sells (pass 1)
// accidentally outperformed forced exits (passes 2–3) by holding through
// sharp dips instead of locking in the bottom.
//
// Fully deterministic: every decision comes from the replay data and fixed
// constants only. No wall clock, no environment reads, no Math.random — the
// two graded executions produce byte-identical action logs.
//
// Revision history (measured on the kit dev window with grade-local):
// pass 1 (enter after +15%, +35% TP, -15% SL, 500 bps sells, re-entry
// allowed): net PnL -2.147 SOL, 282 round trips, 242 coins, win rate
// 36.5%; fees 1.22 SOL. Best so far.
// pass 2 (enter after +20%, +60% TP, -12% SL, 10,000 bps sells, no
// re-entry): -2.615 SOL, win rate 24.8% — entering later was worse.
// pass 3 (early entry on fresh coins, 0.2 SOL stake, +50% TP, -10% SL,
// 30-min hold): -2.651 SOL, 391 trips, win rate 24.6% — early entry
// churned through too many duds. Pass 4 returns to the pass-1 shape
// (best win rate) minus re-entries, plus liquidity and volume filters.

import {
  LAMPORTS_PER_SOL,
  runBot,
  type Batch,
  type BotState,
  type BuyInput,
  type OrderInput,
  type SellInput,
} from './vendor/makefun-bot.ts';

// ---- strategy parameters (all fixed constants) ----
const STAKE_LAMPORTS = (LAMPORTS_PER_SOL * 4n) / 10n; // 0.4 SOL per entry
const MIN_BUY_LAMPORTS = LAMPORTS_PER_SOL / 20n; // 0.05 SOL floor (fees eat smaller trades)
const MIN_REAL_RESERVES_LAMPORTS = LAMPORTS_PER_SOL * 20n; // 20 SOL liquidity floor
const ENTRY_RISE = 1.15; // +15% over the lookback to enter
const LOOKBACK_SAMPLES = 10; // observations spanned by the entry rise
const LOOKBACK_MAX_SLOTS = 7_200; // the rise must happen inside ~48 min
const MIN_BUY_SHARE = 0.6; // >= 60% buys among the last 20 trades
const MIN_BUY_VOLUME_RATIO = 2; // buy SOL volume >= 2x sell SOL volume
const TRADE_WINDOW = 20; // trades spanned by the pressure checks
const TAKE_PROFIT = 1.35; // sell all at +35%
const TRAIL_ARM = 1.15; // trailing stop arms once peak is +15% over entry
const TRAIL_DROP = 0.88; // sell when price falls to 88% of the peak
const STOP_LOSS = 0.85; // sell when price falls to 85% of entry
const MAX_HOLD_SLOTS = 9_000; // ~1 hour maximum hold
const MAX_SLIPPAGE_BPS = 500; // 5% slippage on buys and sells
const MAX_SAMPLES = 24; // price observations kept per mint
const PRUNE_AFTER_SLOTS = 21_600; // drop idle mints after ~2.4 h

interface PriceSample {
  slot: number;
  price: number;
}

interface EntryRecord {
  entryPrice: number;
  peak: number;
  entrySlot: number;
}

/** One observed trade: direction plus its SOL size (lamports). */
interface TradeTick {
  isBuy: boolean;
  sol: bigint;
}

/** Price of one token in SOL, from virtual reserves (SOL-quoted curves). */
function curvePrice(virtualQuote: bigint, virtualToken: bigint): number {
  if (virtualToken <= 0n) return 0;
  return Number(virtualQuote) / Number(virtualToken);
}

// ---- per-mint state ----
const samples = new Map<string, PriceSample[]>();
const recentTrades = new Map<string, TradeTick[]>(); // last TRADE_WINDOW trades
const entries = new Map<string, EntryRecord>();
const tradedMints = new Set<string>(); // every mint ever bought: never re-entered
let firstSlot = -1;

function recordSample(mint: string, slot: number, price: number): void {
  if (!(price > 0)) return;
  let list = samples.get(mint);
  if (list === undefined) {
    list = [];
    samples.set(mint, list);
  }
  const last = list[list.length - 1];
  if (last !== undefined && last.slot === slot) {
    last.price = price; // one sample per slot: refresh with the latest reserves
    return;
  }
  list.push({ slot, price });
  if (list.length > MAX_SAMPLES) list.splice(0, list.length - MAX_SAMPLES);
}

function recordTrade(mint: string, isBuy: boolean, sol: bigint): void {
  let list = recentTrades.get(mint);
  if (list === undefined) {
    list = [];
    recentTrades.set(mint, list);
  }
  list.push({ isBuy, sol });
  if (list.length > TRADE_WINDOW) list.splice(0, list.length - TRADE_WINDOW);
}

/** Buy share and buy/sell SOL volume ratio over the last TRADE_WINDOW trades. */
function pressure(mint: string): { share: number; volumeRatio: number; count: number } {
  const list = recentTrades.get(mint);
  if (list === undefined || list.length === 0) return { share: 0, volumeRatio: 0, count: 0 };
  let buys = 0;
  let buyVol = 0n;
  let sellVol = 0n;
  for (const t of list) {
    if (t.isBuy) {
      buys += 1;
      buyVol += t.sol;
    } else {
      sellVol += t.sol;
    }
  }
  const ratio = sellVol > 0n ? Number(buyVol) / Number(sellVol) : buyVol > 0n ? 99 : 0;
  return { share: buys / list.length, volumeRatio: ratio, count: list.length };
}

/** True when the mint's price has risen enough over the lookback window. */
function hasMomentum(mint: string): boolean {
  const list = samples.get(mint);
  if (list === undefined || list.length < LOOKBACK_SAMPLES) return false;
  const now = list[list.length - 1];
  const base = list[list.length - LOOKBACK_SAMPLES];
  if (now === undefined || base === undefined || base.price <= 0) return false;
  if (now.slot - base.slot > LOOKBACK_MAX_SLOTS) return false;
  return now.price / base.price >= ENTRY_RISE;
}

function pruneStale(slot: number): void {
  for (const [mint, list] of samples) {
    const last = list[list.length - 1];
    if (last !== undefined && slot - last.slot > PRUNE_AFTER_SLOTS && !entries.has(mint)) {
      samples.delete(mint);
      recentTrades.delete(mint);
    }
  }
}

await runBot({
  onBatch(state: BotState, batch: Batch): OrderInput[] {
    const { caps } = state.hello;
    const slot = batch.slot;
    if (firstSlot < 0) firstSlot = slot;
    const orders: OrderInput[] = [];
    const tx = state.txFeeLamports();

    // Fold this batch's fills into the entry records first: a filled buy
    // opens a record at its fill price; a filled sell closes it.
    for (const fill of batch.fills) {
      if (fill.status !== 'filled') continue;
      if (fill.side === 'buy') {
        const tokens = BigInt(fill.token_amount);
        const quote = BigInt(fill.quote_lamports);
        const price = tokens > 0n ? Number(quote) / Number(tokens) : 0;
        if (price > 0) {
          entries.set(fill.mint, { entryPrice: price, peak: price, entrySlot: slot });
          tradedMints.add(fill.mint);
        }
      } else {
        // This bot only ever sells whole positions.
        entries.delete(fill.mint);
      }
    }
    // Drop records for positions the harness closed on its own (a curve
    // completing, or the replay flattening at the end).
    for (const mint of [...entries.keys()]) {
      if (!state.positions.has(mint)) entries.delete(mint);
    }

    // Observe this batch: one price sample per mint that moved, plus each
    // trade's direction and SOL size for the pressure checks.
    const seen = new Set<string>();
    for (const e of batch.events) {
      if (e.event === 'TradeEvent') {
        recordTrade(e.mint, e.is_buy, BigInt(e.sol_amount));
        seen.add(e.mint);
      } else if (e.event === 'CreateEvent') {
        seen.add(e.mint);
      }
    }
    for (const m of batch.marks) seen.add(m.mint);
    for (const mint of seen) {
      const curve = state.curves.get(mint);
      if (curve !== undefined && !curve.complete) {
        recordSample(mint, slot, curvePrice(curve.virtual_quote_reserves, curve.virtual_token_reserves));
      }
    }
    if (slot % 1024 === 0) pruneStale(slot);

    // ---- exits first: free slots before considering new entries ----
    for (const [mint, holding] of state.positions) {
      const rec = entries.get(mint);
      if (rec === undefined) continue; // no record (harness close pending): leave it
      const curve = state.curves.get(mint);
      if (curve === undefined || curve.complete) continue;
      const price = curvePrice(curve.virtual_quote_reserves, curve.virtual_token_reserves);
      if (!(price > 0)) continue;
      if (price > rec.peak) rec.peak = price;
      const heldSlots = slot - rec.entrySlot;
      const takeProfit = price >= rec.entryPrice * TAKE_PROFIT;
      const trailing =
        rec.peak >= rec.entryPrice * TRAIL_ARM && price <= rec.peak * TRAIL_DROP;
      const stopped = price <= rec.entryPrice * STOP_LOSS;
      const expired = heldSlots >= MAX_HOLD_SLOTS;
      if (takeProfit || trailing || stopped || expired) {
        const sell: SellInput = {
          side: 'sell',
          mint,
          token_amount: holding.token_amount,
          max_slippage_bps: MAX_SLIPPAGE_BPS,
        };
        orders.push(sell);
      }
    }

    // ---- entries: liquid momentum breakouts with volume-backed demand ----
    let free = state.freeCashLamports();
    let slotsOpen = caps.max_open_positions - state.committedPositionCount();
    const pendingBuys = new Set<string>();
    for (const p of state.pending.values()) {
      if (p.order.side === 'buy') pendingBuys.add(p.order.mint);
    }
    for (const e of batch.events) {
      if (orders.length >= caps.max_orders_per_batch) break;
      if (e.event !== 'TradeEvent' && e.event !== 'CreateEvent') continue;
      const mint = e.mint;
      if (state.positions.has(mint) || pendingBuys.has(mint)) continue;
      if (entries.has(mint)) continue;
      if (tradedMints.has(mint)) continue; // one entry per mint per run
      const curve = state.curves.get(mint);
      if (curve === undefined || curve.complete) continue;
      if (curve.real_quote_reserves < MIN_REAL_RESERVES_LAMPORTS) continue;
      if (!hasMomentum(mint)) continue;
      const p = pressure(mint);
      if (p.count < TRADE_WINDOW) continue;
      if (p.share < MIN_BUY_SHARE) continue;
      if (p.volumeRatio < MIN_BUY_VOLUME_RATIO) continue;
      if (slotsOpen <= 0) continue;
      const cap = state.maxBuyLamports(mint);
      if (cap < MIN_BUY_LAMPORTS) continue;
      const amount = cap < STAKE_LAMPORTS ? cap : STAKE_LAMPORTS;
      if (free < amount + tx) continue;
      const buy: BuyInput = {
        side: 'buy',
        mint,
        quote_lamports: amount,
        max_slippage_bps: MAX_SLIPPAGE_BPS,
      };
      orders.push(buy);
      pendingBuys.add(mint);
      free -= amount + tx;
      slotsOpen -= 1;
    }
    return orders;
  },

  onDone(state: BotState): void {
    console.error(
      `done: cash=${state.cash} lamports, positions=${state.positions.size}, ` +
        `distinct traded=${tradedMints.size}`,
    );
  },
});
