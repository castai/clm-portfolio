using Microsoft.EntityFrameworkCore;
using RiskManager.Api.Data;
using RiskManager.Api.Models;

namespace RiskManager.Api.Services;

/// <summary>
/// Background service that simulates live market data every 2 seconds.
/// 
/// For each tick:
///   1. Updates every position's price using Geometric Brownian Motion (GBM)
///      - Each stock has its own drift (mu) and volatility (sigma) based on sector
///      - Prices move realistically: tech is more volatile, staples are calmer
///   2. Computes new daily P&L from the price changes
///   3. Appends a P&L data point (or updates today's entry)
///   4. Keeps P&L history to a rolling 30-day window
///
/// Resilience:
///   - Each tick has a 10s timeout so a slow DB doesn't stall the loop
///   - Consecutive failures trigger exponential backoff (up to 30s)
///   - Successful tick resets the backoff immediately
/// </summary>
public class MarketSimulator : BackgroundService
{
    private readonly IServiceScopeFactory _scopeFactory;
    private readonly ILogger<MarketSimulator> _logger;
    private readonly Random _random = new();

    private static readonly TimeSpan TickInterval = TimeSpan.FromSeconds(2);
    private static readonly TimeSpan TickTimeout = TimeSpan.FromSeconds(10);
    private static readonly TimeSpan MaxBackoff = TimeSpan.FromSeconds(30);

    private int _consecutiveFailures = 0;

    // Sector-based volatility profiles
    // Higher sigma = more volatile price swings
    private static readonly Dictionary<string, (double drift, double volatility)> SectorProfiles = new()
    {
        ["Technology"]       = (0.0002,  0.0035),  // high growth, high vol
        ["Financials"]       = (0.00015, 0.0025),  // moderate
        ["Healthcare"]       = (0.0001,  0.0020),  // defensive, lower vol
        ["Energy"]           = (0.00012, 0.0030),  // commodity-driven, volatile
        ["Consumer Staples"] = (0.00008, 0.0015),  // boring and stable
        ["Communication"]    = (0.00015, 0.0028),  // mixed bag
    };

    public MarketSimulator(IServiceScopeFactory scopeFactory, ILogger<MarketSimulator> logger)
    {
        _scopeFactory = scopeFactory;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("MarketSimulator started. Ticking every {Interval}s", TickInterval.TotalSeconds);

        // Wait a bit for DB init to complete
        await Task.Delay(TimeSpan.FromSeconds(10), stoppingToken);

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                // Per-tick timeout so a hung DB connection doesn't block the loop
                using var cts = CancellationTokenSource.CreateLinkedTokenSource(stoppingToken);
                cts.CancelAfter(TickTimeout);

                await SimulateTick(cts.Token);

                // Success - reset backoff
                _consecutiveFailures = 0;
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
            {
                break; // app shutting down
            }
            catch (OperationCanceledException)
            {
                _consecutiveFailures++;
                _logger.LogWarning("MarketSimulator tick timed out after {Timeout}s (failure #{Count})",
                    TickTimeout.TotalSeconds, _consecutiveFailures);
            }
            catch (Exception ex)
            {
                _consecutiveFailures++;
                _logger.LogError(ex, "MarketSimulator tick failed (failure #{Count})", _consecutiveFailures);
            }

            // Exponential backoff on failures: 2s, 4s, 8s, 16s, capped at 30s
            var delay = _consecutiveFailures > 0
                ? TimeSpan.FromMilliseconds(Math.Min(
                    TickInterval.TotalMilliseconds * Math.Pow(2, _consecutiveFailures - 1),
                    MaxBackoff.TotalMilliseconds))
                : TickInterval;

            await Task.Delay(delay, stoppingToken);
        }

        _logger.LogInformation("MarketSimulator stopped.");
    }

    private async Task SimulateTick(CancellationToken ct)
    {
        using var scope = _scopeFactory.CreateScope();
        var context = scope.ServiceProvider.GetRequiredService<AppDbContext>();

        // 1. Load all positions
        var positions = await context.Positions.ToListAsync(ct);
        if (positions.Count == 0) return;

        decimal totalPnLThisTick = 0;

        // 2. Update each position's price using GBM
        foreach (var position in positions)
        {
            var oldPrice = position.Price;
            var newPrice = SimulatePrice(position.Price, position.Sector);

            // Clamp: price can't go below $1 (no penny stocks in this portfolio)
            newPrice = Math.Max(newPrice, 1.0m);

            position.Price = newPrice;

            // Track P&L from this tick
            var pnlChange = (newPrice - oldPrice) * position.Shares;
            totalPnLThisTick += pnlChange;
        }

        // 3. Update or insert today's P&L data point
        var today = DateTime.UtcNow.Date;
        var todayPnL = await context.PnLHistory
            .FirstOrDefaultAsync(p => p.Date == today, ct);

        // Get the last cumulative P&L to build on
        var lastCumulative = await context.PnLHistory
            .Where(p => p.Date < today)
            .OrderByDescending(p => p.Date)
            .Select(p => p.CumulativePnL)
            .FirstOrDefaultAsync(ct);

        var totalAum = positions.Sum(p => p.Shares * p.Price);

        if (todayPnL == null)
        {
            // First tick of the day
            todayPnL = new PnLDataPoint
            {
                Date = today,
                DailyPnL = Math.Round(totalPnLThisTick, 2),
                CumulativePnL = Math.Round(lastCumulative + totalPnLThisTick, 2),
                PortfolioValue = Math.Round(totalAum, 2)
            };
            context.PnLHistory.Add(todayPnL);
        }
        else
        {
            // Accumulate today's ticks
            todayPnL.DailyPnL = Math.Round(todayPnL.DailyPnL + totalPnLThisTick, 2);
            todayPnL.CumulativePnL = Math.Round(lastCumulative + todayPnL.DailyPnL, 2);
            todayPnL.PortfolioValue = Math.Round(totalAum, 2);
        }

        // 4. Prune: keep only last 30 days
        var cutoff = today.AddDays(-30);
        var staleEntries = await context.PnLHistory
            .Where(p => p.Date < cutoff)
            .ToListAsync(ct);

        if (staleEntries.Count > 0)
        {
            context.PnLHistory.RemoveRange(staleEntries);
        }

        // 5. Save everything
        await context.SaveChangesAsync(ct);

        _logger.LogDebug(
            "Tick: AUM={Aum:C0} DailyPnL={DailyPnl:+#,##0;-#,##0} ({Count} positions updated)",
            totalAum, totalPnLThisTick, positions.Count);
    }

    /// <summary>
    /// Geometric Brownian Motion price simulation.
    /// 
    /// dS = S * (mu * dt + sigma * sqrt(dt) * Z)
    /// 
    /// where Z ~ N(0,1) (standard normal via Box-Muller transform)
    /// </summary>
    private decimal SimulatePrice(decimal currentPrice, string sector)
    {
        var (drift, volatility) = SectorProfiles.GetValueOrDefault(sector, (0.0001, 0.0025));

        // Box-Muller transform for standard normal
        var u1 = 1.0 - _random.NextDouble(); // avoid log(0)
        var u2 = _random.NextDouble();
        var z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);

        // GBM step
        var dt = 1.0;
        var change = drift * dt + volatility * Math.Sqrt(dt) * z;
        var newPrice = (double)currentPrice * Math.Exp(change);

        return Math.Round((decimal)newPrice, 4);
    }
}
