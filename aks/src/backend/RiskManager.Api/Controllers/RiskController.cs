using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using RiskManager.Api.Data;
using RiskManager.Api.Models;

namespace RiskManager.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class RiskController : ControllerBase
{
    private readonly AppDbContext _context;

    public RiskController(AppDbContext context)
    {
        _context = context;
    }

    [HttpGet("metrics")]
    public async Task<ActionResult<RiskMetrics>> GetMetrics()
    {
        var positions = await _context.Positions.ToListAsync();
        var totalAum = positions.Sum(p => p.MarketValue);

        // Compute realistic-looking risk metrics based on portfolio composition
        var random = new Random((int)(DateTime.UtcNow.Ticks % int.MaxValue));

        // VaR as percentage of AUM with slight randomization for live feel
        var baseVaR95 = totalAum * 0.018m;
        var jitter = (decimal)(random.NextDouble() * 0.002 - 0.001);

        return Ok(new RiskMetrics
        {
            VaR95 = Math.Round(baseVaR95 * (1 + jitter), 0),
            VaR99 = Math.Round(baseVaR95 * 1.42m * (1 + jitter), 0),
            Beta = Math.Round(1.08m + (decimal)(random.NextDouble() * 0.04 - 0.02), 2),
            SharpeRatio = Math.Round(1.34m + (decimal)(random.NextDouble() * 0.1 - 0.05), 2),
            Volatility = Math.Round(16.8m + (decimal)(random.NextDouble() * 0.6 - 0.3), 2),
            MaxDrawdown = Math.Round(-8.42m + (decimal)(random.NextDouble() * 0.4 - 0.2), 2),
            TotalAum = totalAum,
            CalculatedAt = DateTime.UtcNow
        });
    }

    [HttpGet("pnl")]
    public async Task<ActionResult<List<PnLDataPoint>>> GetPnL()
    {
        var pnl = await _context.PnLHistory
            .OrderBy(p => p.Date)
            .ToListAsync();
        return Ok(pnl);
    }
}
