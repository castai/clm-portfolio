using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using RiskManager.Api.Data;
using RiskManager.Api.Models;

namespace RiskManager.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class PortfolioController : ControllerBase
{
    private readonly AppDbContext _context;

    public PortfolioController(AppDbContext context)
    {
        _context = context;
    }

    [HttpGet("positions")]
    public async Task<ActionResult<List<Position>>> GetPositions()
    {
        var positions = await _context.Positions.ToListAsync();
        return Ok(positions);
    }

    [HttpGet("summary")]
    public async Task<ActionResult<PortfolioSummary>> GetSummary()
    {
        var positions = await _context.Positions.ToListAsync();

        var totalAum = positions.Sum(p => p.MarketValue);
        var totalCost = positions.Sum(p => p.Shares * p.CostBasis);
        var totalPnL = totalAum - totalCost;

        var sectorAllocations = positions
            .GroupBy(p => p.Sector)
            .Select(g => new SectorAllocation
            {
                Sector = g.Key,
                MarketValue = g.Sum(p => p.MarketValue),
                Weight = totalAum > 0 ? Math.Round(g.Sum(p => p.MarketValue) / totalAum * 100, 2) : 0
            })
            .OrderByDescending(s => s.Weight)
            .ToList();

        var topHoldings = positions
            .OrderByDescending(p => p.MarketValue)
            .Take(5)
            .Select(p => new TopHolding
            {
                Ticker = p.Ticker,
                CompanyName = p.CompanyName,
                MarketValue = p.MarketValue,
                Weight = totalAum > 0 ? Math.Round(p.MarketValue / totalAum * 100, 2) : 0
            })
            .ToList();

        return Ok(new PortfolioSummary
        {
            TotalAum = totalAum,
            TotalPnL = totalPnL,
            PnLPercent = totalCost > 0 ? Math.Round(totalPnL / totalCost * 100, 2) : 0,
            PositionCount = positions.Count,
            SectorAllocations = sectorAllocations,
            TopHoldings = topHoldings
        });
    }
}
