namespace RiskManager.Api.Models;

public class PortfolioSummary
{
    public decimal TotalAum { get; set; }
    public decimal TotalPnL { get; set; }
    public decimal PnLPercent { get; set; }
    public int PositionCount { get; set; }
    public List<SectorAllocation> SectorAllocations { get; set; } = new();
    public List<TopHolding> TopHoldings { get; set; } = new();
}

public class SectorAllocation
{
    public string Sector { get; set; } = string.Empty;
    public decimal MarketValue { get; set; }
    public decimal Weight { get; set; }
}

public class TopHolding
{
    public string Ticker { get; set; } = string.Empty;
    public string CompanyName { get; set; } = string.Empty;
    public decimal MarketValue { get; set; }
    public decimal Weight { get; set; }
}
