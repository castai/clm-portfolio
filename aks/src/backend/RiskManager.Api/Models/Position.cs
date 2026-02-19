namespace RiskManager.Api.Models;

public class Position
{
    public int Id { get; set; }
    public string Ticker { get; set; } = string.Empty;
    public string CompanyName { get; set; } = string.Empty;
    public string Sector { get; set; } = string.Empty;
    public int Shares { get; set; }
    public decimal Price { get; set; }
    public decimal CostBasis { get; set; }
    public decimal MarketValue => Shares * Price;
    public decimal UnrealizedPnL => MarketValue - (Shares * CostBasis);
    public decimal ChangePercent => CostBasis != 0 ? Math.Round((Price - CostBasis) / CostBasis * 100, 2) : 0;
}
