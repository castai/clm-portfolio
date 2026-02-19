using Microsoft.EntityFrameworkCore;
using RiskManager.Api.Models;

namespace RiskManager.Api.Data;

public static class DbInitializer
{
    public static async Task InitializeAsync(AppDbContext context)
    {
        await context.Database.EnsureCreatedAsync();

        if (await context.Positions.AnyAsync())
            return;

        var positions = new List<Position>
        {
            new() { Ticker = "AAPL",  CompanyName = "Apple Inc.",                Sector = "Technology",      Shares = 15000, Price = 189.84m, CostBasis = 172.50m },
            new() { Ticker = "MSFT",  CompanyName = "Microsoft Corp.",           Sector = "Technology",      Shares = 12000, Price = 378.91m, CostBasis = 340.20m },
            new() { Ticker = "GOOGL", CompanyName = "Alphabet Inc.",             Sector = "Technology",      Shares = 8000,  Price = 141.80m, CostBasis = 128.40m },
            new() { Ticker = "AMZN",  CompanyName = "Amazon.com Inc.",           Sector = "Technology",      Shares = 10000, Price = 185.07m, CostBasis = 165.30m },
            new() { Ticker = "JPM",   CompanyName = "JPMorgan Chase & Co.",      Sector = "Financials",      Shares = 20000, Price = 196.52m, CostBasis = 178.90m },
            new() { Ticker = "BLK",   CompanyName = "BlackRock Inc.",            Sector = "Financials",      Shares = 5000,  Price = 814.27m, CostBasis = 750.00m },
            new() { Ticker = "GS",    CompanyName = "Goldman Sachs Group",       Sector = "Financials",      Shares = 8000,  Price = 384.65m, CostBasis = 355.80m },
            new() { Ticker = "JNJ",   CompanyName = "Johnson & Johnson",         Sector = "Healthcare",      Shares = 18000, Price = 156.74m, CostBasis = 162.30m },
            new() { Ticker = "UNH",   CompanyName = "UnitedHealth Group",        Sector = "Healthcare",      Shares = 6000,  Price = 527.38m, CostBasis = 495.10m },
            new() { Ticker = "PFE",   CompanyName = "Pfizer Inc.",               Sector = "Healthcare",      Shares = 25000, Price = 28.94m,  CostBasis = 35.20m },
            new() { Ticker = "XOM",   CompanyName = "Exxon Mobil Corp.",         Sector = "Energy",          Shares = 14000, Price = 104.57m, CostBasis = 95.80m },
            new() { Ticker = "CVX",   CompanyName = "Chevron Corp.",             Sector = "Energy",          Shares = 10000, Price = 149.23m, CostBasis = 140.50m },
            new() { Ticker = "PG",    CompanyName = "Procter & Gamble Co.",      Sector = "Consumer Staples", Shares = 12000, Price = 152.48m, CostBasis = 145.60m },
            new() { Ticker = "KO",    CompanyName = "Coca-Cola Co.",             Sector = "Consumer Staples", Shares = 20000, Price = 59.34m,  CostBasis = 55.80m },
            new() { Ticker = "DIS",   CompanyName = "Walt Disney Co.",           Sector = "Communication",   Shares = 15000, Price = 93.42m,  CostBasis = 102.70m },
            new() { Ticker = "NFLX",  CompanyName = "Netflix Inc.",              Sector = "Communication",   Shares = 4000,  Price = 486.88m, CostBasis = 420.00m },
            new() { Ticker = "V",     CompanyName = "Visa Inc.",                 Sector = "Financials",      Shares = 9000,  Price = 261.53m, CostBasis = 240.10m },
            new() { Ticker = "NVDA",  CompanyName = "NVIDIA Corp.",              Sector = "Technology",      Shares = 7000,  Price = 495.22m, CostBasis = 380.00m },
            new() { Ticker = "TSM",   CompanyName = "Taiwan Semiconductor",      Sector = "Technology",      Shares = 11000, Price = 106.38m, CostBasis = 92.40m },
            new() { Ticker = "BRK.B", CompanyName = "Berkshire Hathaway",        Sector = "Financials",      Shares = 6000,  Price = 363.82m, CostBasis = 330.50m },
        };

        context.Positions.AddRange(positions);
        await context.SaveChangesAsync();

        // Seed 30 days of P&L history
        var random = new Random(42); // deterministic seed
        var pnlHistory = new List<PnLDataPoint>();
        decimal cumulativePnL = 0;
        decimal baseValue = positions.Sum(p => p.Shares * p.CostBasis);

        for (int i = 29; i >= 0; i--)
        {
            var dailyPnL = Math.Round((decimal)(random.NextDouble() * 600000 - 200000), 2);
            cumulativePnL += dailyPnL;
            baseValue += dailyPnL;

            pnlHistory.Add(new PnLDataPoint
            {
                Date = DateTime.UtcNow.Date.AddDays(-i),
                DailyPnL = dailyPnL,
                CumulativePnL = cumulativePnL,
                PortfolioValue = baseValue
            });
        }

        context.PnLHistory.AddRange(pnlHistory);
        await context.SaveChangesAsync();
    }
}
