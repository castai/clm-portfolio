using Microsoft.EntityFrameworkCore;
using RiskManager.Api.Models;

namespace RiskManager.Api.Data;

public class AppDbContext : DbContext
{
    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

    public DbSet<Position> Positions => Set<Position>();
    public DbSet<PnLDataPoint> PnLHistory => Set<PnLDataPoint>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<Position>(entity =>
        {
            entity.ToTable("Positions");
            entity.HasKey(e => e.Id);
            entity.Property(e => e.Ticker).HasMaxLength(10).IsRequired();
            entity.Property(e => e.CompanyName).HasMaxLength(100).IsRequired();
            entity.Property(e => e.Sector).HasMaxLength(50).IsRequired();
            entity.Property(e => e.Price).HasColumnType("decimal(18,4)");
            entity.Property(e => e.CostBasis).HasColumnType("decimal(18,4)");
            entity.Ignore(e => e.MarketValue);
            entity.Ignore(e => e.UnrealizedPnL);
            entity.Ignore(e => e.ChangePercent);
        });

        modelBuilder.Entity<PnLDataPoint>(entity =>
        {
            entity.ToTable("PnLHistory");
            entity.HasKey(e => e.Id);
            entity.Property(e => e.DailyPnL).HasColumnType("decimal(18,2)");
            entity.Property(e => e.CumulativePnL).HasColumnType("decimal(18,2)");
            entity.Property(e => e.PortfolioValue).HasColumnType("decimal(18,2)");
        });
    }
}
