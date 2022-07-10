using QuantlibBond;
using Hedging;

// Setup Date
DateTime refDate = new DateTime(2022, 7, 8);
QlBond.SetEvaluationDate(refDate);

// Setup First Bond
var bond = new QlBond(new DateTime(2031, 4, 1), new DateTime(2020, 10, 1), 0.9 / 100.0);
var hBond = new HQlBondWrapper(bond);
hBond.Notional = 1000;
hBond.Price = 83.11;

// Setup Second Bond:
var otherBond = new QlBond(new DateTime(2049, 9, 1), new DateTime(2018, 9, 1), 3.85 / 100.0);
var hedger = new HQlBondWrapper(otherBond);
hedger.Price = 102.79;

// Test the hedging function
IHedgeMethod hedgingMethod = HedgeFactory.GetHedger(HedgeType.DV01);
hedgingMethod.SetBonds(new() { hBond });
hedgingMethod.SetHedgers(new() { hedger });
var orders = hedgingMethod.ComputeHedgeOrders();

double finalRisk = hBond.DV01() * 1000 + hedger.DV01() * orders.First().Value;
Console.WriteLine($"Finished, final risk: {finalRisk}");

// Setup the second bond as a Future with a base bond and a conversion factor
var bondFutureCTD = new QlBond(new DateTime(2049, 9, 1), new DateTime(2018, 9, 1), 3.85 / 100.0);
var bondWrapper = new HQlBondWrapper(bondFutureCTD);
var bondFuture = new HFuture(bondWrapper, 0.807050);
bondFuture.Price = 102.79;

// Test the hedging function for futures
IHedgeMethod hedgingMethod2 = HedgeFactory.GetHedger(HedgeType.DV01);
hedgingMethod2.SetBonds(new() { hBond });
hedgingMethod2.SetHedgers(new() { bondFuture });
var orders2 = hedgingMethod2.ComputeHedgeOrders();

double finalRisk2 = hBond.DV01() * 1000 + bondFuture.DV01() * orders2.First().Value;
Console.WriteLine($"Finished, final risk: {finalRisk2}");