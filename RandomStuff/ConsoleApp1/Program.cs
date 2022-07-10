using Hedging;

HBond bond = new();
HBond hBond = new();

List<IInstrument> bondsToHedge = new() { bond };
List<IInstrument> hedgerBonds = new() { hBond };

IHedgeMethod hedger = HedgeFactory.GetHedger(HedgeType.DV01);
hedger.SetBonds(bondsToHedge);
hedger.SetHedgers(hedgerBonds);

hedger.ComputeHedgeOrders();
