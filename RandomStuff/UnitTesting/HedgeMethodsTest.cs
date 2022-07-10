namespace UnitTesting
{
    [TestClass]
    public class HedgeMethodsTest
    {
        [TestMethod]
        public void TestDV01NullRisk()
        {
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

            Assert.AreEqual(0, finalRisk);
        }

        [TestMethod]
        public void TestSetBonds()
        {
            // Setup Date
            DateTime refDate = new DateTime(2022, 7, 8);
            QlBond.SetEvaluationDate(refDate);

            QlBond? otherBond = new(new DateTime(2049, 9, 1), new DateTime(2018, 9, 1), 3.85 / 100.0);
            HQlBondWrapper? hedger = new(otherBond)
            {
                Price = 102.79
            };

            IHedgeMethod hedgingMethod = HedgeFactory.GetHedger(HedgeType.DV01);
            int result = 1;
            try
            {
                hedgingMethod.SetHedgers(new() { hedger, hedger });
            }
            catch
            {
                result = 0;
            }

            Assert.IsTrue(result == 0);
        }

        [TestMethod]
        public void TestSetBondsInDV01Method()
        {
            // Setup Date
            DateTime refDate = new DateTime(2022, 7, 8);
            QlBond.SetEvaluationDate(refDate);

            QlBond? otherBond = new(new DateTime(2049, 9, 1), new DateTime(2018, 9, 1), 3.85 / 100.0);
            HQlBondWrapper? hedger = new(otherBond)
            {
                Price = 102.79
            };

            IHedgeMethod hedgingMethod = HedgeFactory.GetHedger(HedgeType.DV01);
            int result = 1;
            try
            {
                hedgingMethod.SetHedgers(new() { hedger });
            }
            catch
            {
                result = 0;
            }

            Assert.IsTrue(result == 1);
        }

        [TestMethod]
        public void TestSetBondsInDV01ConvexityMethod()
        {
            // Setup Date
            DateTime refDate = new DateTime(2022, 7, 8);
            QlBond.SetEvaluationDate(refDate);

            QlBond? otherBond = new(new DateTime(2049, 9, 1), new DateTime(2018, 9, 1), 3.85 / 100.0);
            HQlBondWrapper? hedger = new(otherBond)
            {
                Price = 102.79
            };

            IHedgeMethod hedgingMethod = HedgeFactory.GetHedger(HedgeType.DV01);
            int result = 1;
            try
            {
                hedgingMethod.SetHedgers(new() { hedger });
            }
            catch
            {
                result = 0;
            }

            Assert.IsTrue(result == 1);
        }
    }
}